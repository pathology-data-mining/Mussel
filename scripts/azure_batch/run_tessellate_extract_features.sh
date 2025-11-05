#!/bin/bash
#
# Azure Batch task script for running tessellate-extract-features
# This script runs on Azure Batch compute nodes to process whole-slide images
#
# Environment variables expected:
#   SLIDE_PATH - Path to the input slide file (can be s3:// URL)
#   OUTPUT_H5_PATH - Path for output HDF5 file (can be s3:// URL or local path)
#   OUTPUT_PT_PATH - Path for output PyTorch file (can be s3:// URL or local path)
#   INTERMEDIATE_H5_PATH - (Optional) Path for intermediate tile-level features (two-step aggregation)
#   AGGREGATION_METHOD - (Optional) Aggregation method: identity, mean, max, model (default: identity)
#   SLIDE_MODEL_TYPE - (Optional) Slide model type for aggregation_method=model
#   CLASSIFIER_PKL - (Optional) Path to classifier pickle file for filtering
#   CLASSIFIER_THRESHOLD - (Optional) Threshold for classifier (default: 0.75)
#   PREFILTER_MODEL_TYPE - Model type for pre-filter extraction (default: CTRANSPATH)
#   POSTFILTER_MODEL_TYPE - (Optional) Model type for post-filter extraction
#   POSTFILTER_MODEL_TYPES - (Optional) Comma-separated list of postfilter models to run sequentially
#   SEGMENT_THRESHOLD - Tissue segmentation threshold (default: 0)
#   PATCH_SIZE - Patch size in pixels (default: 256)
#   MPP - Microns per pixel (default: 0.5)
#   NUM_WORKERS - Number of workers (default: 4)
#   BATCH_SIZE - Batch size for feature extraction (default: 64)
#   USE_GPU - Whether to use GPU (default: true)
#   KEEP_INTERMEDIATE_FILES - Keep intermediate files (default: false)
#   HF_TOKEN - (Optional) HuggingFace token for gated models
#   AWS_ACCESS_KEY_ID - (Optional) AWS access key for S3
#   AWS_SECRET_ACCESS_KEY - (Optional) AWS secret key for S3
#   AWS_DEFAULT_REGION - (Optional) AWS region (default: us-east-1)

set -e
set -o pipefail

echo "============================================"
echo "Azure Batch Tessellate-Extract-Features Task"
echo "============================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Cleanup function to remove temporary files
cleanup() {
    local exit_code=$?
    if [ -n "$WORK_DIR" ] && [ -d "$WORK_DIR" ]; then
        log "Cleanup: Removing work directory: $WORK_DIR"
        rm -rf "$WORK_DIR" || log "Warning: Failed to remove work directory"
    fi
    # Don't exit here, let the script continue to its natural exit
}

# Set trap to ensure cleanup runs on exit (success, failure, or interruption)
trap cleanup EXIT INT TERM

# Check required environment variables
if [ -z "$SLIDE_PATH" ]; then
    log "ERROR: SLIDE_PATH environment variable is required"
    exit 1
fi

if [ -z "$OUTPUT_H5_PATH" ]; then
    log "ERROR: OUTPUT_H5_PATH environment variable is required"
    exit 1
fi

if [ -z "$OUTPUT_PT_PATH" ]; then
    log "ERROR: OUTPUT_PT_PATH environment variable is required"
    exit 1
fi

# Set defaults for optional parameters
CLASSIFIER_THRESHOLD=${CLASSIFIER_THRESHOLD:-0.75}
PREFILTER_MODEL_TYPE=${PREFILTER_MODEL_TYPE:-CTRANSPATH}
SEGMENT_THRESHOLD=${SEGMENT_THRESHOLD:-0}
PATCH_SIZE=${PATCH_SIZE:-256}
MPP=${MPP:-0.5}
NUM_WORKERS=${NUM_WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-64}
USE_GPU=${USE_GPU:-true}
KEEP_INTERMEDIATE_FILES=${KEEP_INTERMEDIATE_FILES:-false}
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
AGGREGATION_METHOD=${AGGREGATION_METHOD:-identity}

# S3 helper functions
is_s3_path() {
    [[ "$1" =~ ^s3:// ]]
}

download_from_s3() {
    local s3_path="$1"
    local local_path="$2"
    log "Downloading from S3: $s3_path -> $local_path"
    
    if command -v aws &> /dev/null; then
        aws s3 cp "$s3_path" "$local_path"
    else
        log "ERROR: aws CLI not found. Install with: pip install awscli"
        exit 1
    fi
}

upload_to_s3() {
    local local_path="$1"
    local s3_path="$2"
    log "Uploading to S3: $local_path -> $s3_path"
    
    if command -v aws &> /dev/null; then
        aws s3 cp "$local_path" "$s3_path"
    else
        log "ERROR: aws CLI not found. Install with: pip install awscli"
        exit 1
    fi
}

log "Configuration:"
log "  SLIDE_PATH: $SLIDE_PATH"
log "  OUTPUT_H5_PATH: $OUTPUT_H5_PATH"
log "  OUTPUT_PT_PATH: $OUTPUT_PT_PATH"
log "  CLASSIFIER_PKL: ${CLASSIFIER_PKL:-<not set>}"
log "  CLASSIFIER_THRESHOLD: $CLASSIFIER_THRESHOLD"
log "  PREFILTER_MODEL_TYPE: $PREFILTER_MODEL_TYPE"
log "  POSTFILTER_MODEL_TYPE: ${POSTFILTER_MODEL_TYPE:-<not set>}"
log "  SEGMENT_THRESHOLD: $SEGMENT_THRESHOLD"
log "  PATCH_SIZE: $PATCH_SIZE"
log "  MPP: $MPP"
log "  NUM_WORKERS: $NUM_WORKERS"
log "  BATCH_SIZE: $BATCH_SIZE"
log "  USE_GPU: $USE_GPU"
log "  KEEP_INTERMEDIATE_FILES: $KEEP_INTERMEDIATE_FILES"
echo ""

# Stage input slide from S3 if needed
ORIGINAL_SLIDE_PATH="$SLIDE_PATH"
if is_s3_path "$SLIDE_PATH"; then
    log "Slide is in S3, staging locally..."
    WORK_DIR="/tmp/mussel_work_$$"
    mkdir -p "$WORK_DIR"
    LOCAL_SLIDE_PATH="$WORK_DIR/$(basename "$SLIDE_PATH")"
    download_from_s3 "$SLIDE_PATH" "$LOCAL_SLIDE_PATH"
    SLIDE_PATH="$LOCAL_SLIDE_PATH"
    log "Slide staged to: $SLIDE_PATH"
fi

# Check if slide file exists
if [ ! -f "$SLIDE_PATH" ]; then
    log "ERROR: Slide file not found: $SLIDE_PATH"
    exit 1
fi

log "Slide file found: $SLIDE_PATH (size: $(du -h "$SLIDE_PATH" | cut -f1))"

# Set HuggingFace token if provided
if [ -n "$HF_TOKEN" ]; then
    export HUGGINGFACE_TOKEN="$HF_TOKEN"
    log "HuggingFace token set"
fi

# Check if GPU is available when USE_GPU=true
if [ "$USE_GPU" = "true" ]; then
    if command -v nvidia-smi &> /dev/null; then
        log "GPU information:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        log "WARNING: GPU requested but nvidia-smi not found"
    fi
fi

# Prepare output paths (use local temp paths if outputs are S3)
ORIGINAL_OUTPUT_H5_PATH="$OUTPUT_H5_PATH"
ORIGINAL_OUTPUT_PT_PATH="$OUTPUT_PT_PATH"
ORIGINAL_INTERMEDIATE_H5_PATH="$INTERMEDIATE_H5_PATH"

if is_s3_path "$OUTPUT_H5_PATH" || is_s3_path "$OUTPUT_PT_PATH" || is_s3_path "$INTERMEDIATE_H5_PATH"; then
    WORK_DIR="${WORK_DIR:-/tmp/mussel_work_$$}"
    mkdir -p "$WORK_DIR"
    
    if is_s3_path "$OUTPUT_H5_PATH"; then
        LOCAL_OUTPUT_H5_PATH="$WORK_DIR/$(basename "$OUTPUT_H5_PATH")"
        log "Will upload H5 output to S3: $OUTPUT_H5_PATH"
    else
        LOCAL_OUTPUT_H5_PATH="$OUTPUT_H5_PATH"
    fi
    
    if is_s3_path "$OUTPUT_PT_PATH"; then
        LOCAL_OUTPUT_PT_PATH="$WORK_DIR/$(basename "$OUTPUT_PT_PATH")"
        log "Will upload PT output to S3: $OUTPUT_PT_PATH"
    else
        LOCAL_OUTPUT_PT_PATH="$OUTPUT_PT_PATH"
    fi
    
    if [ -n "$INTERMEDIATE_H5_PATH" ] && is_s3_path "$INTERMEDIATE_H5_PATH"; then
        LOCAL_INTERMEDIATE_H5_PATH="$WORK_DIR/$(basename "$INTERMEDIATE_H5_PATH")"
        log "Will upload intermediate H5 (tile-level features) to S3: $INTERMEDIATE_H5_PATH"
    elif [ -n "$INTERMEDIATE_H5_PATH" ]; then
        LOCAL_INTERMEDIATE_H5_PATH="$INTERMEDIATE_H5_PATH"
    fi
else
    LOCAL_OUTPUT_H5_PATH="$OUTPUT_H5_PATH"
    LOCAL_OUTPUT_PT_PATH="$OUTPUT_PT_PATH"
    if [ -n "$INTERMEDIATE_H5_PATH" ]; then
        LOCAL_INTERMEDIATE_H5_PATH="$INTERMEDIATE_H5_PATH"
    fi
fi

# Create output directory if it doesn't exist (for local outputs)
OUTPUT_DIR=$(dirname "$LOCAL_OUTPUT_H5_PATH")
if [ ! -d "$OUTPUT_DIR" ]; then
    log "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Determine which postfilter models to run
if [ -n "$POSTFILTER_MODEL_TYPES" ]; then
    # Multiple models specified - use optimized two-command approach:
    # 1. Run filter-tessellate once (tessellate + prefilter + filter)
    # 2. Run extract-features for each postfilter model
    
    IFS=',' read -ra MODELS <<< "$POSTFILTER_MODEL_TYPES"
    log "Multi-model mode: Will run filter-tessellate once, then extract-features for ${#MODELS[@]} models: ${POSTFILTER_MODEL_TYPES}"
    
    # Step 1: Run filter-tessellate to get filtered coordinates
    log ""
    log "=========================================="
    log "Step 1: Running filter-tessellate (tessellation + prefiltering + filtering)"
    log "=========================================="
    log ""
    
    # Create temp directory for filtered coordinates
    FILTERED_COORDS_H5="$OUTPUT_DIR/filtered_coords.h5"
    FILTERED_FEATURES_PT="$OUTPUT_DIR/filtered_features.pt"
    
    FILTER_CMD_ARGS=(
        "filter_tessellate"
        "slide_path=$SLIDE_PATH"
        "output_h5_path=$FILTERED_COORDS_H5"
        "output_pt_path=$FILTERED_FEATURES_PT"
        "model_type=$PREFILTER_MODEL_TYPE"
        "classifier_pkl=$CLASSIFIER_PKL"
        "classifier_threshold=$CLASSIFIER_THRESHOLD"
        "seg_config.segment_threshold=$SEGMENT_THRESHOLD"
        "seg_config.patch_size=$PATCH_SIZE"
        "seg_config.mpp=$MPP"
        "num_workers=$NUM_WORKERS"
        "batch_size=$BATCH_SIZE"
        "use_gpu=$USE_GPU"
        "keep_intermediate_files=false"
    )
    
    log "Executing filter-tessellate command:"
    log "${FILTER_CMD_ARGS[*]}"
    echo ""
    
    START_TIME=$(date +%s)
    "${FILTER_CMD_ARGS[@]}"
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $EXIT_CODE -ne 0 ]; then
        log "ERROR: filter-tessellate failed with exit code $EXIT_CODE (duration: $DURATION seconds)"
        exit $EXIT_CODE
    fi
    
    log "SUCCESS: filter-tessellate completed in $DURATION seconds"
    log ""
    
    # Step 2: Run extract-features for each postfilter model
    MODEL_INDEX=0
    for MODEL in "${MODELS[@]}"; do
        MODEL_INDEX=$((MODEL_INDEX + 1))
        MODEL=$(echo "$MODEL" | xargs)  # Trim whitespace
        
        log ""
        log "=========================================="
        log "Step $((MODEL_INDEX + 1)): Extracting features with model $MODEL_INDEX/${#MODELS[@]}: $MODEL"
        log "=========================================="
        log ""
        
        # Create model-specific output directories
        MODEL_H5_DIR="$OUTPUT_DIR/$MODEL/h5"
        MODEL_PT_DIR="$OUTPUT_DIR/$MODEL/pt"
        mkdir -p "$MODEL_H5_DIR"
        mkdir -p "$MODEL_PT_DIR"
        
        SLIDE_NAME=$(basename "$SLIDE_PATH" | sed 's/\.[^.]*$//')
        MODEL_H5_PATH="$MODEL_H5_DIR/${SLIDE_NAME}_features.h5"
        MODEL_PT_PATH="$MODEL_PT_DIR/${SLIDE_NAME}_features.pt"
        
        # Handle intermediate path for aggregation
        if [ "$AGGREGATION_METHOD" != "identity" ]; then
            MODEL_TILE_H5_DIR="$OUTPUT_DIR/$MODEL/tile_h5"
            mkdir -p "$MODEL_TILE_H5_DIR"
            MODEL_INTERMEDIATE_H5="$MODEL_TILE_H5_DIR/${SLIDE_NAME}_tile_features.h5"
        else
            MODEL_INTERMEDIATE_H5=""
        fi
        
        EXTRACT_CMD_ARGS=(
            "extract_features"
            "patch_h5_path=$FILTERED_COORDS_H5"
            "slide_path=$SLIDE_PATH"
            "output_h5_path=$MODEL_H5_PATH"
            "output_pt_path=$MODEL_PT_PATH"
            "model_type=$MODEL"
            "batch_size=$BATCH_SIZE"
            "use_gpu=$USE_GPU"
            "num_workers=$NUM_WORKERS"
        )
        
        # Add aggregation parameters if specified
        if [ "$AGGREGATION_METHOD" != "identity" ]; then
            EXTRACT_CMD_ARGS+=("aggregation_method=$AGGREGATION_METHOD")
            EXTRACT_CMD_ARGS+=("intermediate_h5_path=$MODEL_INTERMEDIATE_H5")
        fi
        
        if [ -n "$SLIDE_MODEL_TYPE" ]; then
            EXTRACT_CMD_ARGS+=("slide_model_type=$SLIDE_MODEL_TYPE")
        fi
        
        log "Executing extract-features command for $MODEL:"
        log "${EXTRACT_CMD_ARGS[*]}"
        echo ""
        
        MODEL_START_TIME=$(date +%s)
        "${EXTRACT_CMD_ARGS[@]}"
        MODEL_EXIT_CODE=$?
        MODEL_END_TIME=$(date +%s)
        MODEL_DURATION=$((MODEL_END_TIME - MODEL_START_TIME))
        
        if [ $MODEL_EXIT_CODE -ne 0 ]; then
            log "ERROR: extract-features failed for model $MODEL with exit code $MODEL_EXIT_CODE (duration: $MODEL_DURATION seconds)"
            exit $MODEL_EXIT_CODE
        fi
        
        log "SUCCESS: Model $MODEL completed in $MODEL_DURATION seconds"
    done
    
    # Upload results to S3 if needed
    if is_s3_path "$ORIGINAL_OUTPUT_H5_PATH"; then
        S3_BASE=$(dirname "$ORIGINAL_OUTPUT_H5_PATH")
        
        for MODEL in "${MODELS[@]}"; do
            MODEL=$(echo "$MODEL" | xargs)  # Trim whitespace
            
            SLIDE_NAME=$(basename "$SLIDE_PATH" | sed 's/\.[^.]*$//')
            LOCAL_MODEL_H5="$OUTPUT_DIR/$MODEL/h5/${SLIDE_NAME}_features.h5"
            LOCAL_MODEL_PT="$OUTPUT_DIR/$MODEL/pt/${SLIDE_NAME}_features.pt"
            S3_MODEL_H5="$S3_BASE/$MODEL/h5/${SLIDE_NAME}_features.h5"
            S3_MODEL_PT="$S3_BASE/$MODEL/pt/${SLIDE_NAME}_features.pt"
            
            if [ -f "$LOCAL_MODEL_H5" ]; then
                log "Uploading $MODEL H5 file: $LOCAL_MODEL_H5 -> $S3_MODEL_H5"
                upload_to_s3 "$LOCAL_MODEL_H5" "$S3_MODEL_H5"
            fi
            
            if [ -f "$LOCAL_MODEL_PT" ]; then
                log "Uploading $MODEL PT file: $LOCAL_MODEL_PT -> $S3_MODEL_PT"
                upload_to_s3 "$LOCAL_MODEL_PT" "$S3_MODEL_PT"
            fi
            
            # Upload intermediate tile features if present
            if [ "$AGGREGATION_METHOD" != "identity" ]; then
                LOCAL_MODEL_INT="$OUTPUT_DIR/$MODEL/tile_h5/${SLIDE_NAME}_tile_features.h5"
                S3_MODEL_INT="$S3_BASE/$MODEL/tile_h5/${SLIDE_NAME}_tile_features.h5"
                
                if [ -f "$LOCAL_MODEL_INT" ]; then
                    log "Uploading $MODEL intermediate H5 file: $LOCAL_MODEL_INT -> $S3_MODEL_INT"
                    upload_to_s3 "$LOCAL_MODEL_INT" "$S3_MODEL_INT"
                fi
            fi
        done
    fi

elif [ -n "$POSTFILTER_MODEL_TYPE" ]; then
    # Single model specified - use tessellate-extract-features
    MODEL="$POSTFILTER_MODEL_TYPE"
else
    # No postfilter model specified - use prefilter model
    MODEL="$PREFILTER_MODEL_TYPE"
fi

# Single-model mode (backward compatible - uses tessellate-extract-features)
if [ -z "$POSTFILTER_MODEL_TYPES" ]; then
    MODEL_H5_PATH="$LOCAL_OUTPUT_H5_PATH"
    MODEL_PT_PATH="$LOCAL_OUTPUT_PT_PATH"
    MODEL_INTERMEDIATE_H5_PATH="$LOCAL_INTERMEDIATE_H5_PATH"
    
    # Build the command as an array for safe execution
    CMD_ARGS=(
        "tessellate_extract_features"
        "slide_path=$SLIDE_PATH"
        "output_h5_path=$MODEL_H5_PATH"
        "output_pt_path=$MODEL_PT_PATH"
        "prefilter_model_type=$PREFILTER_MODEL_TYPE"
        "seg_config.segment_threshold=$SEGMENT_THRESHOLD"
        "seg_config.patch_size=$PATCH_SIZE"
        "seg_config.mpp=$MPP"
        "num_workers=$NUM_WORKERS"
        "batch_size=$BATCH_SIZE"
        "use_gpu=$USE_GPU"
        "keep_intermediate_files=$KEEP_INTERMEDIATE_FILES"
    )

    # Add optional parameters
    if [ -n "$CLASSIFIER_PKL" ]; then
        CMD_ARGS+=("classifier_pkl=$CLASSIFIER_PKL")
        CMD_ARGS+=("classifier_threshold=$CLASSIFIER_THRESHOLD")
    fi

    # Add the specific postfilter model
    CMD_ARGS+=("postfilter_model_type=$MODEL")

    # Add aggregation parameters if specified
    if [ "$AGGREGATION_METHOD" != "identity" ]; then
        CMD_ARGS+=("aggregation_method=$AGGREGATION_METHOD")
    fi

    if [ -n "$MODEL_INTERMEDIATE_H5_PATH" ]; then
        CMD_ARGS+=("intermediate_h5_path=$MODEL_INTERMEDIATE_H5_PATH")
    fi

    if [ -n "$SLIDE_MODEL_TYPE" ]; then
        CMD_ARGS+=("slide_model_type=$SLIDE_MODEL_TYPE")
    fi

    log "Executing command:"
    log "${CMD_ARGS[*]}"
    echo ""

    # Execute the command
    START_TIME=$(date +%s)
    "${CMD_ARGS[@]}"
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    if [ $EXIT_CODE -ne 0 ]; then
        log "ERROR: Processing failed with exit code $EXIT_CODE (duration: $DURATION seconds)"
        exit $EXIT_CODE
    fi
    
    log "SUCCESS: Processing completed in $DURATION seconds"
    
    # Upload results to S3 if needed
    if is_s3_path "$ORIGINAL_OUTPUT_H5_PATH"; then
        if [ -f "$MODEL_H5_PATH" ]; then
            log "Local H5 file: $MODEL_H5_PATH (size: $(du -h "$MODEL_H5_PATH" | cut -f1))"
            upload_to_s3 "$MODEL_H5_PATH" "$ORIGINAL_OUTPUT_H5_PATH"
            log "Uploaded H5 file to S3: $ORIGINAL_OUTPUT_H5_PATH"
        fi
        
        if [ -f "$MODEL_PT_PATH" ]; then
            log "Local PT file: $MODEL_PT_PATH (size: $(du -h "$MODEL_PT_PATH" | cut -f1))"
            upload_to_s3 "$MODEL_PT_PATH" "$ORIGINAL_OUTPUT_PT_PATH"
            log "Uploaded PT file to S3: $ORIGINAL_OUTPUT_PT_PATH"
        fi
        
        if [ -n "$MODEL_INTERMEDIATE_H5_PATH" ] && [ -f "$MODEL_INTERMEDIATE_H5_PATH" ]; then
            log "Local intermediate H5 file: $MODEL_INTERMEDIATE_H5_PATH (size: $(du -h "$MODEL_INTERMEDIATE_H5_PATH" | cut -f1))"
            upload_to_s3 "$MODEL_INTERMEDIATE_H5_PATH" "$ORIGINAL_INTERMEDIATE_H5_PATH"
            log "Uploaded intermediate H5 file to S3: $ORIGINAL_INTERMEDIATE_H5_PATH"
        fi
    else
        # Local output
        if [ -f "$MODEL_H5_PATH" ]; then
            log "Output H5 file: $MODEL_H5_PATH (size: $(du -h "$MODEL_H5_PATH" | cut -f1))"
        fi
        if [ -f "$MODEL_PT_PATH" ]; then
            log "Output PT file: $MODEL_PT_PATH (size: $(du -h "$MODEL_PT_PATH" | cut -f1))"
        fi
        if [ -n "$MODEL_INTERMEDIATE_H5_PATH" ] && [ -f "$MODEL_INTERMEDIATE_H5_PATH" ]; then
            log "Intermediate H5 file: $MODEL_INTERMEDIATE_H5_PATH (size: $(du -h "$MODEL_INTERMEDIATE_H5_PATH" | cut -f1))"
        fi
    fi
fi

log ""
log "=========================================="
log "Processing completed successfully"
log "=========================================="

echo "============================================"
echo "End time: $(date)"
echo "============================================"

# Final cleanup happens via trap
exit 0
