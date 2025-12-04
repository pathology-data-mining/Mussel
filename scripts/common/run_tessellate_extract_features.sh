#!/bin/bash
#
# Task script for running tessellate-extract-features
# This script runs on distributed compute nodes to process whole-slide images
# Supports: Azure Batch, HTCondor, SLURM
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
#   PREFILTER_MODEL_TYPE - (Optional) Model type for pre-filter extraction
#   PREFILTER_MODEL_PATH - (Optional) Path to prefilter model weights (can be s3://)
#   MODEL_TYPE - (Optional) Model type for post-filter extraction (deprecated)
#   MODEL_PATH - (Optional) Path to model weights (can be s3://)
#   MODEL_TYPES - (Optional) Comma-separated list of tile encoder models for multi-model extraction
#   SLIDE_MODEL_TYPES - (Optional) Comma-separated list of slide encoder models for aggregation
#   SLIDE_MODEL_PATH - (Optional) Path to slide encoder model weights (can be s3://)
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
#   AWS_ENDPOINT_URL - (Optional) Custom S3 endpoint URL for S3-compatible storage (e.g., MinIO, Ceph)

set -e
set -o pipefail

echo "============================================"
echo "Tessellate-Extract-Features Task"
echo "============================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to get model-specific default patch size
get_model_patch_size() {
    local model_type="$1"
    case "$model_type" in
        CTRANSPATH|VIRCHOW|VIRCHOW2|OPTIMUS|CLIP|GOOGLEPATH)
            echo "224"
            ;;
        CONCH1_5|TITAN_SLIDE)
            echo "512"
            ;;
        *)
            echo "256"
            ;;
    esac
}

# Detect uv virtual environment
UV_PREFIX=""
if command -v uv >/dev/null 2>&1; then
    if [ -d ".venv" ] || [ -n "$VIRTUAL_ENV" ]; then
        UV_PREFIX="uv run"
        log "Detected uv environment - using 'uv run' for CLI commands"
    fi
fi

# Cleanup function to remove temporary files
cleanup_staging() {
    if [ -n "$WORK_DIR" ] && [ -d "$WORK_DIR" ]; then
        log "Cleanup: Removing work directory: $WORK_DIR"
        # Wait a bit to allow NFS locks to clear
        sleep 2
        # Try to remove, but don't fail if NFS locks remain
        rm -rf "$WORK_DIR" 2>/dev/null || {
            log "Warning: Failed to remove work directory cleanly (likely NFS locks)"
            log "Attempting background cleanup..."
            # Try async cleanup in background to not block
            (sleep 5 && rm -rf "$WORK_DIR" 2>/dev/null) &
        }
    fi
}

# NOTE: We do NOT set a trap for cleanup here because in Docker/Apptainer
# batch mode, child processes (multiprocessing workers) can exit before
# the main process is done, which would trigger premature cleanup of staged
# files. Instead, cleanup is called manually after processing completes.

# Check required environment variables
# Support both single-slide (SLIDE_PATH) and batch (SLIDE_PATHS) modes
if [ -z "$SLIDE_PATH" ] && [ -z "$SLIDE_PATHS" ]; then
    log "ERROR: Either SLIDE_PATH or SLIDE_PATHS environment variable is required"
    exit 1
fi

# Batch mode detection
if [ -n "$SLIDE_PATHS" ]; then
    BATCH_MODE=true
    log "Batch processing mode detected"
    
    if [ -z "$OUTPUT_DIR" ]; then
        log "ERROR: OUTPUT_DIR environment variable is required for batch processing"
        exit 1
    fi
else
    BATCH_MODE=false
    
    if [ -z "$OUTPUT_H5_PATH" ]; then
        log "ERROR: OUTPUT_H5_PATH environment variable is required"
        exit 1
    fi
    
    if [ -z "$OUTPUT_PT_PATH" ]; then
        log "ERROR: OUTPUT_PT_PATH environment variable is required"
        exit 1
    fi
fi

# Set defaults for optional parameters
CLASSIFIER_THRESHOLD=${CLASSIFIER_THRESHOLD:-0.75}
PREFILTER_MODEL_TYPE=${PREFILTER_MODEL_TYPE:-}
SEGMENT_THRESHOLD=${SEGMENT_THRESHOLD:-0}
# PATCH_SIZE - Let Python code apply model-specific defaults (no default here)
PATCH_SIZE=${PATCH_SIZE:-}
MPP=${MPP:-0.5}
NUM_WORKERS=${NUM_WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-64}
SLIDE_BATCH_SIZE=${SLIDE_BATCH_SIZE:-8}
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
        # Add custom endpoint URL if specified
        if [ -n "$AWS_ENDPOINT_URL" ]; then
            aws s3 --endpoint-url "$AWS_ENDPOINT_URL" cp "$s3_path" "$local_path"
        else
            aws s3 cp "$s3_path" "$local_path"
        fi
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
        # Add custom endpoint URL if specified
        if [ -n "$AWS_ENDPOINT_URL" ]; then
            aws s3 --endpoint-url "$AWS_ENDPOINT_URL" cp "$local_path" "$s3_path"
        else
            aws s3 cp "$local_path" "$s3_path"
        fi
    else
        log "ERROR: aws CLI not found. Install with: pip install awscli"
        exit 1
    fi
}

log "Configuration:"
if [ "$BATCH_MODE" = true ]; then
    log "  Mode: BATCH"
    log "  SLIDE_PATHS: $SLIDE_PATHS"
    log "  SLIDE_IDS: ${SLIDE_IDS:-<auto-generated>}"
    log "  OUTPUT_DIR: $OUTPUT_DIR"
    log "  SLIDE_BATCH_SIZE: $SLIDE_BATCH_SIZE"
else
    log "  Mode: SINGLE"
    log "  SLIDE_PATH: $SLIDE_PATH"
    log "  OUTPUT_H5_PATH: $OUTPUT_H5_PATH"
    log "  OUTPUT_PT_PATH: $OUTPUT_PT_PATH"
fi
log "  CLASSIFIER_PKL: ${CLASSIFIER_PKL:-<not set>}"
log "  CLASSIFIER_THRESHOLD: $CLASSIFIER_THRESHOLD"
log "  PREFILTER_MODEL_TYPE: $PREFILTER_MODEL_TYPE"
log "  PREFILTER_MODEL_PATH: ${PREFILTER_MODEL_PATH:-<not set>}"
log "  MODEL_TYPE: ${MODEL_TYPE:-<not set>}"
log "  MODEL_PATH: ${MODEL_PATH:-<not set>}"
log "  AGGREGATION_METHOD: $AGGREGATION_METHOD"
log "  SLIDE_MODEL_TYPE: ${SLIDE_MODEL_TYPE:-<not set>}"
log "  SLIDE_MODEL_TYPES: ${SLIDE_MODEL_TYPES:-<not set>}"
log "  SLIDE_MODEL_PATH: ${SLIDE_MODEL_PATH:-<not set>}"
log "  SEGMENT_THRESHOLD: $SEGMENT_THRESHOLD"
log "  PATCH_SIZE: $PATCH_SIZE"
log "  MPP: $MPP"
log "  NUM_WORKERS: $NUM_WORKERS"
log "  BATCH_SIZE: $BATCH_SIZE"
log "  USE_GPU: $USE_GPU"
log "  KEEP_INTERMEDIATE_FILES: $KEEP_INTERMEDIATE_FILES"
echo ""

# Stage input slide from S3 if needed (single-slide mode only)
if [ "$BATCH_MODE" = false ]; then
    ORIGINAL_SLIDE_PATH="$SLIDE_PATH"
    if is_s3_path "$SLIDE_PATH"; then
        log "Slide is in S3, staging locally..."
        WORK_DIR="${TMPDIR:-$HOME/tmp}/mussel_work_$$"
        mkdir -p "$WORK_DIR"
        LOCAL_SLIDE_PATH="$WORK_DIR/$(basename "$SLIDE_PATH")"
        download_from_s3 "$SLIDE_PATH" "$LOCAL_SLIDE_PATH"
        SLIDE_PATH="$LOCAL_SLIDE_PATH"
        log "Slide staged to: $SLIDE_PATH"
    fi

    # Check if slide file exists
    if [ ! -f "$SLIDE_PATH" ]; then
        log "ERROR: Slide file not found: $SLIDE_PATH"
        cleanup_staging
        exit 1
    fi

    log "Slide file found: $SLIDE_PATH (size: $(du -h "$SLIDE_PATH" | cut -f1))"
fi

# Stage model files from S3 if needed
if [ -n "$PREFILTER_MODEL_PATH" ] && is_s3_path "$PREFILTER_MODEL_PATH"; then
    log "Prefilter model is in S3, staging locally..."
    WORK_DIR="${WORK_DIR:-${TMPDIR:-$HOME/tmp}/mussel_work_$$}"
    mkdir -p "$WORK_DIR"
    LOCAL_PREFILTER_MODEL_PATH="$WORK_DIR/$(basename "$PREFILTER_MODEL_PATH")"
    download_from_s3 "$PREFILTER_MODEL_PATH" "$LOCAL_PREFILTER_MODEL_PATH"
    PREFILTER_MODEL_PATH="$LOCAL_PREFILTER_MODEL_PATH"
    log "Prefilter model staged to: $PREFILTER_MODEL_PATH"
fi

if [ -n "$MODEL_PATH" ] && is_s3_path "$MODEL_PATH"; then
    log "Postfilter model is in S3, staging locally..."
    WORK_DIR="${WORK_DIR:-${TMPDIR:-$HOME/tmp}/mussel_work_$$}"
    mkdir -p "$WORK_DIR"
    LOCAL_MODEL_PATH="$WORK_DIR/$(basename "$MODEL_PATH")"
    download_from_s3 "$MODEL_PATH" "$LOCAL_MODEL_PATH"
    MODEL_PATH="$LOCAL_MODEL_PATH"
    log "Postfilter model staged to: $MODEL_PATH"
fi

if [ -n "$SLIDE_MODEL_PATH" ] && is_s3_path "$SLIDE_MODEL_PATH"; then
    log "Slide model is in S3, staging locally..."
    WORK_DIR="${WORK_DIR:-${TMPDIR:-$HOME/tmp}/mussel_work_$$}"
    mkdir -p "$WORK_DIR"
    LOCAL_SLIDE_MODEL_PATH="$WORK_DIR/$(basename "$SLIDE_MODEL_PATH")"
    download_from_s3 "$SLIDE_MODEL_PATH" "$LOCAL_SLIDE_MODEL_PATH"
    SLIDE_MODEL_PATH="$LOCAL_SLIDE_MODEL_PATH"
    log "Slide model staged to: $SLIDE_MODEL_PATH"
fi

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
    WORK_DIR="${WORK_DIR:-${TMPDIR:-$HOME/tmp}/mussel_work_$$}"
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

# Handle batch processing mode
if [ "$BATCH_MODE" = true ]; then
    log ""
    log "=========================================="
    log "Batch Processing Mode"
    log "=========================================="
    log "Slides: $SLIDE_PATHS"
    log "Slide IDs: ${SLIDE_IDS:-auto-generated}"
    log "Output directory: $OUTPUT_DIR"
    log "Slide batch size: $SLIDE_BATCH_SIZE"
    log ""
    
    # Stage slides from S3 if needed
    # Parse comma-separated slide paths
    IFS=',' read -ra SLIDE_PATH_ARRAY <<< "$SLIDE_PATHS"
    LOCAL_SLIDE_PATHS=()
    NEEDS_STAGING=false
    
    for slide_path in "${SLIDE_PATH_ARRAY[@]}"; do
        if is_s3_path "$slide_path"; then
            NEEDS_STAGING=true
            break
        fi
    done
    
    if [ "$NEEDS_STAGING" = true ]; then
        log "One or more slides are in S3, staging locally..."
        WORK_DIR="${TMPDIR:-$HOME/tmp}/mussel_work_$$"
        mkdir -p "$WORK_DIR"
        
        for slide_path in "${SLIDE_PATH_ARRAY[@]}"; do
            if is_s3_path "$slide_path"; then
                local_slide_path="$WORK_DIR/$(basename "$slide_path")"
                download_from_s3 "$slide_path" "$local_slide_path"
                LOCAL_SLIDE_PATHS+=("$local_slide_path")
                log "  Staged: $slide_path -> $local_slide_path"
            else
                LOCAL_SLIDE_PATHS+=("$slide_path")
            fi
        done
        
        # Reconstruct slide paths string
        SLIDE_PATHS=$(IFS=,; echo "${LOCAL_SLIDE_PATHS[*]}")
        log "Updated slide paths after staging: $SLIDE_PATHS"
    fi
    
    # Multi-model batch mode: Pass all models directly to CLI
    # The tessellate_extract_features CLI handles multi-model processing efficiently
    if [ -n "$MODEL_TYPES" ] || [ -n "$SLIDE_MODEL_TYPES" ]; then
        log "Multi-model batch mode: Passing all models to tessellate_extract_features CLI"
        log ""
        
        MODEL_CMD_ARGS=(
            "tessellate_extract_features"
            "slide_paths=[${SLIDE_PATHS}]"
            "output_dir=${OUTPUT_DIR}"
            "num_workers=${NUM_WORKERS}"
            "batch_size=${BATCH_SIZE}"
            "slide_batch_size=${SLIDE_BATCH_SIZE}"
            "use_gpu=${USE_GPU}"
            "hydra.run.dir=${TMPDIR:-$HOME/tmp}/hydra_$$"
            "hydra.output_subdir=null"
        )
        
        # Add model_dir if specified
        if [ -n "$MODEL_DIR" ]; then
            MODEL_CMD_ARGS+=("model_dir=${MODEL_DIR}")
        fi
        
        # Add model types as a list
        if [ -n "$MODEL_TYPES" ]; then
            MODEL_CMD_ARGS+=("model_type=[${MODEL_TYPES}]")
        fi
        
        # Add slide model types as a list
        if [ -n "$SLIDE_MODEL_TYPES" ]; then
            MODEL_CMD_ARGS+=("slide_model_type=[${SLIDE_MODEL_TYPES}]")
        fi
        
        # Add aggregation method
        if [ "$AGGREGATION_METHOD" != "identity" ]; then
            MODEL_CMD_ARGS+=("aggregation_method=${AGGREGATION_METHOD}")
        fi
        
        # Add prefilter model type if specified
        if [ -n "$PREFILTER_MODEL_TYPE" ]; then
            MODEL_CMD_ARGS+=("prefilter_model_type=${PREFILTER_MODEL_TYPE}")
        fi
        
        # Add seg_config group if specified
        if [ -n "$SEG_CONFIG_GROUP" ]; then
            MODEL_CMD_ARGS+=("seg_config=${SEG_CONFIG_GROUP}")
        fi
        
        # Add seg_config parameters
        if [ -n "$PATCH_SIZE" ]; then
            MODEL_CMD_ARGS+=("seg_config.patch_size=${PATCH_SIZE}")
        fi
        if [ -n "$SEGMENT_THRESHOLD" ]; then
            MODEL_CMD_ARGS+=("seg_config.segment_threshold=${SEGMENT_THRESHOLD}")
        fi
        if [ -n "$STEP_SIZE" ]; then
            MODEL_CMD_ARGS+=("seg_config.step_size=${STEP_SIZE}")
        fi
        if [ -n "$MPP" ]; then
            MODEL_CMD_ARGS+=("seg_config.mpp=${MPP}")
        fi
        if [ -n "$SEG_LEVEL" ]; then
            MODEL_CMD_ARGS+=("seg_config.seg_level=${SEG_LEVEL}")
        fi
        if [ -n "$SEGMENT_MAX_VALUE" ]; then
            MODEL_CMD_ARGS+=("seg_config.segment_max_value=${SEGMENT_MAX_VALUE}")
        fi
        if [ -n "$MEDIAN_BLUR_KSIZE" ]; then
            MODEL_CMD_ARGS+=("seg_config.median_blur_ksize=${MEDIAN_BLUR_KSIZE}")
        fi
        if [ -n "$MORPHOLOGY_EX_KERNEL" ]; then
            MODEL_CMD_ARGS+=("seg_config.morphology_ex_kernel=${MORPHOLOGY_EX_KERNEL}")
        fi
        if [ -n "$REF_PATCH_SIZE" ]; then
            MODEL_CMD_ARGS+=("seg_config.ref_patch_size=${REF_PATCH_SIZE}")
        fi
        if [ -n "$USE_OTSU" ]; then
            MODEL_CMD_ARGS+=("seg_config.use_otsu=${USE_OTSU}")
        fi
        if [ -n "$TISSUE_AREA_THRESHOLD" ]; then
            MODEL_CMD_ARGS+=("seg_config.tissue_area_threshold=${TISSUE_AREA_THRESHOLD}")
        fi
        if [ -n "$HOLE_AREA_THRESHOLD" ]; then
            MODEL_CMD_ARGS+=("seg_config.hole_area_threshold=${HOLE_AREA_THRESHOLD}")
        fi
        if [ -n "$MAX_NUM_HOLES" ]; then
            MODEL_CMD_ARGS+=("seg_config.max_num_holes=${MAX_NUM_HOLES}")
        fi
        
        # Add slide IDs if provided
        if [ -n "$SLIDE_IDS" ]; then
            MODEL_CMD_ARGS+=("slide_ids=[${SLIDE_IDS}]")
        fi
        
        # Add model paths if specified
        if [ -n "$PREFILTER_MODEL_PATH" ]; then
            MODEL_CMD_ARGS+=("prefilter_model_path=$PREFILTER_MODEL_PATH")
        fi
        if [ -n "$MODEL_PATH" ]; then
            MODEL_CMD_ARGS+=("model_path=$MODEL_PATH")
        fi
        if [ -n "$SLIDE_MODEL_PATH" ]; then
            MODEL_CMD_ARGS+=("slide_model_path=$SLIDE_MODEL_PATH")
        fi
        
        # Add classifier if specified
        if [ -n "$CLASSIFIER_PKL" ]; then
            MODEL_CMD_ARGS+=("classifier_pkl=$CLASSIFIER_PKL")
            MODEL_CMD_ARGS+=("classifier_threshold=$CLASSIFIER_THRESHOLD")
        fi
        
        log "Executing multi-model command:"
        log "${MODEL_CMD_ARGS[*]}"
        echo ""
        
        START_TIME=$(date +%s)
        EXIT_CODE=0
        ${UV_PREFIX} "${MODEL_CMD_ARGS[@]}" || {
            EXIT_CODE=$?
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            log "WARNING: Multi-model processing failed with exit code $EXIT_CODE (duration: $DURATION seconds)"
            log "Continuing with remaining processing..."
        }
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        if [ $EXIT_CODE -eq 0 ]; then
            log "SUCCESS: Multi-model processing completed in $DURATION seconds"
        fi
        log "=========================================="
        
        echo "============================================"
        echo "End time: $(date)"
        echo "============================================"
        
        exit 0
    fi
    
    # Single model mode (legacy)
    if [ -n "$MODEL_TYPE" ] || [ -n "$SLIDE_MODEL_TYPE" ]; then
        log "Single model mode: Processing one model at a time"
        log ""
        
        MODEL_CMD_ARGS=(
            "tessellate_extract_features"
            "slide_paths=[${SLIDE_PATHS}]"
            "output_dir=${OUTPUT_DIR}"
            "num_workers=${NUM_WORKERS}"
            "batch_size=${BATCH_SIZE}"
            "slide_batch_size=${SLIDE_BATCH_SIZE}"
            "use_gpu=${USE_GPU}"
        )
        
        # Add model_dir if specified
        if [ -n "$MODEL_DIR" ]; then
            MODEL_CMD_ARGS+=("model_dir=${MODEL_DIR}")
        fi
        
        if [ -n "$MODEL_TYPE" ]; then
            MODEL_CMD_ARGS+=("model_type=${MODEL_TYPE}")
        fi
        if [ -n "$SLIDE_MODEL_TYPE" ]; then
            MODEL_CMD_ARGS+=("slide_model_type=${SLIDE_MODEL_TYPE}")
        fi
        
        # Add aggregation method
        if [ "$AGGREGATION_METHOD" != "identity" ]; then
            MODEL_CMD_ARGS+=("aggregation_method=${AGGREGATION_METHOD}")
        fi
        
        # Add prefilter model type if specified
        if [ -n "$PREFILTER_MODEL_TYPE" ]; then
            MODEL_CMD_ARGS+=("prefilter_model_type=${PREFILTER_MODEL_TYPE}")
        fi
        
        # Add seg_config group if specified
        if [ -n "$SEG_CONFIG_GROUP" ]; then
            MODEL_CMD_ARGS+=("seg_config=${SEG_CONFIG_GROUP}")
        fi
        
        # Add seg_config parameters
        if [ -n "$PATCH_SIZE" ]; then
            MODEL_CMD_ARGS+=("seg_config.patch_size=${PATCH_SIZE}")
        fi
        if [ -n "$SEGMENT_THRESHOLD" ]; then
            MODEL_CMD_ARGS+=("seg_config.segment_threshold=${SEGMENT_THRESHOLD}")
        fi
        if [ -n "$STEP_SIZE" ]; then
            MODEL_CMD_ARGS+=("seg_config.step_size=${STEP_SIZE}")
        fi
        if [ -n "$MPP" ]; then
            MODEL_CMD_ARGS+=("seg_config.mpp=${MPP}")
        fi
        if [ -n "$SEG_LEVEL" ]; then
            MODEL_CMD_ARGS+=("seg_config.seg_level=${SEG_LEVEL}")
        fi
        if [ -n "$SEGMENT_MAX_VALUE" ]; then
            MODEL_CMD_ARGS+=("seg_config.segment_max_value=${SEGMENT_MAX_VALUE}")
        fi
        if [ -n "$MEDIAN_BLUR_KSIZE" ]; then
            MODEL_CMD_ARGS+=("seg_config.median_blur_ksize=${MEDIAN_BLUR_KSIZE}")
        fi
        if [ -n "$MORPHOLOGY_EX_KERNEL" ]; then
            MODEL_CMD_ARGS+=("seg_config.morphology_ex_kernel=${MORPHOLOGY_EX_KERNEL}")
        fi
        if [ -n "$REF_PATCH_SIZE" ]; then
            MODEL_CMD_ARGS+=("seg_config.ref_patch_size=${REF_PATCH_SIZE}")
        fi
        if [ -n "$USE_OTSU" ]; then
            MODEL_CMD_ARGS+=("seg_config.use_otsu=${USE_OTSU}")
        fi
        if [ -n "$TISSUE_AREA_THRESHOLD" ]; then
            MODEL_CMD_ARGS+=("seg_config.tissue_area_threshold=${TISSUE_AREA_THRESHOLD}")
        fi
        if [ -n "$HOLE_AREA_THRESHOLD" ]; then
            MODEL_CMD_ARGS+=("seg_config.hole_area_threshold=${HOLE_AREA_THRESHOLD}")
        fi
        if [ -n "$MAX_NUM_HOLES" ]; then
            MODEL_CMD_ARGS+=("seg_config.max_num_holes=${MAX_NUM_HOLES}")
        fi
        
        # Add slide IDs if provided
        if [ -n "$SLIDE_IDS" ]; then
            MODEL_CMD_ARGS+=("slide_ids=[${SLIDE_IDS}]")
        fi
        
        # Add model paths if specified
        if [ -n "$PREFILTER_MODEL_PATH" ]; then
            MODEL_CMD_ARGS+=("prefilter_model_path=$PREFILTER_MODEL_PATH")
        fi
        if [ -n "$MODEL_PATH" ]; then
            MODEL_CMD_ARGS+=("model_path=$MODEL_PATH")
        fi
        if [ -n "$SLIDE_MODEL_PATH" ]; then
            MODEL_CMD_ARGS+=("slide_model_path=$SLIDE_MODEL_PATH")
        fi
        
        # Add classifier if specified
        if [ -n "$CLASSIFIER_PKL" ]; then
            MODEL_CMD_ARGS+=("classifier_pkl=$CLASSIFIER_PKL")
            MODEL_CMD_ARGS+=("classifier_threshold=$CLASSIFIER_THRESHOLD")
        fi
        
        log "Executing single model command:"
        log "${MODEL_CMD_ARGS[*]}"
        echo ""
        
        START_TIME=$(date +%s)
        EXIT_CODE=0
        ${UV_PREFIX} "${MODEL_CMD_ARGS[@]}" || {
            EXIT_CODE=$?
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            log "WARNING: Processing failed with exit code $EXIT_CODE (duration: $DURATION seconds)"
            log "Continuing with remaining processing..."
        }
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        if [ $EXIT_CODE -eq 0 ]; then
            log "SUCCESS: Processing completed in $DURATION seconds"
        fi
        log "=========================================="
        
        echo "============================================"
        echo "End time: $(date)"
        echo "============================================"
        
        exit 0
    fi
    
    # Default batch processing (no model specified)
    # Build command for batch processing
    # Note: slide_paths parameter uses Hydra list syntax: slide_paths=[item1,item2,...]
    CMD_ARGS=(
        "tessellate_extract_features"
        "slide_paths=[${SLIDE_PATHS}]"
        "output_dir=${OUTPUT_DIR}"
        "num_workers=${NUM_WORKERS}"
        "batch_size=${BATCH_SIZE}"
        "slide_batch_size=${SLIDE_BATCH_SIZE}"
        "use_gpu=${USE_GPU}"
    )
    
    # Add model_dir if specified
    if [ -n "$MODEL_DIR" ]; then
        CMD_ARGS+=("model_dir=${MODEL_DIR}")
    fi
    
    # Add prefilter model type if specified
    if [ -n "$PREFILTER_MODEL_TYPE" ]; then
        CMD_ARGS+=("prefilter_model_type=${PREFILTER_MODEL_TYPE}")
    fi
    
    # Add seg_config group if specified
    if [ -n "$SEG_CONFIG_GROUP" ]; then
        CMD_ARGS+=("seg_config=${SEG_CONFIG_GROUP}")
    fi
    
    # Add individual SegConfig parameters if specified (these override group defaults)
    if [ -n "$SEGMENT_THRESHOLD" ]; then
        CMD_ARGS+=("seg_config.segment_threshold=${SEGMENT_THRESHOLD}")
    fi
    if [ -n "$PATCH_SIZE" ]; then
        CMD_ARGS+=("seg_config.patch_size=${PATCH_SIZE}")
    fi
    if [ -n "$STEP_SIZE" ]; then
        CMD_ARGS+=("seg_config.step_size=${STEP_SIZE}")
    fi
    if [ -n "$MPP" ]; then
        CMD_ARGS+=("seg_config.mpp=${MPP}")
    fi
    if [ -n "$SEG_LEVEL" ]; then
        CMD_ARGS+=("seg_config.seg_level=${SEG_LEVEL}")
    fi
    if [ -n "$SEGMENT_MAX_VALUE" ]; then
        CMD_ARGS+=("seg_config.segment_max_value=${SEGMENT_MAX_VALUE}")
    fi
    if [ -n "$MEDIAN_BLUR_KSIZE" ]; then
        CMD_ARGS+=("seg_config.median_blur_ksize=${MEDIAN_BLUR_KSIZE}")
    fi
    if [ -n "$MORPHOLOGY_EX_KERNEL" ]; then
        CMD_ARGS+=("seg_config.morphology_ex_kernel=${MORPHOLOGY_EX_KERNEL}")
    fi
    if [ -n "$REF_PATCH_SIZE" ]; then
        CMD_ARGS+=("seg_config.ref_patch_size=${REF_PATCH_SIZE}")
    fi
    if [ -n "$USE_OTSU" ]; then
        CMD_ARGS+=("seg_config.use_otsu=${USE_OTSU}")
    fi
    if [ -n "$TISSUE_AREA_THRESHOLD" ]; then
        CMD_ARGS+=("seg_config.tissue_area_threshold=${TISSUE_AREA_THRESHOLD}")
    fi
    if [ -n "$HOLE_AREA_THRESHOLD" ]; then
        CMD_ARGS+=("seg_config.hole_area_threshold=${HOLE_AREA_THRESHOLD}")
    fi
    if [ -n "$MAX_NUM_HOLES" ]; then
        CMD_ARGS+=("seg_config.max_num_holes=${MAX_NUM_HOLES}")
    fi
    
    # Add slide IDs if provided
    if [ -n "$SLIDE_IDS" ]; then
        CMD_ARGS+=("slide_ids=[${SLIDE_IDS}]")
    fi
    
    # Add prefilter model path if specified
    if [ -n "$PREFILTER_MODEL_PATH" ]; then
        CMD_ARGS+=("prefilter_model_path=$PREFILTER_MODEL_PATH")
    fi
    
    # Add classifier if specified
    if [ -n "$CLASSIFIER_PKL" ]; then
        CMD_ARGS+=("classifier_pkl=$CLASSIFIER_PKL")
        CMD_ARGS+=("classifier_threshold=$CLASSIFIER_THRESHOLD")
    fi
    
    # Add model if specified
    if [ -n "$MODEL_TYPE" ]; then
        CMD_ARGS+=("model_type=$MODEL_TYPE")
    fi
    
    if [ -n "$MODEL_PATH" ]; then
        CMD_ARGS+=("model_path=$MODEL_PATH")
    fi
    
    # Add aggregation parameters if specified
    if [ "$AGGREGATION_METHOD" != "identity" ]; then
        CMD_ARGS+=("aggregation_method=$AGGREGATION_METHOD")
    fi
    
    if [ -n "$SLIDE_MODEL_TYPE" ]; then
        CMD_ARGS+=("slide_model_type=$SLIDE_MODEL_TYPE")
    fi
    
    if [ -n "$SLIDE_MODEL_PATH" ]; then
        CMD_ARGS+=("slide_model_path=$SLIDE_MODEL_PATH")
    fi
    
    log "Executing batch processing command:"
    log "${CMD_ARGS[*]}"
    echo ""
    
    # Execute the command
    START_TIME=$(date +%s)
    ${UV_PREFIX} "${CMD_ARGS[@]}"
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    if [ $EXIT_CODE -ne 0 ]; then
        log "ERROR: Batch processing failed with exit code $EXIT_CODE (duration: $DURATION seconds)"
        cleanup_staging
        exit $EXIT_CODE
    fi
    
    log "SUCCESS: Batch processing completed in $DURATION seconds"
    
    # Note: The tessellate_extract_features CLI handles S3 output paths natively
    # No additional upload logic needed here
    
    log ""
    log "=========================================="
    log "Batch processing completed successfully"
    log "=========================================="
    
    echo "============================================"
    echo "End time: $(date)"
    echo "============================================"
    
    # Clean up staged files after successful processing
    cleanup_staging
    
    exit 0
fi

# Single-slide processing mode (original behavior)
# Create output directory if it doesn't exist (for local outputs)
OUTPUT_DIR=$(dirname "$LOCAL_OUTPUT_H5_PATH")
if [ ! -d "$OUTPUT_DIR" ]; then
    log "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Determine which models to run
if [ -n "$MODEL_TYPES" ]; then
    # Multiple models specified
    IFS=',' read -ra MODELS <<< "$MODEL_TYPES"
    log "Multi-model mode: Will process ${#MODELS[@]} models: ${MODEL_TYPES}"
    
    # Check if we need to run filter-tessellate first (only if prefilter is specified)
    if [ -n "$PREFILTER_MODEL_TYPE" ]; then
        # Use optimized two-command approach:
        # 1. Run filter-tessellate once (tessellate + prefilter + filter)
        # 2. Run extract-features for each model
        
        log "Prefilter specified: Will run filter-tessellate once, then extract-features for ${#MODELS[@]} models"
        
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
            "classifier_pkl=$CLASSIFIER_PKL"
            "classifier_threshold=$CLASSIFIER_THRESHOLD"
            "num_workers=$NUM_WORKERS"
            "batch_size=$BATCH_SIZE"
            "use_gpu=$USE_GPU"
            "keep_intermediate_files=false"
        )
        
        # Only add model_type if prefilter model is specified
        if [ -n "$PREFILTER_MODEL_TYPE" ]; then
            FILTER_CMD_ARGS+=("model_type=$PREFILTER_MODEL_TYPE")
        fi
        USE_FILTERED_COORDS=true
    else
        # No prefilter - just run tessellate and then extract features
        log "No prefilter specified: Will run tessellate once, then extract-features for ${#MODELS[@]} models"
        USE_FILTERED_COORDS=false
    fi
    
    # Add seg_config group if specified
    if [ -n "$SEG_CONFIG_GROUP" ]; then
        FILTER_CMD_ARGS+=("seg_config=${SEG_CONFIG_GROUP}")
    fi
    
    # Add individual SegConfig parameters if specified (these override group defaults)
    if [ -n "$SEGMENT_THRESHOLD" ]; then
        FILTER_CMD_ARGS+=("seg_config.segment_threshold=$SEGMENT_THRESHOLD")
    fi
    if [ -n "$PATCH_SIZE" ]; then
        FILTER_CMD_ARGS+=("seg_config.patch_size=$PATCH_SIZE")
    fi
    if [ -n "$STEP_SIZE" ]; then
        FILTER_CMD_ARGS+=("seg_config.step_size=$STEP_SIZE")
    fi
    if [ -n "$MPP" ]; then
        FILTER_CMD_ARGS+=("seg_config.mpp=$MPP")
    fi
    if [ -n "$SEG_LEVEL" ]; then
        FILTER_CMD_ARGS+=("seg_config.seg_level=$SEG_LEVEL")
    fi
    if [ -n "$SEGMENT_MAX_VALUE" ]; then
        FILTER_CMD_ARGS+=("seg_config.segment_max_value=$SEGMENT_MAX_VALUE")
    fi
    if [ -n "$MEDIAN_BLUR_KSIZE" ]; then
        FILTER_CMD_ARGS+=("seg_config.median_blur_ksize=$MEDIAN_BLUR_KSIZE")
    fi
    if [ -n "$MORPHOLOGY_EX_KERNEL" ]; then
        FILTER_CMD_ARGS+=("seg_config.morphology_ex_kernel=$MORPHOLOGY_EX_KERNEL")
    fi
    if [ -n "$REF_PATCH_SIZE" ]; then
        FILTER_CMD_ARGS+=("seg_config.ref_patch_size=$REF_PATCH_SIZE")
    fi
    if [ -n "$USE_OTSU" ]; then
        FILTER_CMD_ARGS+=("seg_config.use_otsu=$USE_OTSU")
    fi
    if [ -n "$TISSUE_AREA_THRESHOLD" ]; then
        FILTER_CMD_ARGS+=("seg_config.tissue_area_threshold=$TISSUE_AREA_THRESHOLD")
    fi
    if [ -n "$HOLE_AREA_THRESHOLD" ]; then
        FILTER_CMD_ARGS+=("seg_config.hole_area_threshold=$HOLE_AREA_THRESHOLD")
    fi
    if [ -n "$MAX_NUM_HOLES" ]; then
        FILTER_CMD_ARGS+=("seg_config.max_num_holes=$MAX_NUM_HOLES")
    fi
    
    # Add model_path if specified
    if [ -n "$PREFILTER_MODEL_PATH" ]; then
        FILTER_CMD_ARGS+=("model_path=$PREFILTER_MODEL_PATH")
    fi
    
    log "Executing filter-tessellate command:"
    log "${FILTER_CMD_ARGS[*]}"
    echo ""
    
    START_TIME=$(date +%s)
    ${UV_PREFIX} "${FILTER_CMD_ARGS[@]}"
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $EXIT_CODE -ne 0 ]; then
        log "ERROR: filter-tessellate failed with exit code $EXIT_CODE (duration: $DURATION seconds)"
        cleanup_staging
        exit $EXIT_CODE
    fi
    
    log "SUCCESS: filter-tessellate completed in $DURATION seconds"
    log ""
    
    # Step 2: Run extract-features for each model
    MODEL_INDEX=0
    for MODEL in "${MODELS[@]}"; do
        MODEL_INDEX=$((MODEL_INDEX + 1))
        MODEL=$(echo "$MODEL" | xargs)  # Trim whitespace
        
        log ""
        log "=========================================="
        if [ "$USE_FILTERED_COORDS" = true ]; then
            log "Step $((MODEL_INDEX + 1)): Extracting features with model $MODEL_INDEX/${#MODELS[@]}: $MODEL"
        else
            log "Step $MODEL_INDEX: Processing with model $MODEL_INDEX/${#MODELS[@]}: $MODEL"
        fi
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
        
        if [ "$USE_FILTERED_COORDS" = true ]; then
            # Use extract-features with filtered coordinates
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
        else
            # Use tessellate-extract-features for each model
            EXTRACT_CMD_ARGS=(
                "tessellate_extract_features"
                "slide_path=$SLIDE_PATH"
                "output_h5_path=$MODEL_H5_PATH"
                "output_pt_path=$MODEL_PT_PATH"
                "model_type=$MODEL"
                "num_workers=$NUM_WORKERS"
                "batch_size=$BATCH_SIZE"
                "use_gpu=$USE_GPU"
                "keep_intermediate_files=$KEEP_INTERMEDIATE_FILES"
            )
            
            # Add seg_config group if specified
            if [ -n "$SEG_CONFIG_GROUP" ]; then
                EXTRACT_CMD_ARGS+=("seg_config=${SEG_CONFIG_GROUP}")
            fi
            
            # Add individual SegConfig parameters if specified
            if [ -n "$SEGMENT_THRESHOLD" ]; then
                EXTRACT_CMD_ARGS+=("seg_config.segment_threshold=$SEGMENT_THRESHOLD")
            fi
            if [ -n "$PATCH_SIZE" ]; then
                EXTRACT_CMD_ARGS+=("seg_config.patch_size=$PATCH_SIZE")
            fi
            if [ -n "$STEP_SIZE" ]; then
                EXTRACT_CMD_ARGS+=("seg_config.step_size=$STEP_SIZE")
            fi
            if [ -n "$MPP" ]; then
                EXTRACT_CMD_ARGS+=("seg_config.mpp=$MPP")
            fi
            if [ -n "$SEG_LEVEL" ]; then
                EXTRACT_CMD_ARGS+=("seg_config.seg_level=$SEG_LEVEL")
            fi
        fi
        
        # Add model_path if specified
        if [ -n "$MODEL_PATH" ]; then
            EXTRACT_CMD_ARGS+=("model_path=$MODEL_PATH")
        fi
        
        # Add aggregation parameters if specified
        if [ "$AGGREGATION_METHOD" != "identity" ]; then
            EXTRACT_CMD_ARGS+=("aggregation_method=$AGGREGATION_METHOD")
            EXTRACT_CMD_ARGS+=("intermediate_h5_path=$MODEL_INTERMEDIATE_H5")
        fi
        
        if [ -n "$SLIDE_MODEL_TYPE" ]; then
            EXTRACT_CMD_ARGS+=("slide_model_type=$SLIDE_MODEL_TYPE")
        fi
        
        if [ -n "$SLIDE_MODEL_PATH" ]; then
            EXTRACT_CMD_ARGS+=("slide_model_path=$SLIDE_MODEL_PATH")
        fi
        
        log "Executing extract-features command for $MODEL:"
        log "${EXTRACT_CMD_ARGS[*]}"
        echo ""
        
        MODEL_START_TIME=$(date +%s)
        ${UV_PREFIX} "${EXTRACT_CMD_ARGS[@]}"
        MODEL_EXIT_CODE=$?
        MODEL_END_TIME=$(date +%s)
        MODEL_DURATION=$((MODEL_END_TIME - MODEL_START_TIME))
        
        if [ $MODEL_EXIT_CODE -ne 0 ]; then
            log "ERROR: extract-features failed for model $MODEL with exit code $MODEL_EXIT_CODE (duration: $MODEL_DURATION seconds)"
            cleanup_staging
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

elif [ -n "$MODEL_TYPE" ]; then
    # Single model specified - use tessellate-extract-features
    MODEL="$MODEL_TYPE"
else
    # No model specified - use prefilter model if available
    MODEL="$PREFILTER_MODEL_TYPE"
fi

# Single-model mode (backward compatible - uses tessellate-extract-features)
if [ -z "$MODEL_TYPES" ]; then
    MODEL_H5_PATH="$LOCAL_OUTPUT_H5_PATH"
    MODEL_PT_PATH="$LOCAL_OUTPUT_PT_PATH"
    MODEL_INTERMEDIATE_H5_PATH="$LOCAL_INTERMEDIATE_H5_PATH"
    
    # Build the command as an array for safe execution
    CMD_ARGS=(
        "tessellate_extract_features"
        "slide_path=$SLIDE_PATH"
        "output_h5_path=$MODEL_H5_PATH"
        "output_pt_path=$MODEL_PT_PATH"
        "num_workers=$NUM_WORKERS"
        "batch_size=$BATCH_SIZE"
        "use_gpu=$USE_GPU"
        "keep_intermediate_files=$KEEP_INTERMEDIATE_FILES"
    )
    
    # Add model_dir if specified
    if [ -n "$MODEL_DIR" ]; then
        CMD_ARGS+=("model_dir=$MODEL_DIR")
    fi
    
    # Add prefilter model type if specified
    if [ -n "$PREFILTER_MODEL_TYPE" ]; then
        CMD_ARGS+=("prefilter_model_type=$PREFILTER_MODEL_TYPE")
    fi
    
    # Add seg_config group if specified
    if [ -n "$SEG_CONFIG_GROUP" ]; then
        CMD_ARGS+=("seg_config=${SEG_CONFIG_GROUP}")
    fi
    
    # Add individual SegConfig parameters if specified (these override group defaults)
    if [ -n "$SEGMENT_THRESHOLD" ]; then
        CMD_ARGS+=("seg_config.segment_threshold=$SEGMENT_THRESHOLD")
    fi
    
    # Determine patch size for this specific model
    # Priority: 1) Explicit PATCH_SIZE env var, 2) Model-specific default
    if [ -n "$PATCH_SIZE" ]; then
        MODEL_PATCH_SIZE="$PATCH_SIZE"
    else
        # Get model-specific default patch size
        MODEL_PATCH_SIZE=$(get_model_patch_size "$MODEL")
        log "Using model-specific patch size for $MODEL: $MODEL_PATCH_SIZE"
    fi
    CMD_ARGS+=("seg_config.patch_size=$MODEL_PATCH_SIZE")
    
    if [ -n "$STEP_SIZE" ]; then
        CMD_ARGS+=("seg_config.step_size=$STEP_SIZE")
    fi
    if [ -n "$MPP" ]; then
        CMD_ARGS+=("seg_config.mpp=$MPP")
    fi
    if [ -n "$SEG_LEVEL" ]; then
        CMD_ARGS+=("seg_config.seg_level=$SEG_LEVEL")
    fi
    if [ -n "$SEGMENT_MAX_VALUE" ]; then
        CMD_ARGS+=("seg_config.segment_max_value=$SEGMENT_MAX_VALUE")
    fi
    if [ -n "$MEDIAN_BLUR_KSIZE" ]; then
        CMD_ARGS+=("seg_config.median_blur_ksize=$MEDIAN_BLUR_KSIZE")
    fi
    if [ -n "$MORPHOLOGY_EX_KERNEL" ]; then
        CMD_ARGS+=("seg_config.morphology_ex_kernel=$MORPHOLOGY_EX_KERNEL")
    fi
    if [ -n "$REF_PATCH_SIZE" ]; then
        CMD_ARGS+=("seg_config.ref_patch_size=$REF_PATCH_SIZE")
    fi
    if [ -n "$USE_OTSU" ]; then
        CMD_ARGS+=("seg_config.use_otsu=$USE_OTSU")
    fi
    if [ -n "$TISSUE_AREA_THRESHOLD" ]; then
        CMD_ARGS+=("seg_config.tissue_area_threshold=$TISSUE_AREA_THRESHOLD")
    fi
    if [ -n "$HOLE_AREA_THRESHOLD" ]; then
        CMD_ARGS+=("seg_config.hole_area_threshold=$HOLE_AREA_THRESHOLD")
    fi
    if [ -n "$MAX_NUM_HOLES" ]; then
        CMD_ARGS+=("seg_config.max_num_holes=$MAX_NUM_HOLES")
    fi

    # Add optional parameters
    if [ -n "$CLASSIFIER_PKL" ]; then
        CMD_ARGS+=("classifier_pkl=$CLASSIFIER_PKL")
        CMD_ARGS+=("classifier_threshold=$CLASSIFIER_THRESHOLD")
    fi

    # Add model paths if specified
    if [ -n "$PREFILTER_MODEL_PATH" ]; then
        CMD_ARGS+=("prefilter_model_path=$PREFILTER_MODEL_PATH")
    fi

    # Add the specific model
    CMD_ARGS+=("model_type=$MODEL")
    
    if [ -n "$MODEL_PATH" ]; then
        CMD_ARGS+=("model_path=$MODEL_PATH")
    fi

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
    
    if [ -n "$SLIDE_MODEL_PATH" ]; then
        CMD_ARGS+=("slide_model_path=$SLIDE_MODEL_PATH")
    fi

    log "Executing command:"
    log "${CMD_ARGS[*]}"
    echo ""

    # Execute the command
    START_TIME=$(date +%s)
    ${UV_PREFIX} "${CMD_ARGS[@]}"
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    if [ $EXIT_CODE -ne 0 ]; then
        log "ERROR: Processing failed with exit code $EXIT_CODE (duration: $DURATION seconds)"
        cleanup_staging
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

# Clean up staged files after successful processing
cleanup_staging
exit 0
