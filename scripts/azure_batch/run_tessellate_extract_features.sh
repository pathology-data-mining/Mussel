#!/bin/bash
#
# Azure Batch task script for running tessellate-extract-features
# This script runs on Azure Batch compute nodes to process whole-slide images
#
# Environment variables expected:
#   SLIDE_PATH - Path to a single input slide file (can be s3:// or azfiles:// URL)
#   SLIDE_PATHS - Comma-separated paths for batch processing multiple slides
#   OUTPUT_H5_PATH - Path for output HDF5 file (can be s3:// URL or local path)
#   OUTPUT_PT_PATH - Path for output PyTorch file (can be s3:// URL or local path)
#   OUTPUT_H5_PATHS - Comma-separated output H5 paths for batch processing
#   OUTPUT_PT_PATHS - Comma-separated output PT paths for batch processing
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
#   CLEANUP_STAGED_FILE - (Optional) Cleanup staged Azure Files file after task completion (default: false)
#   AZURE_STORAGE_ACCOUNT - (Optional) Azure Storage account name for cleanup
#   AZURE_STORAGE_KEY - (Optional) Azure Storage account key for cleanup
#   AZURE_FILES_SHARE - (Optional) Azure Files share name for cleanup
#   HF_TOKEN - (Optional) HuggingFace token for gated models
#   AWS_ACCESS_KEY_ID - (Optional) AWS access key for S3
#   AWS_SECRET_ACCESS_KEY - (Optional) AWS secret key for S3
#   AWS_DEFAULT_REGION - (Optional) AWS region (default: us-east-1)
#   AWS_ENDPOINT_URL - (Optional) Custom S3 endpoint URL for S3-compatible storage (e.g., MinIO, Ceph)

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
    
    # Cleanup temporary work directory
    if [ -n "$WORK_DIR" ] && [ -d "$WORK_DIR" ]; then
        log "Cleanup: Removing work directory: $WORK_DIR"
        rm -rf "$WORK_DIR" || log "Warning: Failed to remove work directory"
    fi
    
    # Cleanup staged Azure Files file if requested and task succeeded
    if [ "$CLEANUP_STAGED_FILE" = "true" ] && [ $exit_code -eq 0 ] && [ -n "$STAGED_FILE_PATH" ]; then
        log "Cleanup: Removing staged file from Azure Files: $STAGED_FILE_PATH"
        
        # Check if we have the required Azure credentials
        if [ -n "$AZURE_STORAGE_ACCOUNT" ] && [ -n "$AZURE_STORAGE_KEY" ] && [ -n "$AZURE_FILES_SHARE" ]; then
            # Use az CLI to delete the file
            if command -v az &> /dev/null; then
                az storage file delete \
                    --account-name "$AZURE_STORAGE_ACCOUNT" \
                    --account-key "$AZURE_STORAGE_KEY" \
                    --share-name "$AZURE_FILES_SHARE" \
                    --path "$STAGED_FILE_PATH" 2>&1 | grep -v "^$" || log "Warning: Failed to delete staged file from Azure Files"
                log "Cleanup: Staged file deleted successfully"
            else
                log "Warning: az CLI not available, cannot cleanup staged file"
            fi
        else
            log "Warning: Missing Azure credentials for cleanup, skipping staged file deletion"
        fi
    fi
    
    # Don't exit here, let the script continue to its natural exit
}

# Set trap to ensure cleanup runs on exit (success, failure, or interruption)
trap cleanup EXIT INT TERM

# Detect batch processing mode
BATCH_MODE=false
if [ -n "$SLIDE_PATHS" ]; then
    BATCH_MODE=true
    log "Batch processing mode detected (SLIDE_PATHS is set)"
    
    # Convert comma-separated paths to arrays
    IFS=',' read -ra SLIDE_PATH_ARRAY <<< "$SLIDE_PATHS"
    
    # Check if OUTPUT_H5_PATHS/OUTPUT_PT_PATHS are provided, otherwise generate from OUTPUT_DIR
    if [ -n "$OUTPUT_H5_PATHS" ] && [ -n "$OUTPUT_PT_PATHS" ]; then
        IFS=',' read -ra OUTPUT_H5_PATH_ARRAY <<< "$OUTPUT_H5_PATHS"
        IFS=',' read -ra OUTPUT_PT_PATH_ARRAY <<< "$OUTPUT_PT_PATHS"
    else
        # Generate output paths from OUTPUT_DIR and slide IDs
        if [ -z "$OUTPUT_DIR" ]; then
            log "ERROR: Either OUTPUT_H5_PATHS/OUTPUT_PT_PATHS or OUTPUT_DIR must be set in batch mode"
            exit 1
        fi
        
        # Parse SLIDE_IDS if provided
        if [ -n "$SLIDE_IDS" ]; then
            IFS=',' read -ra SLIDE_ID_ARRAY <<< "$SLIDE_IDS"
        else
            # Extract slide IDs from slide paths (basename without extension)
            SLIDE_ID_ARRAY=()
            for slide_path in "${SLIDE_PATH_ARRAY[@]}"; do
                slide_basename=$(basename "$slide_path")
                slide_id="${slide_basename%.*}"
                SLIDE_ID_ARRAY+=("$slide_id")
            done
        fi
        
        # Generate output paths
        OUTPUT_H5_PATH_ARRAY=()
        OUTPUT_PT_PATH_ARRAY=()
        for slide_id in "${SLIDE_ID_ARRAY[@]}"; do
            # Check if OUTPUT_DIR is an azfiles:// URL
            if is_azfiles_path "$OUTPUT_DIR"; then
                OUTPUT_H5_PATH_ARRAY+=("${OUTPUT_DIR}/${slide_id}_features.h5")
                OUTPUT_PT_PATH_ARRAY+=("${OUTPUT_DIR}/${slide_id}_features.pt")
            else
                OUTPUT_H5_PATH_ARRAY+=("${OUTPUT_DIR}/${slide_id}_features.h5")
                OUTPUT_PT_PATH_ARRAY+=("${OUTPUT_DIR}/${slide_id}_features.pt")
            fi
        done
        
        log "Generated output paths from OUTPUT_DIR: $OUTPUT_DIR"
    fi
    
    NUM_SLIDES=${#SLIDE_PATH_ARRAY[@]}
    log "Processing $NUM_SLIDES slides in batch mode"
else
    log "Single slide mode (SLIDE_PATH is set)"
    
    # Check required environment variables for single slide mode
    if [ -z "$SLIDE_PATH" ]; then
        log "ERROR: Either SLIDE_PATH or SLIDE_PATHS environment variable is required"
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
CLEANUP_STAGED_FILE=${CLEANUP_STAGED_FILE:-false}

# Variable to track staged file path for cleanup
STAGED_FILE_PATH=""

# Storage helper functions
is_s3_path() {
    [[ "$1" =~ ^s3:// ]]
}

is_azfiles_path() {
    [[ "$1" =~ ^azfiles:// ]]
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

upload_to_azfiles() {
    local local_path="$1"
    local azfiles_url="$2"
    log "Uploading to Azure Files: $local_path -> $azfiles_url"
    
    # Extract path from azfiles:// URL: azfiles://account/share/path -> path
    local remote_path=$(echo "$azfiles_url" | sed 's|^azfiles://[^/]*/[^/]*/||')
    
    if [ -z "$AZURE_STORAGE_ACCOUNT" ] || [ -z "$AZURE_STORAGE_KEY" ] || [ -z "$AZURE_FILES_SHARE" ]; then
        log "ERROR: Azure Storage credentials not set (AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, AZURE_FILES_SHARE)"
        exit 1
    fi
    
    # Use Azure CLI if available
    if command -v az &> /dev/null; then
        az storage file upload \
            --account-name "$AZURE_STORAGE_ACCOUNT" \
            --account-key "$AZURE_STORAGE_KEY" \
            --share-name "$AZURE_FILES_SHARE" \
            --source "$local_path" \
            --path "$remote_path" \
            --no-progress
    else
        log "ERROR: az CLI not found. Install with: pip install azure-cli"
        exit 1
    fi
}

resolve_azfiles_path() {
    # Convert azfiles://account/share/path to /mnt/batch/tasks/fsmounts/azfiles/path
    local azfiles_url="$1"
    
    # Validate URL format
    if ! [[ "$azfiles_url" =~ ^azfiles://[^/]+/[^/]+/.+ ]]; then
        log "ERROR: Invalid Azure Files URL format: $azfiles_url"
        log "Expected format: azfiles://account/share/path"
        exit 1
    fi
    
    # Extract path after share name: azfiles://account/share/path -> path
    # Format: azfiles://account/share/path
    local path_part=$(echo "$azfiles_url" | sed 's|^azfiles://[^/]*/[^/]*/||')
    
    # Validate path extraction
    if [ -z "$path_part" ]; then
        log "ERROR: Could not extract path from Azure Files URL: $azfiles_url"
        exit 1
    fi
    
    # Azure Batch mounts Azure Files at /mnt/batch/tasks/fsmounts/azfiles
    local local_path="/mnt/batch/tasks/fsmounts/azfiles/$path_part"
    
    echo "$local_path"
}

log "Configuration:"
if [ "$BATCH_MODE" = "true" ]; then
    log "  Mode: BATCH ($NUM_SLIDES slides)"
    for i in "${!SLIDE_PATH_ARRAY[@]}"; do
        log "    Slide $((i+1)): ${SLIDE_PATH_ARRAY[$i]}"
    done
else
    log "  Mode: SINGLE SLIDE"
    log "  SLIDE_PATH: $SLIDE_PATH"
    log "  OUTPUT_H5_PATH: $OUTPUT_H5_PATH"
    log "  OUTPUT_PT_PATH: $OUTPUT_PT_PATH"
fi
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

# Function to process a single slide
process_slide() {
    local SLIDE_PATH="$1"
    local OUTPUT_H5_PATH="$2"
    local OUTPUT_PT_PATH="$3"
    local SLIDE_NUM="${4:-1}"
    
    log ""
    log "=========================================="
    log "Processing slide $SLIDE_NUM"
    log "=========================================="
    log "  Input: $SLIDE_PATH"
    log "  Output H5: $OUTPUT_H5_PATH"
    log "  Output PT: $OUTPUT_PT_PATH"
    log ""

# Stage input slide - handle S3, Azure Files, or local paths
ORIGINAL_SLIDE_PATH="$SLIDE_PATH"
if is_azfiles_path "$SLIDE_PATH"; then
    # Resolve Azure Files path to local mount point
    log "Slide is in Azure Files, resolving to mount point..."
    
    # Extract the file path from azfiles:// URL for cleanup
    # Format: azfiles://account/share/path -> path
    STAGED_FILE_PATH=$(echo "$SLIDE_PATH" | sed 's|^azfiles://[^/]*/[^/]*/||')
    
    SLIDE_PATH=$(resolve_azfiles_path "$SLIDE_PATH")
    log "Resolved to: $SLIDE_PATH"
    
    if [ "$CLEANUP_STAGED_FILE" = "true" ]; then
        log "Staged file will be cleaned up after task completion: $STAGED_FILE_PATH"
    fi
elif is_s3_path "$SLIDE_PATH"; then
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
        # Resolve azfiles:// path if needed
        if is_azfiles_path "$CLASSIFIER_PKL"; then
            RESOLVED_CLASSIFIER_PKL=$(resolve_azfiles_path "$CLASSIFIER_PKL")
            CMD_ARGS+=("classifier_pkl=$RESOLVED_CLASSIFIER_PKL")
        else
            CMD_ARGS+=("classifier_pkl=$CLASSIFIER_PKL")
        fi
        CMD_ARGS+=("classifier_threshold=$CLASSIFIER_THRESHOLD")
    fi
    
    # Add prefilter model path if provided
    if [ -n "$PREFILTER_MODEL_PATH" ]; then
        # Resolve azfiles:// path if needed
        if is_azfiles_path "$PREFILTER_MODEL_PATH"; then
            RESOLVED_PREFILTER_PATH=$(resolve_azfiles_path "$PREFILTER_MODEL_PATH")
            CMD_ARGS+=("prefilter_model_path=$RESOLVED_PREFILTER_PATH")
        else
            CMD_ARGS+=("prefilter_model_path=$PREFILTER_MODEL_PATH")
        fi
    fi
    
    # Add postfilter model path if provided
    if [ -n "$POSTFILTER_MODEL_PATH" ]; then
        # Resolve azfiles:// path if needed
        if is_azfiles_path "$POSTFILTER_MODEL_PATH"; then
            RESOLVED_POSTFILTER_PATH=$(resolve_azfiles_path "$POSTFILTER_MODEL_PATH")
            CMD_ARGS+=("postfilter_model_path=$RESOLVED_POSTFILTER_PATH")
        else
            CMD_ARGS+=("postfilter_model_path=$POSTFILTER_MODEL_PATH")
        fi
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
    
    # Upload results to S3 or Azure Files if needed
    if is_s3_path "$ORIGINAL_OUTPUT_H5_PATH" || is_azfiles_path "$ORIGINAL_OUTPUT_H5_PATH"; then
        if [ -f "$MODEL_H5_PATH" ]; then
            log "Local H5 file: $MODEL_H5_PATH (size: $(du -h "$MODEL_H5_PATH" | cut -f1))"
            if is_s3_path "$ORIGINAL_OUTPUT_H5_PATH"; then
                upload_to_s3 "$MODEL_H5_PATH" "$ORIGINAL_OUTPUT_H5_PATH"
                log "Uploaded H5 file to S3: $ORIGINAL_OUTPUT_H5_PATH"
            else
                upload_to_azfiles "$MODEL_H5_PATH" "$ORIGINAL_OUTPUT_H5_PATH"
                log "Uploaded H5 file to Azure Files: $ORIGINAL_OUTPUT_H5_PATH"
            fi
        fi
        
        if [ -f "$MODEL_PT_PATH" ]; then
            log "Local PT file: $MODEL_PT_PATH (size: $(du -h "$MODEL_PT_PATH" | cut -f1))"
            if is_s3_path "$ORIGINAL_OUTPUT_PT_PATH"; then
                upload_to_s3 "$MODEL_PT_PATH" "$ORIGINAL_OUTPUT_PT_PATH"
                log "Uploaded PT file to S3: $ORIGINAL_OUTPUT_PT_PATH"
            else
                upload_to_azfiles "$MODEL_PT_PATH" "$ORIGINAL_OUTPUT_PT_PATH"
                log "Uploaded PT file to Azure Files: $ORIGINAL_OUTPUT_PT_PATH"
            fi
        fi
        
        if [ -n "$MODEL_INTERMEDIATE_H5_PATH" ] && [ -f "$MODEL_INTERMEDIATE_H5_PATH" ]; then
            log "Local intermediate H5 file: $MODEL_INTERMEDIATE_H5_PATH (size: $(du -h "$MODEL_INTERMEDIATE_H5_PATH" | cut -f1))"
            if is_s3_path "$ORIGINAL_INTERMEDIATE_H5_PATH"; then
                upload_to_s3 "$MODEL_INTERMEDIATE_H5_PATH" "$ORIGINAL_INTERMEDIATE_H5_PATH"
                log "Uploaded intermediate H5 file to S3: $ORIGINAL_INTERMEDIATE_H5_PATH"
            else
                upload_to_azfiles "$MODEL_INTERMEDIATE_H5_PATH" "$ORIGINAL_INTERMEDIATE_H5_PATH"
                log "Uploaded intermediate H5 file to Azure Files: $ORIGINAL_INTERMEDIATE_H5_PATH"
            fi
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
}  # End of process_slide function

# Main execution logic
if [ "$BATCH_MODE" = "true" ]; then
    log "Starting batch processing of $NUM_SLIDES slides..."
    
    # Process each slide
    for i in "${!SLIDE_PATH_ARRAY[@]}"; do
        SLIDE_PATH="${SLIDE_PATH_ARRAY[$i]}"
        OUTPUT_H5_PATH="${OUTPUT_H5_PATH_ARRAY[$i]}"
        OUTPUT_PT_PATH="${OUTPUT_PT_PATH_ARRAY[$i]}"
        
        # Process this slide
        process_slide "$SLIDE_PATH" "$OUTPUT_H5_PATH" "$OUTPUT_PT_PATH" "$((i+1))"
        
        if [ $? -ne 0 ]; then
            log "ERROR: Failed to process slide $((i+1)): $SLIDE_PATH"
            exit 1
        fi
    done
    
    log ""
    log "=========================================="
    log "Batch processing completed: $NUM_SLIDES/$NUM_SLIDES slides successful"
    log "=========================================="
else
    # Single slide mode - call process_slide function
    process_slide "$SLIDE_PATH" "$OUTPUT_H5_PATH" "$OUTPUT_PT_PATH" "1"
fi

echo "============================================"
echo "End time: $(date)"
echo "============================================"

# Final cleanup happens via trap
exit 0
