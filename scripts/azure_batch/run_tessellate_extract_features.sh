#!/bin/bash
#
# Azure Batch task script for running tessellate-extract-features
# This script runs on Azure Batch compute nodes to process whole-slide images
#
# Environment variables expected:
#   SLIDE_PATH - Path to the input slide file (can be s3:// URL)
#   OUTPUT_H5_PATH - Path for output HDF5 file (can be s3:// URL or local path)
#   OUTPUT_PT_PATH - Path for output PyTorch file (can be s3:// URL or local path)
#   CLASSIFIER_PKL - (Optional) Path to classifier pickle file for filtering
#   CLASSIFIER_THRESHOLD - (Optional) Threshold for classifier (default: 0.75)
#   PREFILTER_MODEL_TYPE - Model type for pre-filter extraction (default: CTRANSPATH)
#   POSTFILTER_MODEL_TYPE - (Optional) Model type for post-filter extraction
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

if is_s3_path "$OUTPUT_H5_PATH" || is_s3_path "$OUTPUT_PT_PATH"; then
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
else
    LOCAL_OUTPUT_H5_PATH="$OUTPUT_H5_PATH"
    LOCAL_OUTPUT_PT_PATH="$OUTPUT_PT_PATH"
fi

# Create output directory if it doesn't exist (for local outputs)
OUTPUT_DIR=$(dirname "$LOCAL_OUTPUT_H5_PATH")
if [ ! -d "$OUTPUT_DIR" ]; then
    log "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Build the command as an array for safe execution
CMD_ARGS=(
    "tessellate_extract_features"
    "slide_path=$SLIDE_PATH"
    "output_h5_path=$LOCAL_OUTPUT_H5_PATH"
    "output_pt_path=$LOCAL_OUTPUT_PT_PATH"
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

if [ -n "$POSTFILTER_MODEL_TYPE" ]; then
    CMD_ARGS+=("postfilter_model_type=$POSTFILTER_MODEL_TYPE")
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
if [ $EXIT_CODE -eq 0 ]; then
    log "SUCCESS: Processing completed in $DURATION seconds"
    
    # Upload results to S3 if needed
    if is_s3_path "$ORIGINAL_OUTPUT_H5_PATH"; then
        if [ -f "$LOCAL_OUTPUT_H5_PATH" ]; then
            log "Local H5 file: $LOCAL_OUTPUT_H5_PATH (size: $(du -h "$LOCAL_OUTPUT_H5_PATH" | cut -f1))"
            upload_to_s3 "$LOCAL_OUTPUT_H5_PATH" "$ORIGINAL_OUTPUT_H5_PATH"
            log "Uploaded H5 file to S3: $ORIGINAL_OUTPUT_H5_PATH"
        fi
    else
        if [ -f "$LOCAL_OUTPUT_H5_PATH" ]; then
            log "Output H5 file: $LOCAL_OUTPUT_H5_PATH (size: $(du -h "$LOCAL_OUTPUT_H5_PATH" | cut -f1))"
        fi
    fi
    
    if is_s3_path "$ORIGINAL_OUTPUT_PT_PATH"; then
        if [ -f "$LOCAL_OUTPUT_PT_PATH" ]; then
            log "Local PT file: $LOCAL_OUTPUT_PT_PATH (size: $(du -h "$LOCAL_OUTPUT_PT_PATH" | cut -f1))"
            upload_to_s3 "$LOCAL_OUTPUT_PT_PATH" "$ORIGINAL_OUTPUT_PT_PATH"
            log "Uploaded PT file to S3: $ORIGINAL_OUTPUT_PT_PATH"
        fi
    else
        if [ -f "$LOCAL_OUTPUT_PT_PATH" ]; then
            log "Output PT file: $LOCAL_OUTPUT_PT_PATH (size: $(du -h "$LOCAL_OUTPUT_PT_PATH" | cut -f1))"
        fi
    fi
else
    log "ERROR: Processing failed with exit code $EXIT_CODE after $DURATION seconds"
fi

# Note: Cleanup is handled by the EXIT trap set at the beginning of the script

echo "============================================"
echo "End time: $(date)"
echo "============================================"

exit $EXIT_CODE
