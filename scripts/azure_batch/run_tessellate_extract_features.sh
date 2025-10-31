#!/bin/bash
#
# Azure Batch task script for running tessellate-extract-features
# This script runs on Azure Batch compute nodes to process whole-slide images
#
# Environment variables expected:
#   SLIDE_PATH - Path to the input slide file
#   OUTPUT_H5_PATH - Path for output HDF5 file
#   OUTPUT_PT_PATH - Path for output PyTorch file
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

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_H5_PATH")
if [ ! -d "$OUTPUT_DIR" ]; then
    log "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Build the command as an array for safe execution
CMD_ARGS=(
    "tessellate_extract_features"
    "slide_path=$SLIDE_PATH"
    "output_h5_path=$OUTPUT_H5_PATH"
    "output_pt_path=$OUTPUT_PT_PATH"
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
    
    # Show output file sizes
    if [ -f "$OUTPUT_H5_PATH" ]; then
        log "Output H5 file: $OUTPUT_H5_PATH (size: $(du -h "$OUTPUT_H5_PATH" | cut -f1))"
    fi
    if [ -f "$OUTPUT_PT_PATH" ]; then
        log "Output PT file: $OUTPUT_PT_PATH (size: $(du -h "$OUTPUT_PT_PATH" | cut -f1))"
    fi
else
    log "ERROR: Processing failed with exit code $EXIT_CODE after $DURATION seconds"
fi

echo "============================================"
echo "End time: $(date)"
echo "============================================"

exit $EXIT_CODE
