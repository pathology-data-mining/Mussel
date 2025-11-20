#!/bin/bash
#
# SIMPLIFIED Azure Batch script for tessellate-extract-features
# 
# Key changes:
# - NO model pre-staging from Azure Blob
# - Models are downloaded directly from HuggingFace on first use
# - Persistent cache at /mnt/batch_models/.cache for reuse across tasks
# - Python's file locking handles concurrent downloads safely
#

set -e
set -o pipefail

echo "============================================"
echo "Azure Batch - Simplified Model Caching"
echo "============================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >&2
}

# Set up persistent model cache directories
export HF_HOME="${HF_HOME:-/mnt/batch_models/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="${TORCH_HOME:-/mnt/batch_models/.cache/torch}"

# Create cache directories
mkdir -p "$HF_HOME" "$TORCH_HOME"
log "Persistent model cache at: $HF_HOME"
log "Torch cache at: $TORCH_HOME"

# Use tmpdir in /mnt/batch (has more space than /tmp)
WORK_DIR="${TMPDIR:-/mnt/batch/tasks/workitems/tmp}/mussel_work_$$"

cleanup() {
    local exit_code=$?
    if [ -n "$WORK_DIR" ] && [ -d "$WORK_DIR" ]; then
        log "Cleanup: Removing work directory"
        rm -rf "$WORK_DIR" || true
    fi
    log "Cleanup complete. Persistent cache kept at /mnt/batch_models"
}

trap cleanup EXIT INT TERM

# Function to stage remote files (slides only, not models)
stage_remote_file() {
    local remote_path="$1"
    local local_path="$2"
    
    if [[ "$remote_path" =~ ^azfiles:// ]]; then
        log "Staging from Azure Files: $remote_path"
        local file_path=$(echo "$remote_path" | sed 's|^azfiles://[^/]*/[^/]*/||')
        local azfiles_mount="/mnt/batch/tasks/fsmounts/azfiles"
        cp "$azfiles_mount/$file_path" "$local_path"
        echo "$local_path"
        
    elif [[ "$remote_path" =~ ^s3:// ]]; then
        log "Staging from S3: $remote_path"
        aws s3 cp "$remote_path" "$local_path" --no-progress
        echo "$local_path"
        
    elif [[ "$remote_path" =~ ^azblob:// ]]; then
        log "Staging from Azure Blob: $remote_path"
        local blob_url=$(echo "$remote_path" | sed 's|^azblob://||')
        local storage_account=$(echo "$blob_url" | cut -d'/' -f1 | cut -d'.' -f1)
        local container=$(echo "$blob_url" | cut -d'/' -f2)
        local blob_name=$(echo "$blob_url" | cut -d'/' -f3-)
        
        if [ -n "$AZURE_STORAGE_KEY" ]; then
            az storage blob download \
                --account-name "$storage_account" \
                --account-key "$AZURE_STORAGE_KEY" \
                --container-name "$container" \
                --name "$blob_name" \
                --file "$local_path" >&2
        else
            az storage blob download \
                --account-name "$storage_account" \
                --container-name "$container" \
                --name "$blob_name" \
                --file "$local_path" \
                --auth-mode login >&2
        fi
        echo "$local_path"
    else
        echo "$remote_path"
    fi
}

# Process slide paths (batch or single mode)
if [ -n "$SLIDE_PATHS" ]; then
    log "Batch mode: $SLIDE_PATHS"
    IFS=',' read -ra PATH_ARRAY <<< "$SLIDE_PATHS"
    STAGED_PATHS=()
    mkdir -p "$WORK_DIR"
    
    for slide_path in "${PATH_ARRAY[@]}"; do
        slide_path=$(echo "$slide_path" | xargs)
        if [[ "$slide_path" =~ ^(azfiles|s3|azblob):// ]]; then
            local_path="$WORK_DIR/$(basename "$slide_path")"
            staged_path=$(stage_remote_file "$slide_path" "$local_path")
            STAGED_PATHS+=("$staged_path")
        else
            STAGED_PATHS+=("$slide_path")
        fi
    done
    EFFECTIVE_SLIDE_PATH="[$(IFS=,; echo "${STAGED_PATHS[*]}")]"
    
elif [ -n "$SLIDE_PATH" ]; then
    if [[ "$SLIDE_PATH" =~ ^(azfiles|s3|azblob):// ]]; then
        mkdir -p "$WORK_DIR"
        local_path="$WORK_DIR/$(basename "$SLIDE_PATH")"
        EFFECTIVE_SLIDE_PATH=$(stage_remote_file "$SLIDE_PATH" "$local_path")
    else
        EFFECTIVE_SLIDE_PATH="$SLIDE_PATH"
    fi
else
    log "ERROR: Neither SLIDE_PATH nor SLIDE_PATHS is set"
    exit 1
fi

# Set HuggingFace token
[ -n "$HF_TOKEN" ] && export HUGGINGFACE_TOKEN="$HF_TOKEN"

# NO MODEL STAGING - Models download from HuggingFace automatically
# Python code uses file locking to prevent concurrent download conflicts

# Build command using Python CLI directly
# The CLI reads all configuration from environment variables via Hydra
CMD="python -m mussel.cli.tessellate_extract_features"

# Add slide path
CMD="$CMD slide_path='$EFFECTIVE_SLIDE_PATH'"

# Add output paths or output_dir
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD output_dir='$OUTPUT_DIR'"
fi
if [ -n "$OUTPUT_H5_PATH" ] && [ "$OUTPUT_H5_PATH" != "None" ]; then
    CMD="$CMD output_h5_path='$OUTPUT_H5_PATH'"
fi
if [ -n "$OUTPUT_PT_PATH" ] && [ "$OUTPUT_PT_PATH" != "None" ]; then
    CMD="$CMD output_pt_path='$OUTPUT_PT_PATH'"
fi

# Add model types
if [ -n "$MODEL_TYPES" ]; then
    # MODEL_TYPES is a comma-separated list, wrap in brackets for Hydra
    CMD="$CMD model_type=[$MODEL_TYPES]"
elif [ -n "$MODEL_TYPE" ]; then
    CMD="$CMD model_type=$MODEL_TYPE"
fi

# Add common parameters
[ -n "$NUM_WORKERS" ] && CMD="$CMD num_workers=$NUM_WORKERS"
[ -n "$BATCH_SIZE" ] && CMD="$CMD batch_size=$BATCH_SIZE"
[ -n "$USE_GPU" ] && CMD="$CMD use_gpu=$USE_GPU"
[ -n "$AGGREGATION_METHOD" ] && CMD="$CMD aggregation_method=$AGGREGATION_METHOD"
[ -n "$KEEP_INTERMEDIATE_FILES" ] && CMD="$CMD keep_intermediate_files=$KEEP_INTERMEDIATE_FILES"

# Add segmentation config
if [ -n "$SEG_CONFIG_GROUP" ]; then
    CMD="$CMD seg_config=$SEG_CONFIG_GROUP"
fi

# Add model batch sizes as Hydra overrides
# Convert JSON dict to Hydra syntax: +model_batch_sizes.MODEL=SIZE
if [ -n "$MODEL_BATCH_SIZES" ]; then
    # Parse JSON and add each model's batch size with + override syntax
    # Example: {"OPTIMUS": 384, "VIRCHOW2": 256} -> +model_batch_sizes.OPTIMUS=384 +model_batch_sizes.VIRCHOW2=256
    for pair in $(echo "$MODEL_BATCH_SIZES" | python3 -c "import sys, json; d=json.load(sys.stdin); print(' '.join([f'{k}={v}' for k,v in d.items()]))"); do
        model=$(echo "$pair" | cut -d'=' -f1)
        size=$(echo "$pair" | cut -d'=' -f2)
        CMD="$CMD +model_batch_sizes.$model=$size"
    done
fi

log "Running command: $CMD"
log ""

# Execute
eval "$CMD"

log ""
log "Task completed successfully"
log "End time: $(date)"
