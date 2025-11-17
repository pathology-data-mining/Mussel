#!/bin/bash
#
# Azure Batch task script for running tessellate-extract-features
# This script:
# 1. Stages input files from Azure/S3 to local storage if needed
# 2. Runs processing with output directly to OUTPUT_DIR (supports remote paths)
#

set -e
set -o pipefail

echo "============================================"
echo "Azure Batch Tessellate-Extract-Features Task"
echo "============================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

# Function to log with timestamp (send to stderr to not interfere with function return values)
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >&2
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [ -n "$WORK_DIR" ] && [ -d "$WORK_DIR" ]; then
        log "Cleanup: Removing work directory: $WORK_DIR"
        rm -rf "$WORK_DIR" || log "Warning: Failed to remove work directory"
    fi
    
    # Cleanup staged Azure Files file if requested and task succeeded
    if [ "$CLEANUP_STAGED_FILE" = "true" ] && [ $exit_code -eq 0 ] && [ -n "$STAGED_FILE_PATH" ]; then
        log "Cleanup: Removing staged file from Azure Files: $STAGED_FILE_PATH"
        if [ -n "$AZURE_STORAGE_ACCOUNT" ] && [ -n "$AZURE_STORAGE_KEY" ] && [ -n "$AZURE_FILES_SHARE" ]; then
            if command -v az &> /dev/null; then
                az storage file delete \
                    --account-name "$AZURE_STORAGE_ACCOUNT" \
                    --account-key "$AZURE_STORAGE_KEY" \
                    --share-name "$AZURE_FILES_SHARE" \
                    --path "$STAGED_FILE_PATH" 2>&1 | grep -v "^$" || log "Warning: Failed to delete staged file"
                log "Cleanup: Staged file deleted successfully"
            fi
        fi
    fi
    
    # Cleanup model cache to free disk space after task completion
    # Try environment variables first, then fall back to known cache locations
    local cache_dirs=()
    
    # Add environment variable paths if set
    [ -n "$HF_HOME" ] && cache_dirs+=("$HF_HOME")
    [ -n "$TRANSFORMERS_CACHE" ] && cache_dirs+=("$TRANSFORMERS_CACHE")
    [ -n "$TORCH_HOME" ] && cache_dirs+=("$TORCH_HOME")
    
    # Add known cache locations used by Azure Batch
    cache_dirs+=(
        "/mnt/batch/tasks/workitems/hf_cache"
        "/mnt/batch/tasks/workitems/torch_cache"
        "/root/.cache/huggingface"
        "/root/.cache/torch"
    )
    
    # Remove duplicates and clean up cache directories
    local cleaned_dirs=()
    for dir in "${cache_dirs[@]}"; do
        if [ -d "$dir" ] && [[ ! " ${cleaned_dirs[@]} " =~ " ${dir} " ]]; then
            log "Cleanup: Removing cache directory: $dir"
            local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            [ -n "$size" ] && log "  Cache size: $size"
            rm -rf "$dir" || log "Warning: Failed to remove cache directory: $dir"
            cleaned_dirs+=("$dir")
        fi
    done
    
    if [ ${#cleaned_dirs[@]} -eq 0 ]; then
        log "Cleanup: No cache directories found to clean"
    else
        log "Cleanup: Cleaned ${#cleaned_dirs[@]} cache director(ies)"
    fi
}

trap cleanup EXIT INT TERM

# Stage slide from Azure Files/S3/Azure Blob to local SSD if needed
# Use batch task working directory instead of root filesystem
WORK_DIR="${TMPDIR:-/mnt/batch/tasks/workitems/tmp}/mussel_work_$$"
STAGED_FILE_PATH=""
CLEANUP_STAGED_FILE=${CLEANUP_STAGED_FILE:-false}

# Function to stage a single remote file to local
stage_remote_file() {
    local remote_path="$1"
    local local_path="$2"
    
    if [[ "$remote_path" =~ ^azfiles:// ]]; then
        # Azure Files staging
        log "Staging from Azure Files: $remote_path"
        STAGED_FILE_PATH=$(echo "$remote_path" | sed 's|^azfiles://[^/]*/[^/]*/||')
        AZFILES_MOUNT="/mnt/batch/tasks/fsmounts/azfiles"
        
        log "Copying: $AZFILES_MOUNT/$STAGED_FILE_PATH -> $local_path"
        START=$(date +%s)
        if cp "$AZFILES_MOUNT/$STAGED_FILE_PATH" "$local_path"; then
            DURATION=$(($(date +%s) - START))
            log "Copied in $DURATION seconds (size: $(du -h "$local_path" | cut -f1))"
        else
            log "ERROR: Failed to copy from Azure Files: $AZFILES_MOUNT/$STAGED_FILE_PATH"
            exit 1
        fi
        echo "$local_path"
        return
        
    elif [[ "$remote_path" =~ ^s3:// ]]; then
        # S3 staging
        log "Staging from S3: $remote_path"
        START=$(date +%s)
        if command -v aws &> /dev/null; then
            if aws s3 cp "$remote_path" "$local_path" --no-progress; then
                DURATION=$(($(date +%s) - START))
                log "Downloaded in $DURATION seconds (size: $(du -h "$local_path" | cut -f1))"
            else
                log "ERROR: Failed to download from S3: $remote_path"
                exit 1
            fi
        else
            log "ERROR: aws CLI not available for S3 download"
            exit 1
        fi
        echo "$local_path"
        return
        
    elif [[ "$remote_path" =~ ^azblob:// ]]; then
        # Azure Blob staging
        log "Staging from Azure Blob: $remote_path"
        BLOB_URL=$(echo "$remote_path" | sed 's|^azblob://||')
        STORAGE_ACCOUNT=$(echo "$BLOB_URL" | cut -d'/' -f1 | cut -d'.' -f1)
        CONTAINER=$(echo "$BLOB_URL" | cut -d'/' -f2)
        BLOB_NAME=$(echo "$BLOB_URL" | cut -d'/' -f3-)
        
        START=$(date +%s)
        DOWNLOAD_SUCCESS=false
        if command -v az &> /dev/null; then
            # Use azcopy for better performance if available
            if command -v azcopy &> /dev/null && [ -n "$AZURE_STORAGE_KEY" ]; then
                export AZCOPY_AUTO_LOGIN_TYPE=SPN
                if azcopy copy "https://${STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER}/${BLOB_NAME}?${AZURE_STORAGE_KEY}" "$local_path"; then
                    DOWNLOAD_SUCCESS=true
                fi
            else
                # Fall back to az storage blob download with account key if available
                if [ -n "$AZURE_STORAGE_KEY" ]; then
                    if az storage blob download \
                        --account-name "$STORAGE_ACCOUNT" \
                        --account-key "$AZURE_STORAGE_KEY" \
                        --container-name "$CONTAINER" \
                        --name "$BLOB_NAME" \
                        --file "$local_path" >&2; then
                        DOWNLOAD_SUCCESS=true
                    fi
                else
                    if az storage blob download \
                        --account-name "$STORAGE_ACCOUNT" \
                        --container-name "$CONTAINER" \
                        --name "$BLOB_NAME" \
                        --file "$local_path" \
                        --auth-mode login >&2; then
                        DOWNLOAD_SUCCESS=true
                    fi
                fi
            fi
            
            if [ "$DOWNLOAD_SUCCESS" = true ]; then
                DURATION=$(($(date +%s) - START))
                log "Downloaded in $DURATION seconds (size: $(du -h "$local_path" | cut -f1))"
            else
                log "ERROR: Failed to download from Azure Blob: $remote_path"
                exit 1
            fi
        else
            log "ERROR: az CLI not available for Azure Blob download"
            exit 1
        fi
        echo "$local_path"
        return
    else
        # Not a remote path - return as-is
        echo "$remote_path"
        return
    fi
}

# Check if batch mode (SLIDE_PATHS) or single mode (SLIDE_PATH)
if [ -n "$SLIDE_PATHS" ]; then
    # Batch mode - process multiple slides
    log "Batch mode: Processing ${SLIDE_PATHS}"
    
    # Split comma-separated paths and stage each remote file
    IFS=',' read -ra PATH_ARRAY <<< "$SLIDE_PATHS"
    STAGED_PATHS=()
    mkdir -p "$WORK_DIR"
    
    for slide_path in "${PATH_ARRAY[@]}"; do
        # Trim whitespace
        slide_path=$(echo "$slide_path" | xargs)
        
        if [[ "$slide_path" =~ ^(azfiles|s3|azblob):// ]]; then
            # Remote path - stage it locally
            local_path="$WORK_DIR/$(basename "$slide_path")"
            staged_path=$(stage_remote_file "$slide_path" "$local_path")
            STAGED_PATHS+=("$staged_path")
        else
            # Local path - use as-is
            STAGED_PATHS+=("$slide_path")
        fi
    done
    
    # Join staged paths with commas and wrap in brackets for Hydra
    EFFECTIVE_SLIDE_PATH="[$(IFS=,; echo "${STAGED_PATHS[*]}")]"
    
elif [ -n "$SLIDE_PATH" ]; then
    # Single slide mode
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

# Setup output directories
# Store the remote output path for later upload
REMOTE_OUTPUT_DIR="${OUTPUT_DIR:-/mnt/batch/tasks/shared/output}"
IS_REMOTE_OUTPUT=false

# Check if output is remote (azblob://, az://, s3://, etc.)
if [[ "$REMOTE_OUTPUT_DIR" =~ ^(azblob://|az://|s3://|http://|https://) ]]; then
    log "Remote output detected: $REMOTE_OUTPUT_DIR"
    log "Using local temp directory for processing, will upload at end"
    IS_REMOTE_OUTPUT=true
    # Use local temp directory for processing
    LOCAL_OUTPUT_DIR="${WORK_DIR}/output"
    mkdir -p "$LOCAL_OUTPUT_DIR"
    EFFECTIVE_OUTPUT_DIR="$LOCAL_OUTPUT_DIR"
else
    # Local output path - use directly
    EFFECTIVE_OUTPUT_DIR="$REMOTE_OUTPUT_DIR"
fi

# Build command arguments - pass everything through to Python CLI
CMD_ARGS=(
    "tessellate_extract_features"
    "hydra.run.dir=/mnt/batch/tasks/workitems/tmp/hydra_outputs"
    "output_dir=$EFFECTIVE_OUTPUT_DIR"
)

# Slide path - use slide_paths for lists (with brackets), slide_path for single
if [[ "$EFFECTIVE_SLIDE_PATH" == \[*\] ]]; then
    CMD_ARGS+=("slide_paths=$EFFECTIVE_SLIDE_PATH")
else
    CMD_ARGS+=("slide_path=$EFFECTIVE_SLIDE_PATH")
fi

# Model configuration - add brackets for lists (comma-separated values)
[ -n "$MODEL_TYPES" ] && CMD_ARGS+=("model_type=[$MODEL_TYPES]")
[ -n "$MODEL_TYPE" ] && CMD_ARGS+=("model_type=$MODEL_TYPE")
[ -n "$MODEL_PATH" ] && CMD_ARGS+=("model_path=$MODEL_PATH")
[ -n "$SLIDE_MODEL_TYPES" ] && CMD_ARGS+=("slide_model_type=[$SLIDE_MODEL_TYPES]")
[ -n "$SLIDE_MODEL_TYPE" ] && CMD_ARGS+=("slide_model_type=$SLIDE_MODEL_TYPE")
[ -n "$SLIDE_MODEL_PATH" ] && CMD_ARGS+=("slide_model_path=$SLIDE_MODEL_PATH")

# Processing parameters
[ -n "$NUM_WORKERS" ] && CMD_ARGS+=("num_workers=$NUM_WORKERS")
[ -n "$BATCH_SIZE" ] && CMD_ARGS+=("batch_size=$BATCH_SIZE")
[ -n "$SLIDE_BATCH_SIZE" ] && CMD_ARGS+=("slide_batch_size=$SLIDE_BATCH_SIZE")
[ -n "$USE_GPU" ] && CMD_ARGS+=("use_gpu=$USE_GPU")

# Segmentation config
[ -n "$SEG_CONFIG_GROUP" ] && CMD_ARGS+=("seg_config=$SEG_CONFIG_GROUP")
[ -n "$SEGMENT_THRESHOLD" ] && CMD_ARGS+=("seg_config.segment_threshold=$SEGMENT_THRESHOLD")
[ -n "$PATCH_SIZE" ] && CMD_ARGS+=("seg_config.patch_size=$PATCH_SIZE")
[ -n "$MPP" ] && CMD_ARGS+=("seg_config.mpp=$MPP")
[ -n "$STEP_SIZE" ] && CMD_ARGS+=("seg_config.step_size=$STEP_SIZE")
[ -n "$SEG_LEVEL" ] && CMD_ARGS+=("seg_config.seg_level=$SEG_LEVEL")

# Aggregation
[ -n "$AGGREGATION_METHOD" ] && [ "$AGGREGATION_METHOD" != "identity" ] && CMD_ARGS+=("aggregation_method=$AGGREGATION_METHOD")

# Additional config (optional)
[ -n "$KEEP_INTERMEDIATE_FILES" ] && CMD_ARGS+=("keep_intermediate_files=$KEEP_INTERMEDIATE_FILES")

log "Executing:"
log "${CMD_ARGS[*]}"
echo ""

# Multi-GPU parallel processing
# Detect number of GPUs and split slides across them
NUM_GPUS=0
if command -v nvidia-smi &> /dev/null; then
    # Use --query-gpu with count to get reliable number, or fall back to parsing the table
    if nvidia-smi --query-gpu=count --format=csv,noheader | head -n1 | grep -qE '^[0-9]+$' 2>/dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)
    else
        # Count GPUs from the main table output
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    fi
    log "Detected $NUM_GPUS GPU(s)"
    [ $NUM_GPUS -gt 0 ] && log "GPU info: $(nvidia-smi --query-gpu=index,name --format=csv,noheader | tr '\n' '; ')"
else
    log "WARNING: nvidia-smi not found, running in single-process mode"
    NUM_GPUS=1
fi

# Enable multi-GPU processing if we have multiple GPUs and multiple slides
MULTI_GPU_MODE=false
if [ $NUM_GPUS -gt 1 ] && [[ "$EFFECTIVE_SLIDE_PATH" == \[*\] ]]; then
    # Extract slides from bracketed list and count them
    SLIDES_STR=$(echo "$EFFECTIVE_SLIDE_PATH" | sed 's/^\[//;s/\]$//')
    IFS=',' read -ra SLIDES_TEMP <<< "$SLIDES_STR"
    SLIDE_COUNT=${#SLIDES_TEMP[@]}
    
    if [ $SLIDE_COUNT -ge $NUM_GPUS ]; then
        log "Multi-GPU mode enabled: $NUM_GPUS GPUs, $SLIDE_COUNT slides"
        MULTI_GPU_MODE=true
    else
        log "Multi-GPU mode disabled: $SLIDE_COUNT slides < $NUM_GPUS GPUs (not worth parallelizing)"
    fi
fi

START_TIME=$(date +%s)

if [ "$MULTI_GPU_MODE" = true ]; then
    # Multi-GPU parallel execution
    log "Launching $NUM_GPUS parallel processes, one per GPU"
    
    # Parse slides from bracketed list
    SLIDES_STR=$(echo "$EFFECTIVE_SLIDE_PATH" | sed 's/^\[//;s/\]$//')
    IFS=',' read -ra SLIDES <<< "$SLIDES_STR"
    TOTAL_SLIDES=${#SLIDES[@]}
    
    # Calculate slides per GPU (round up)
    SLIDES_PER_GPU=$(( (TOTAL_SLIDES + NUM_GPUS - 1) / NUM_GPUS ))
    log "Distributing $TOTAL_SLIDES slides across $NUM_GPUS GPUs (~$SLIDES_PER_GPU slides per GPU)"
    
    # Launch parallel processes
    PIDS=()
    ACTUAL_GPUS=0
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        START_IDX=$((gpu_id * SLIDES_PER_GPU))
        END_IDX=$((START_IDX + SLIDES_PER_GPU - 1))
        
        # Don't exceed total slides
        if [ $START_IDX -ge $TOTAL_SLIDES ]; then
            log "GPU $gpu_id: No slides assigned (all slides already distributed)"
            break
        fi
        if [ $END_IDX -ge $TOTAL_SLIDES ]; then
            END_IDX=$((TOTAL_SLIDES - 1))
        fi
        
        ACTUAL_GPUS=$((ACTUAL_GPUS + 1))
        
        # Extract slides for this GPU
        GPU_SLIDES=()
        for idx in $(seq $START_IDX $END_IDX); do
            GPU_SLIDES+=("${SLIDES[$idx]}")
        done
        
        GPU_SLIDES_STR="[$(IFS=,; echo "${GPU_SLIDES[*]}")]"
        log "GPU $gpu_id: Processing ${#GPU_SLIDES[@]} slides"
        
        # Build command for this GPU
        GPU_CMD_ARGS=()
        for arg in "${CMD_ARGS[@]}"; do
            # Replace slide_paths argument with GPU-specific slides
            if [[ "$arg" == slide_paths=* ]]; then
                GPU_CMD_ARGS+=("slide_paths=$GPU_SLIDES_STR")
            else
                GPU_CMD_ARGS+=("$arg")
            fi
        done
        
        # Launch process with CUDA_VISIBLE_DEVICES set to this GPU
        (
            export CUDA_VISIBLE_DEVICES=$gpu_id
            log "GPU $gpu_id: Starting process (CUDA_VISIBLE_DEVICES=$gpu_id)"
            python -m mussel.cli.tessellate_extract_features "${GPU_CMD_ARGS[@]:1}" 2>&1 | sed "s/^/[GPU $gpu_id] /"
            exit_code=${PIPESTATUS[0]}
            log "GPU $gpu_id: Finished with exit code $exit_code"
            exit $exit_code
        ) &
        
        PIDS+=($!)
    done
    
    # Wait for all processes to complete
    log "Waiting for $ACTUAL_GPUS GPU processes to complete..."
    EXIT_CODE=0
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        if wait $pid; then
            log "GPU $i process (PID $pid) completed successfully"
        else
            proc_exit=$?
            log "ERROR: GPU $i process (PID $pid) failed with exit code $proc_exit"
            EXIT_CODE=$proc_exit
        fi
    done
else
    # Single-process execution (original behavior)
    python -m mussel.cli.tessellate_extract_features "${CMD_ARGS[@]:1}"
    EXIT_CODE=$?
fi

DURATION=$(($(date +%s) - START_TIME))

echo ""
if [ $EXIT_CODE -ne 0 ]; then
    log "ERROR: Processing failed with exit code $EXIT_CODE (duration: $DURATION seconds)"
    exit $EXIT_CODE
fi

log "SUCCESS: Processing completed in $DURATION seconds"

# Upload output files to remote storage if needed
if [ "$IS_REMOTE_OUTPUT" = true ]; then
    log "Uploading output files to remote storage: $REMOTE_OUTPUT_DIR"
    UPLOAD_START=$(date +%s)
    
    # Determine upload method based on remote path type
    if [[ "$REMOTE_OUTPUT_DIR" =~ ^(azblob://|az://) ]]; then
        # Azure Blob Storage upload
        log "Using az CLI for Azure Blob upload"
        
        # Convert az:// to azblob:// format for az storage
        BLOB_URL="$REMOTE_OUTPUT_DIR"
        if [[ "$BLOB_URL" =~ ^az:// ]]; then
            # az://container/path -> need to reconstruct with account name
            CONTAINER_PATH="${BLOB_URL#az://}"
            BLOB_URL="https://${AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER_PATH}"
        fi
        
        # Upload all files in output directory
        if command -v azcopy &> /dev/null && [ -n "$AZURE_STORAGE_KEY" ]; then
            log "Using azcopy for batch upload"
            # Upload entire directory with azcopy
            azcopy copy "$LOCAL_OUTPUT_DIR/*" "$BLOB_URL" --recursive --overwrite=true
            UPLOAD_EXIT=$?
        elif command -v az &> /dev/null && [ -n "$AZURE_STORAGE_KEY" ]; then
            log "Using az storage blob upload-batch"
            # Extract container and path from URL
            if [[ "$REMOTE_OUTPUT_DIR" =~ ^az://([^/]+)/(.*)$ ]]; then
                CONTAINER="${BASH_REMATCH[1]}"
                BLOB_PREFIX="${BASH_REMATCH[2]}"
                
                # Upload all files
                az storage blob upload-batch \
                    --account-name "$AZURE_STORAGE_ACCOUNT" \
                    --account-key "$AZURE_STORAGE_KEY" \
                    --destination "$CONTAINER" \
                    --destination-path "$BLOB_PREFIX" \
                    --source "$LOCAL_OUTPUT_DIR" \
                    --overwrite true
                UPLOAD_EXIT=$?
            else
                log "ERROR: Could not parse Azure Blob path: $REMOTE_OUTPUT_DIR"
                UPLOAD_EXIT=1
            fi
        else
            log "ERROR: No Azure upload tool available (need azcopy or az CLI with credentials)"
            UPLOAD_EXIT=1
        fi
        
    elif [[ "$REMOTE_OUTPUT_DIR" =~ ^s3:// ]]; then
        # S3 upload
        log "Using aws CLI for S3 upload"
        if command -v aws &> /dev/null; then
            aws s3 sync "$LOCAL_OUTPUT_DIR" "$REMOTE_OUTPUT_DIR" --no-progress
            UPLOAD_EXIT=$?
        else
            log "ERROR: aws CLI not available for S3 upload"
            UPLOAD_EXIT=1
        fi
    else
        log "WARNING: Unknown remote storage type: $REMOTE_OUTPUT_DIR"
        UPLOAD_EXIT=1
    fi
    
    UPLOAD_DURATION=$(($(date +%s) - UPLOAD_START))
    
    if [ $UPLOAD_EXIT -eq 0 ]; then
        log "SUCCESS: Files uploaded to $REMOTE_OUTPUT_DIR in $UPLOAD_DURATION seconds"
        
        # Clean up local output directory
        log "Cleaning up local output directory: $LOCAL_OUTPUT_DIR"
        rm -rf "$LOCAL_OUTPUT_DIR"
    else
        log "ERROR: Failed to upload files to remote storage (exit code: $UPLOAD_EXIT)"
        log "Local files preserved at: $LOCAL_OUTPUT_DIR"
        exit $UPLOAD_EXIT
    fi
else
    log "Output written to: $EFFECTIVE_OUTPUT_DIR"
fi

echo "============================================"
echo "End time: $(date)"
echo "============================================"

exit 0
