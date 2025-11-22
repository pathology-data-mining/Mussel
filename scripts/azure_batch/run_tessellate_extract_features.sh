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

# Debug: Print environment variables
echo "DEBUG: Environment variables related to MODEL_BATCH_SIZES:" >&2
env | grep -i "batch\|model" | sort >&2
echo "" >&2

# Function to log with timestamp (send to stderr to not interfere with function return values)
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >&2
}

# Cleanup function - only cleanup work directory, NOT cache or output directories
cleanup() {
    local exit_code=$?
    
    # Only cleanup temporary work directory (staged slides)
    # DO NOT cleanup output directory - Azure Batch needs it for output file staging
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
    
    log "Cleanup: Model cache directories preserved for reuse across tasks"
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

# Check if SLIDE_PATHS or legacy SLIDE_PATH is set
if [ -n "$SLIDE_PATHS" ]; then
    # Batch mode - process one or more slides
    log "Processing slides: ${SLIDE_PATHS}"
elif [ -n "$SLIDE_PATH" ]; then
    # Legacy single slide mode - convert to SLIDE_PATHS format for uniform processing
    log "Converting legacy SLIDE_PATH to SLIDE_PATHS format"
    SLIDE_PATHS="$SLIDE_PATH"
else
    log "ERROR: Neither SLIDE_PATH nor SLIDE_PATHS is set"
    exit 1
fi

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

# Join staged paths with commas and wrap in brackets for Hydra list format
EFFECTIVE_SLIDE_PATH="[$(IFS=,; echo "${STAGED_PATHS[*]}")]"

# Set HuggingFace token
[ -n "$HF_TOKEN" ] && export HUGGINGFACE_TOKEN="$HF_TOKEN"

# Setup output directory
# Always write to local directory - Azure Batch will handle upload via output file staging
# IMPORTANT: Always use "output" as the local directory, ignoring any remote paths from config
# Azure Batch output file staging will automatically upload files to the correct blob storage location
OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"
log "Using local output directory: $OUTPUT_DIR (relative to $(pwd))"
log "Azure Batch will automatically upload files on task success"

# Build command arguments - pass everything through to Python CLI
CMD_ARGS=(
    "tessellate_extract_features"
    "hydra.run.dir=/mnt/batch/tasks/workitems/tmp/hydra_outputs"
    "output_dir=$OUTPUT_DIR"
)

# Slide paths - always use slide_paths (with brackets) for consistency
CMD_ARGS+=("slide_paths=$EFFECTIVE_SLIDE_PATH")

# Model configuration - add brackets for lists (comma-separated values)
[ -n "$MODEL_TYPES" ] && CMD_ARGS+=("model_type=[$MODEL_TYPES]")
[ -n "$MODEL_TYPE" ] && CMD_ARGS+=("model_type=$MODEL_TYPE")
[ -n "$MODEL_PATH" ] && CMD_ARGS+=("model_path=$MODEL_PATH")
[ -n "$MODEL_DIR" ] && CMD_ARGS+=("model_dir=$MODEL_DIR")
[ -n "$SLIDE_MODEL_TYPES" ] && CMD_ARGS+=("slide_model_type=[$SLIDE_MODEL_TYPES]")
[ -n "$SLIDE_MODEL_TYPE" ] && CMD_ARGS+=("slide_model_type=$SLIDE_MODEL_TYPE")
[ -n "$SLIDE_MODEL_PATH" ] && CMD_ARGS+=("slide_model_path=$SLIDE_MODEL_PATH")

# Processing parameters
[ -n "$NUM_WORKERS" ] && CMD_ARGS+=("num_workers=$NUM_WORKERS")
[ -n "$BATCH_SIZE" ] && CMD_ARGS+=("batch_size=$BATCH_SIZE")
[ -n "$SLIDE_BATCH_SIZE" ] && CMD_ARGS+=("slide_batch_size=$SLIDE_BATCH_SIZE")
[ -n "$USE_GPU" ] && CMD_ARGS+=("use_gpu=$USE_GPU")

# Model-specific batch sizes (passed as JSON string, convert to Hydra overrides)
log "DEBUG: MODEL_BATCH_SIZES value: '${MODEL_BATCH_SIZES}'"
log "DEBUG: MODEL_BATCH_SIZES length: ${#MODEL_BATCH_SIZES}"
if [ -n "$MODEL_BATCH_SIZES" ]; then
    log "Parsing MODEL_BATCH_SIZES: $MODEL_BATCH_SIZES"
    # Parse JSON and convert to Hydra overrides: +model_batch_sizes.MODEL=SIZE
    while IFS= read -r line; do
        model=$(echo "$line" | cut -d: -f1 | tr -d ' "')
        size=$(echo "$line" | cut -d: -f2 | tr -d ' ,')
        CMD_ARGS+=("+model_batch_sizes.$model=$size")
        log "  $model: $size"
    done < <(echo "$MODEL_BATCH_SIZES" | tr -d '{}' | tr ',' '\n')
else
    log "DEBUG: MODEL_BATCH_SIZES is empty or not set"
fi

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
if [ $NUM_GPUS -gt 1 ]; then
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
log "Output files will be automatically uploaded by Azure Batch"

echo "============================================"
echo "End time: $(date)"
echo "============================================"

exit 0
