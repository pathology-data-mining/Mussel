#!/bin/bash
#
# Azure Batch task script for running tessellate-extract-features
# This script:
# 1. Stages input files from Azure/S3 to local storage if needed
# 2. Uses persistent model cache to avoid re-downloading models
# 3. Runs processing with output directly to OUTPUT_DIR (supports remote paths)
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

# Download scripts from Azure Blob if SCRIPT_BLOB_URL is provided
# This allows updating scripts without rebuilding containers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "$SCRIPT_BLOB_URL" ]; then
    log "Downloading scripts from Azure Blob: $SCRIPT_BLOB_URL"
    
    # Create temporary directory for downloaded scripts
    DOWNLOADED_SCRIPTS_DIR="/tmp/batch_scripts_$$"
    mkdir -p "$DOWNLOADED_SCRIPTS_DIR"
    
    # Download scripts using Azure CLI or wget/curl
    if [ -n "$AZURE_STORAGE_ACCOUNT" ] && [ -n "$AZURE_STORAGE_KEY" ] && command -v az &> /dev/null; then
        # Use Azure CLI with storage account credentials
        # Expected format: https://<account>.blob.core.windows.net/<container>/<path>/
        # Extract path after blob.core.windows.net/
        URL_PATH=$(echo "$SCRIPT_BLOB_URL" | sed 's|^https://[^/]*/||')
        CONTAINER=$(echo "$URL_PATH" | cut -d'/' -f1)
        BLOB_PREFIX=$(echo "$URL_PATH" | cut -d'/' -f2-)
        
        log "Downloading from container: $CONTAINER, prefix: $BLOB_PREFIX"
        
        # Download required scripts
        for script in "run_tessellate_extract_features.sh" "persistent_model_cache.sh"; do
            BLOB_PATH="${BLOB_PREFIX}${script}"
            az storage blob download \
                --account-name "$AZURE_STORAGE_ACCOUNT" \
                --account-key "$AZURE_STORAGE_KEY" \
                --container-name "$CONTAINER" \
                --name "$BLOB_PATH" \
                --file "$DOWNLOADED_SCRIPTS_DIR/$script" \
                --no-progress 2>&1 | grep -v "^$" || log "Warning: Failed to download $script"
            
            if [ -f "$DOWNLOADED_SCRIPTS_DIR/$script" ]; then
                chmod +x "$DOWNLOADED_SCRIPTS_DIR/$script"
                log "Downloaded: $script"
            fi
        done
        
        # Use downloaded scripts directory
        SCRIPT_DIR="$DOWNLOADED_SCRIPTS_DIR"
        log "Using scripts from: $SCRIPT_DIR"
    else
        log "Warning: Cannot download scripts (missing Azure credentials or az CLI)"
        log "Using bundled scripts from container image"
    fi
fi

# Load persistent model caching functions
if [ -f "$SCRIPT_DIR/persistent_model_cache.sh" ]; then
    source "$SCRIPT_DIR/persistent_model_cache.sh"
    log "Persistent model cache enabled"
    show_cache_stats
else
    log "Warning: persistent_model_cache.sh not found, models will be downloaded to temp"
fi

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
    # IMPORTANT: Do NOT cleanup persistent model cache at /mnt/batch_models
    # Only cleanup temporary HuggingFace/Torch caches
    local cache_dirs=()
    local persistent_cache="/mnt/batch_models"
    
    # Add environment variable paths if set (but exclude persistent cache)
    [ -n "$HF_HOME" ] && [ "$HF_HOME" != "$persistent_cache" ] && [ "$HF_HOME" != "$MODEL_CACHE_DIR" ] && cache_dirs+=("$HF_HOME")
    [ -n "$TRANSFORMERS_CACHE" ] && [ "$TRANSFORMERS_CACHE" != "$persistent_cache" ] && [ "$TRANSFORMERS_CACHE" != "$MODEL_CACHE_DIR" ] && cache_dirs+=("$TRANSFORMERS_CACHE")
    [ -n "$TORCH_HOME" ] && [ "$TORCH_HOME" != "$persistent_cache" ] && [ "$TORCH_HOME" != "$MODEL_CACHE_DIR" ] && cache_dirs+=("$TORCH_HOME")
    
    # Add known cache locations used by Azure Batch (but exclude persistent cache)
    cache_dirs+=(
        "/mnt/batch/tasks/workitems/hf_cache"
        "/mnt/batch/tasks/workitems/torch_cache"
        "/root/.cache/huggingface"
        "/root/.cache/torch"
    )
    
    # Cleanup downloaded scripts directory if it exists
    if [ -n "$DOWNLOADED_SCRIPTS_DIR" ] && [ -d "$DOWNLOADED_SCRIPTS_DIR" ]; then
        log "Cleanup: Removing downloaded scripts directory: $DOWNLOADED_SCRIPTS_DIR"
        rm -rf "$DOWNLOADED_SCRIPTS_DIR" || log "Warning: Failed to remove downloaded scripts directory"
    fi
    
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

# Function to stage model with persistent cache support
stage_model_with_cache() {
    local remote_path="$1"
    local model_type="$2"  # e.g., "prefilter", "model", "slide_model"
    
    # Check if persistent cache is available
    if type get_or_cache_model &>/dev/null; then
        # Use persistent cache
        log "Using persistent cache for: $model_type"
        local model_name="${model_type}_$(basename "$remote_path" | sed 's/[^a-zA-Z0-9._-]/_/g')"
        local cached_path=$(get_or_cache_model "$remote_path" "$model_name")
        if [ $? -eq 0 ] && [ -d "$cached_path" ]; then
            log "Model retrieved from persistent cache: $cached_path"
            echo "$cached_path"
            return 0
        else
            log "Warning: Failed to use persistent cache, falling back to temp download"
        fi
    fi
    
    # Fallback: Download to temporary location
    log "Downloading to temporary location: $remote_path"
    local local_model_path="$WORK_DIR/models/$(basename "$remote_path")"
    mkdir -p "$(dirname "$local_model_path")"
    staged_path=$(stage_remote_file "$remote_path" "$local_model_path")
    echo "$staged_path"
}

# Stage models from remote storage if needed
log "Checking model paths for remote staging..."
mkdir -p "$WORK_DIR"

# Stage PREFILTER_MODEL_PATH if it's remote
if [ -n "$PREFILTER_MODEL_PATH" ] && [[ "$PREFILTER_MODEL_PATH" =~ ^(azfiles|s3|azblob):// ]]; then
    log "Staging prefilter model from remote: $PREFILTER_MODEL_PATH"
    # Use prefilter model type name if available, otherwise generic name
    model_cache_name="${PREFILTER_MODEL_TYPES:-prefilter}"
    PREFILTER_MODEL_PATH=$(stage_model_with_cache "$PREFILTER_MODEL_PATH" "$model_cache_name")
    log "Staged prefilter model to: $PREFILTER_MODEL_PATH"
fi

# Stage MODEL_PATH if it's remote
if [ -n "$MODEL_PATH" ] && [[ "$MODEL_PATH" =~ ^(azfiles|s3|azblob):// ]]; then
    log "Staging model from remote: $MODEL_PATH"
    # Use model type name if available, otherwise generic name
    model_cache_name="${MODEL_TYPES:-${MODEL_TYPE:-model}}"
    MODEL_PATH=$(stage_model_with_cache "$MODEL_PATH" "$model_cache_name")
    log "Staged model to: $MODEL_PATH"
fi

# Stage SLIDE_MODEL_PATH if it's remote
if [ -n "$SLIDE_MODEL_PATH" ] && [[ "$SLIDE_MODEL_PATH" =~ ^(azfiles|s3|azblob):// ]]; then
    log "Staging slide model from remote: $SLIDE_MODEL_PATH"
    # Use slide model type name if available, otherwise generic name
    model_cache_name="${SLIDE_MODEL_TYPES:-${SLIDE_MODEL_TYPE:-slide_model}}"
    SLIDE_MODEL_PATH=$(stage_model_with_cache "$SLIDE_MODEL_PATH" "$model_cache_name")
    log "Staged slide model to: $SLIDE_MODEL_PATH"
fi

# Stage SLIDE_MODEL_PATHS (multiple models) if provided
if [ -n "$SLIDE_MODEL_PATHS" ]; then
    log "Staging multiple slide models: $SLIDE_MODEL_PATHS"
    IFS=',' read -ra MODEL_PATH_ARRAY <<< "$SLIDE_MODEL_PATHS"
    
    # Also parse model types to use proper names in cache
    if [ -n "$SLIDE_MODEL_TYPES" ]; then
        IFS=',' read -ra MODEL_TYPE_ARRAY <<< "$SLIDE_MODEL_TYPES"
    else
        MODEL_TYPE_ARRAY=()
    fi
    
    STAGED_MODEL_PATHS=()
    
    for idx in "${!MODEL_PATH_ARRAY[@]}"; do
        model_path=$(echo "${MODEL_PATH_ARRAY[$idx]}" | xargs)  # Trim whitespace
        
        # Get model type name if available, otherwise use generic name
        if [ ${#MODEL_TYPE_ARRAY[@]} -gt $idx ]; then
            model_name=$(echo "${MODEL_TYPE_ARRAY[$idx]}" | xargs)
        else
            model_name="slide_model_${idx}"
        fi
        
        if [[ "$model_path" =~ ^(azfiles|s3|azblob):// ]]; then
            log "  Staging model $((idx+1))/${#MODEL_PATH_ARRAY[@]} ($model_name): $model_path"
            staged_path=$(stage_model_with_cache "$model_path" "$model_name")
            STAGED_MODEL_PATHS+=("$staged_path")
            log "  Staged to: $staged_path"
        else
            STAGED_MODEL_PATHS+=("$model_path")
        fi
    done
    
    # Rebuild SLIDE_MODEL_PATHS with staged paths
    SLIDE_MODEL_PATHS="$(IFS=,; echo "${STAGED_MODEL_PATHS[*]}")"
    log "Updated SLIDE_MODEL_PATHS: $SLIDE_MODEL_PATHS"
fi

# Stage CLASSIFIER_PKL if it's remote
if [ -n "$CLASSIFIER_PKL" ] && [[ "$CLASSIFIER_PKL" =~ ^(azfiles|s3|azblob):// ]]; then
    log "Staging classifier from remote: $CLASSIFIER_PKL"
    CLASSIFIER_PKL=$(stage_model_with_cache "$CLASSIFIER_PKL" "classifier")
    log "Staged classifier to: $CLASSIFIER_PKL"
fi

# Handle MODEL_CACHE_DIR (model_dir parameter for multi-model mode)
# If MODEL_CACHE_DIR is provided as remote URL, download to persistent cache
# Otherwise use it directly
PERSISTENT_CACHE_DIR="/mnt/batch_models"

if [ -n "$MODEL_CACHE_DIR" ]; then
    log "MODEL_CACHE_DIR specified: $MODEL_CACHE_DIR"
    
    # Check if it's a remote URL
    if [[ "$MODEL_CACHE_DIR" =~ ^(azblob://|azfiles://|s3://) ]]; then
        log "Remote MODEL_CACHE_DIR detected: $MODEL_CACHE_DIR"
        
        # Check if models already exist in persistent cache
        if [ -d "$PERSISTENT_CACHE_DIR" ] && [ "$(find "$PERSISTENT_CACHE_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)" -gt 0 ]; then
            log "Models already exist in persistent cache: $PERSISTENT_CACHE_DIR"
            log "  Contains $(find "$PERSISTENT_CACHE_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) model directories"
            log "Skipping download"
            MODEL_CACHE_DIR="$PERSISTENT_CACHE_DIR"
        else
            log "Persistent cache empty, downloading to: $PERSISTENT_CACHE_DIR"
            
            # Download model cache directory from remote storage
            if [[ "$MODEL_CACHE_DIR" =~ ^azblob:// ]]; then
                # Azure Blob Storage
                BLOB_URL=$(echo "$MODEL_CACHE_DIR" | sed 's|^azblob://||')
                STORAGE_ACCOUNT=$(echo "$BLOB_URL" | cut -d'/' -f1 | cut -d'.' -f1)
                CONTAINER=$(echo "$BLOB_URL" | cut -d'/' -f2)
                BLOB_PREFIX=$(echo "$BLOB_URL" | cut -d'/' -f3-)
                
                log "Downloading model cache from Azure Blob:"
                log "  Account: $STORAGE_ACCOUNT"
                log "  Container: $CONTAINER"
                log "  Prefix: $BLOB_PREFIX"
                
                START=$(date +%s)
                if command -v az &> /dev/null && [ -n "$AZURE_STORAGE_KEY" ]; then
                    # Download model cache directory from Azure Blob
                    mkdir -p "$PERSISTENT_CACHE_DIR"
                    chmod 777 "$PERSISTENT_CACHE_DIR"
                    # Verify directory exists and is writable
                    if [ ! -d "$PERSISTENT_CACHE_DIR" ] || [ ! -w "$PERSISTENT_CACHE_DIR" ]; then
                        log "ERROR: Cannot create or write to $PERSISTENT_CACHE_DIR"
                        MODEL_CACHE_DIR=""
                    else
                        # Pattern should include the blob prefix and wildcard
                        if az storage blob download-batch \
                            --account-name "$STORAGE_ACCOUNT" \
                            --account-key "$AZURE_STORAGE_KEY" \
                            --source "$CONTAINER" \
                            --destination "$PERSISTENT_CACHE_DIR" \
                            --pattern "${BLOB_PREFIX}/*" \
                            --max-connections 16 \
                            --no-progress 2>&1 | grep -v "^$"; then
                            DURATION=$(($(date +%s) - START))
                            log "Model cache downloaded in $DURATION seconds"
                            # Files are downloaded with the prefix path, need to move them up
                            if [ -d "$PERSISTENT_CACHE_DIR/$BLOB_PREFIX" ]; then
                                log "Moving models from $PERSISTENT_CACHE_DIR/$BLOB_PREFIX to $PERSISTENT_CACHE_DIR"
                                mv "$PERSISTENT_CACHE_DIR/$BLOB_PREFIX"/* "$PERSISTENT_CACHE_DIR/" 2>/dev/null
                                rmdir "$PERSISTENT_CACHE_DIR/$BLOB_PREFIX" 2>/dev/null || true
                            fi
                            MODEL_CACHE_DIR="$PERSISTENT_CACHE_DIR"
                        else
                            log "ERROR: Failed to download model cache from Azure Blob"
                            log "Will fall back to downloading models from HuggingFace"
                            MODEL_CACHE_DIR=""
                        fi
                    fi
                else
                    log "ERROR: az CLI or AZURE_STORAGE_KEY not available"
                    log "Will fall back to downloading models from HuggingFace"
                    MODEL_CACHE_DIR=""
                fi
            else
                log "WARNING: Remote storage type not yet supported for MODEL_CACHE_DIR: $MODEL_CACHE_DIR"
                log "Will fall back to downloading models from HuggingFace"
                MODEL_CACHE_DIR=""
            fi
        fi
    elif [ "$MODEL_CACHE_DIR" = "$PERSISTENT_CACHE_DIR" ] || [ -d "$MODEL_CACHE_DIR" ]; then
        # Local directory
        log "Using model_dir: $MODEL_CACHE_DIR"
        if [ -d "$MODEL_CACHE_DIR" ]; then
            log "  Directory exists with $(find "$MODEL_CACHE_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) model directories"
        fi
    else
        log "WARNING: MODEL_CACHE_DIR specified but directory does not exist: $MODEL_CACHE_DIR"
        log "Will fall back to downloading models from HuggingFace"
        MODEL_CACHE_DIR=""
    fi
fi

# If MODEL_CACHE_DIR is set (either from parameter or discovered), use it
if [ -n "$MODEL_CACHE_DIR" ] && [ -d "$MODEL_CACHE_DIR" ]; then
    # Check if the directory actually has models in it
    MODEL_COUNT=$(find "$MODEL_CACHE_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    
    if [ "$MODEL_COUNT" -gt 0 ]; then
        log "Using model_dir with $MODEL_COUNT model directories: $MODEL_CACHE_DIR"
        # Don't set HF_HOME/TRANSFORMERS_CACHE/TORCH_HOME when using model_dir
        # The model_dir parameter tells the CLI to load models from local directories
        # Setting these env vars would cause HuggingFace to cache in the wrong location
    else
        log "WARNING: MODEL_CACHE_DIR specified but empty: $MODEL_CACHE_DIR"
        log "Models will be downloaded from HuggingFace to persistent cache"
        # Set cache variables to persistent location for new downloads
        export HF_HOME="$MODEL_CACHE_DIR"
        export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
        export TORCH_HOME="$MODEL_CACHE_DIR/torch_cache"
        mkdir -p "$HF_HOME" "$TORCH_HOME"
        log "Set HuggingFace cache to: $HF_HOME"
        # Clear MODEL_CACHE_DIR so CLI doesn't try to use model_dir parameter with empty directory
        MODEL_CACHE_DIR=""
    fi
else
    # No MODEL_CACHE_DIR - check for existing persistent cache
    if [ -d "$PERSISTENT_CACHE_DIR" ] && [ "$(find "$PERSISTENT_CACHE_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)" -gt 0 ]; then
        log "No MODEL_CACHE_DIR specified, using existing persistent cache: $PERSISTENT_CACHE_DIR"
        MODEL_CACHE_DIR="$PERSISTENT_CACHE_DIR"
        log "Using model_dir with $(find "$MODEL_CACHE_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) model directories"
    else
        log "No MODEL_CACHE_DIR available, models will be downloaded from HuggingFace"
        # Set cache variables to persistent location for new downloads
        export HF_HOME="$PERSISTENT_CACHE_DIR/hf_cache"
        export TRANSFORMERS_CACHE="$PERSISTENT_CACHE_DIR/hf_cache"
        export TORCH_HOME="$PERSISTENT_CACHE_DIR/torch_cache"
        mkdir -p "$HF_HOME" "$TORCH_HOME"
        log "Set HuggingFace cache to: $HF_HOME"
        MODEL_CACHE_DIR=""
    fi
fi

log "Model staging complete"

# Setup output directories
# Store the remote output path for later upload
REMOTE_OUTPUT_DIR="${OUTPUT_DIR:-/mnt/batch/tasks/shared/output}"
IS_REMOTE_OUTPUT=false

# Check if output is remote (azblob://, az://, s3://, etc.)
if [[ "$REMOTE_OUTPUT_DIR" =~ ^(azblob://|az://|s3://|http://|https://) ]]; then
    log "Remote output detected: $REMOTE_OUTPUT_DIR"
    log "Will write to local storage first, then copy to remote at task end"
    IS_REMOTE_OUTPUT=true
    # Use local directory for processing
    LOCAL_OUTPUT_DIR="${WORK_DIR}/output"
    EFFECTIVE_OUTPUT_DIR="$LOCAL_OUTPUT_DIR"
else
    # Local output path - use directly
    EFFECTIVE_OUTPUT_DIR="$REMOTE_OUTPUT_DIR"
    LOCAL_OUTPUT_DIR="$REMOTE_OUTPUT_DIR"
fi

# Create local output directory
mkdir -p "$LOCAL_OUTPUT_DIR"

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

# Pass model_dir if MODEL_CACHE_DIR is set and exists (for multi-model mode)
if [ -n "$MODEL_CACHE_DIR" ] && [ -d "$MODEL_CACHE_DIR" ]; then
    CMD_ARGS+=("model_dir=$MODEL_CACHE_DIR")
    log "Using model_dir: $MODEL_CACHE_DIR"
fi

# Processing parameters
[ -n "$NUM_WORKERS" ] && CMD_ARGS+=("num_workers=$NUM_WORKERS")
[ -n "$BATCH_SIZE" ] && CMD_ARGS+=("batch_size=$BATCH_SIZE")
[ -n "$SLIDE_BATCH_SIZE" ] && CMD_ARGS+=("slide_batch_size=$SLIDE_BATCH_SIZE")
[ -n "$USE_GPU" ] && CMD_ARGS+=("use_gpu=$USE_GPU")

# Model batch sizes (per-model batch size configuration)
# Parse MODEL_BATCH_SIZES JSON and convert to Hydra CLI format
if [ -n "$MODEL_BATCH_SIZES" ]; then
    log "Parsing MODEL_BATCH_SIZES: $MODEL_BATCH_SIZES"
    # Use python to parse JSON and output Hydra-compatible arguments
    BATCH_SIZE_ARGS=$(python3 -c "
import json
import sys
try:
    sizes = json.loads('$MODEL_BATCH_SIZES')
    for model, batch_size in sizes.items():
        print(f'+model_batch_sizes.{model}={batch_size}')
except Exception as e:
    sys.stderr.write(f'Error parsing MODEL_BATCH_SIZES: {e}\n')
    sys.exit(1)
" 2>&1)
    
    if [ $? -eq 0 ]; then
        # Read each line and add to CMD_ARGS
        while IFS= read -r arg; do
            [ -n "$arg" ] && CMD_ARGS+=("$arg")
        done <<< "$BATCH_SIZE_ARGS"
        log "Added per-model batch sizes to command arguments"
    else
        log "WARNING: Failed to parse MODEL_BATCH_SIZES, using default batch_size"
    fi
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

# Save features to H5 (default: false for disk space savings)
if [ -n "$SAVE_FEATURES_TO_H5" ]; then
    CMD_ARGS+=("save_features_to_h5=$SAVE_FEATURES_TO_H5")
else
    # Default to false (coords-only H5, features in PT)
    CMD_ARGS+=("save_features_to_h5=false")
fi

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

# Copy output files to remote storage if needed
if [ "$IS_REMOTE_OUTPUT" = true ]; then
    log "Copying output files to remote storage: $REMOTE_OUTPUT_DIR"
    
    # Parse remote path to determine upload method
    if [[ "$REMOTE_OUTPUT_DIR" =~ ^(azblob://|az://) ]]; then
        # Azure Blob upload - handle both azblob:// and az:// prefixes
        BLOB_URL=$(echo "$REMOTE_OUTPUT_DIR" | sed -E 's#^(azblob://|az://)##')
        
        # Parse URL - format can be:
        # - az://container/path
        # - azblob://account.blob.core.windows.net/container/path
        if [[ "$BLOB_URL" =~ \. ]]; then
            # Full URL format: account.blob.core.windows.net/container/path
            STORAGE_ACCOUNT=$(echo "$BLOB_URL" | cut -d'/' -f1 | cut -d'.' -f1)
            CONTAINER=$(echo "$BLOB_URL" | cut -d'/' -f2)
            BLOB_PREFIX=$(echo "$BLOB_URL" | cut -d'/' -f3-)
        else
            # Short format: container/path (use AZURE_STORAGE_ACCOUNT env var)
            STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-mskpdmgen2}"
            CONTAINER=$(echo "$BLOB_URL" | cut -d'/' -f1)
            BLOB_PREFIX=$(echo "$BLOB_URL" | cut -d'/' -f2-)
            # If no slash, BLOB_PREFIX will equal CONTAINER - fix it
            if [ "$BLOB_PREFIX" = "$CONTAINER" ]; then
                BLOB_PREFIX=""
            fi
        fi
        
        log "Uploading to Azure Blob: account=$STORAGE_ACCOUNT, container=$CONTAINER, prefix=$BLOB_PREFIX"
        
        UPLOAD_START=$(date +%s)
        if command -v az &> /dev/null; then
            # Use azcopy for directory upload if available
            if command -v azcopy &> /dev/null && [ -n "$AZURE_STORAGE_KEY" ]; then
                log "Using azcopy for bulk upload"
                export AZCOPY_AUTO_LOGIN_TYPE=SPN
                if azcopy copy "$LOCAL_OUTPUT_DIR/*" "https://${STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER}/${BLOB_PREFIX}?${AZURE_STORAGE_KEY}" --recursive; then
                    UPLOAD_SUCCESS=true
                fi
            else
                # Fall back to az storage blob upload-batch
                log "Using az storage blob upload-batch"
                if [ -n "$AZURE_STORAGE_KEY" ]; then
                    if az storage blob upload-batch \
                        --account-name "$STORAGE_ACCOUNT" \
                        --account-key "$AZURE_STORAGE_KEY" \
                        --destination "$CONTAINER" \
                        --destination-path "$BLOB_PREFIX" \
                        --source "$LOCAL_OUTPUT_DIR" \
                        --pattern "*" \
                        --overwrite true 2>&1 | grep -v "^$"; then
                        UPLOAD_SUCCESS=true
                    fi
                else
                    if az storage blob upload-batch \
                        --account-name "$STORAGE_ACCOUNT" \
                        --destination "$CONTAINER" \
                        --destination-path "$BLOB_PREFIX" \
                        --source "$LOCAL_OUTPUT_DIR" \
                        --pattern "*" \
                        --auth-mode login \
                        --overwrite true 2>&1 | grep -v "^$"; then
                        UPLOAD_SUCCESS=true
                    fi
                fi
            fi
            
            if [ "$UPLOAD_SUCCESS" = true ]; then
                UPLOAD_DURATION=$(($(date +%s) - UPLOAD_START))
                log "Upload completed in $UPLOAD_DURATION seconds"
                
                # Get total size uploaded
                TOTAL_SIZE=$(du -sh "$LOCAL_OUTPUT_DIR" 2>/dev/null | cut -f1)
                log "Total size uploaded: $TOTAL_SIZE"
            else
                log "ERROR: Failed to upload output files to Azure Blob"
                exit 1
            fi
        else
            log "ERROR: az CLI not available for Azure Blob upload"
            exit 1
        fi
        
    elif [[ "$REMOTE_OUTPUT_DIR" =~ ^s3:// ]]; then
        # S3 upload
        log "Uploading to S3: $REMOTE_OUTPUT_DIR"
        UPLOAD_START=$(date +%s)
        if command -v aws &> /dev/null; then
            if aws s3 cp "$LOCAL_OUTPUT_DIR" "$REMOTE_OUTPUT_DIR" --recursive; then
                UPLOAD_DURATION=$(($(date +%s) - UPLOAD_START))
                log "Upload completed in $UPLOAD_DURATION seconds"
            else
                log "ERROR: Failed to upload output files to S3"
                exit 1
            fi
        else
            log "ERROR: aws CLI not available for S3 upload"
            exit 1
        fi
    else
        log "WARNING: Unsupported remote storage scheme: $REMOTE_OUTPUT_DIR"
        log "Output left in local directory: $LOCAL_OUTPUT_DIR"
    fi
    
    log "Output files uploaded to: $REMOTE_OUTPUT_DIR"
else
    log "Output written to: $EFFECTIVE_OUTPUT_DIR"
fi

echo "============================================"
echo "End time: $(date)"
echo "============================================"

exit 0
