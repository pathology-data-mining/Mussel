#!/bin/bash
#
# Example script for submitting Mussel jobs to Azure Batch
# 
# This script demonstrates common usage patterns for processing
# whole-slide images on Azure Batch.
#
# Prerequisites:
# - Azure Batch account created
# - Azure credentials configured
# - Python dependencies installed: pip install azure-batch azure-storage-blob azure-identity
#

set -e

# ==============================================================================
# Configuration - UPDATE THESE VALUES
# ==============================================================================

# Azure Batch account details
BATCH_ACCOUNT_NAME="mybatchaccount"
BATCH_ACCOUNT_KEY="your-batch-account-key-here"
BATCH_ACCOUNT_URL="https://mybatchaccount.eastus.batch.azure.com"

# Optional: Azure Storage account (for storing slides and results)
STORAGE_ACCOUNT_NAME="mystorageaccount"
STORAGE_ACCOUNT_KEY="your-storage-account-key-here"

# Pool configuration
POOL_ID="mussel-pool-$(date +%Y%m%d)"
VM_SIZE="Standard_NC6s_v3"  # GPU-enabled VM
NODE_COUNT=2  # Number of VMs in the pool

# Job configuration
JOB_ID="mussel-job-$(date +%Y%m%d-%H%M%S)"

# Docker image
CONTAINER_IMAGE="mskmind/mussel:latest-torch-gpu"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================================================================
# Helper Functions
# ==============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

check_credentials() {
    if [ "$BATCH_ACCOUNT_KEY" = "your-batch-account-key-here" ]; then
        log "ERROR: Please update BATCH_ACCOUNT_KEY in this script"
        exit 1
    fi
}

# ==============================================================================
# Example 1: Process a single slide
# ==============================================================================

example_single_slide() {
    log "Example 1: Processing a single slide"
    
    python "$SCRIPT_DIR/submit_batch_jobs.py" \
        --batch-account-name "$BATCH_ACCOUNT_NAME" \
        --batch-account-key "$BATCH_ACCOUNT_KEY" \
        --batch-account-url "$BATCH_ACCOUNT_URL" \
        --pool-id "$POOL_ID" \
        --create-pool \
        --vm-size "$VM_SIZE" \
        --node-count 1 \
        --container-image "$CONTAINER_IMAGE" \
        --job-id "$JOB_ID" \
        --create-job \
        --task-id "slide-001" \
        --slide-path "/mnt/data/slides/slide_001.svs" \
        --output-h5-path "/mnt/output/slide_001_features.h5" \
        --output-pt-path "/mnt/output/slide_001_features.pt" \
        --monitor
    
    log "Single slide processing complete"
}

# ==============================================================================
# Example 2: Process multiple slides from a config file
# ==============================================================================

example_batch_processing() {
    log "Example 2: Batch processing multiple slides"
    
    # Create a temporary config file
    CONFIG_FILE="/tmp/batch_config_$(date +%s).json"
    
    cat > "$CONFIG_FILE" <<'EOF'
{
  "defaults": {
    "prefilter_model_type": "CTRANSPATH",
    "segment_threshold": 0,
    "patch_size": 256,
    "mpp": 0.5,
    "num_workers": 4,
    "batch_size": 64,
    "use_gpu": true,
    "keep_intermediate_files": false
  },
  "tasks": [
    {
      "task_id": "slide_001",
      "slide_path": "/mnt/data/slides/slide_001.svs",
      "output_h5_path": "/mnt/output/slide_001_features.h5",
      "output_pt_path": "/mnt/output/slide_001_features.pt"
    },
    {
      "task_id": "slide_002",
      "slide_path": "/mnt/data/slides/slide_002.svs",
      "output_h5_path": "/mnt/output/slide_002_features.h5",
      "output_pt_path": "/mnt/output/slide_002_features.pt"
    },
    {
      "task_id": "slide_003",
      "slide_path": "/mnt/data/slides/slide_003.svs",
      "output_h5_path": "/mnt/output/slide_003_features.h5",
      "output_pt_path": "/mnt/output/slide_003_features.pt"
    }
  ]
}
EOF
    
    log "Created config file: $CONFIG_FILE"
    
    python "$SCRIPT_DIR/submit_batch_jobs.py" \
        --batch-account-name "$BATCH_ACCOUNT_NAME" \
        --batch-account-key "$BATCH_ACCOUNT_KEY" \
        --batch-account-url "$BATCH_ACCOUNT_URL" \
        --pool-id "$POOL_ID" \
        --create-pool \
        --vm-size "$VM_SIZE" \
        --node-count "$NODE_COUNT" \
        --container-image "$CONTAINER_IMAGE" \
        --job-id "$JOB_ID" \
        --create-job \
        --config-file "$CONFIG_FILE" \
        --monitor
    
    log "Batch processing complete"
    rm -f "$CONFIG_FILE"
}

# ==============================================================================
# Example 3: Process slides with filtering (dual extraction)
# ==============================================================================

example_with_filtering() {
    log "Example 3: Processing with tissue filtering"
    
    CONFIG_FILE="/tmp/batch_config_filtered_$(date +%s).json"
    
    cat > "$CONFIG_FILE" <<'EOF'
{
  "defaults": {
    "prefilter_model_type": "CTRANSPATH",
    "classifier_pkl": "/mnt/data/tissue_classifier.pkl",
    "classifier_threshold": 0.75,
    "postfilter_model_type": "CLIP",
    "segment_threshold": 0,
    "patch_size": 256,
    "mpp": 0.5,
    "num_workers": 4,
    "batch_size": 64,
    "use_gpu": true
  },
  "tasks": [
    {
      "task_id": "slide_001_filtered",
      "slide_path": "/mnt/data/slides/slide_001.svs",
      "output_h5_path": "/mnt/output/slide_001_filtered_features.h5",
      "output_pt_path": "/mnt/output/slide_001_filtered_features.pt"
    }
  ]
}
EOF
    
    log "Created config file with filtering: $CONFIG_FILE"
    
    python "$SCRIPT_DIR/submit_batch_jobs.py" \
        --batch-account-name "$BATCH_ACCOUNT_NAME" \
        --batch-account-key "$BATCH_ACCOUNT_KEY" \
        --batch-account-url "$BATCH_ACCOUNT_URL" \
        --pool-id "$POOL_ID" \
        --create-pool \
        --vm-size "$VM_SIZE" \
        --node-count 1 \
        --container-image "$CONTAINER_IMAGE" \
        --job-id "$JOB_ID" \
        --create-job \
        --config-file "$CONFIG_FILE" \
        --monitor
    
    log "Filtered processing complete"
    rm -f "$CONFIG_FILE"
}

# ==============================================================================
# Example 4: Monitor an existing job
# ==============================================================================

example_monitor_job() {
    log "Example 4: Monitoring an existing job"
    
    read -p "Enter Job ID to monitor: " EXISTING_JOB_ID
    
    python "$SCRIPT_DIR/submit_batch_jobs.py" \
        --batch-account-name "$BATCH_ACCOUNT_NAME" \
        --batch-account-key "$BATCH_ACCOUNT_KEY" \
        --batch-account-url "$BATCH_ACCOUNT_URL" \
        --pool-id "$POOL_ID" \
        --job-id "$EXISTING_JOB_ID" \
        --monitor
}

# ==============================================================================
# Example 5: Cleanup (delete job and pool)
# ==============================================================================

example_cleanup() {
    log "Example 5: Cleaning up resources"
    
    read -p "Enter Job ID to delete (or press Enter to skip): " DELETE_JOB_ID
    read -p "Enter Pool ID to delete (or press Enter to skip): " DELETE_POOL_ID
    
    if [ -n "$DELETE_JOB_ID" ]; then
        log "Deleting job: $DELETE_JOB_ID"
        python "$SCRIPT_DIR/submit_batch_jobs.py" \
            --batch-account-name "$BATCH_ACCOUNT_NAME" \
            --batch-account-key "$BATCH_ACCOUNT_KEY" \
            --batch-account-url "$BATCH_ACCOUNT_URL" \
            --job-id "$DELETE_JOB_ID" \
            --delete-job
    fi
    
    if [ -n "$DELETE_POOL_ID" ]; then
        log "Deleting pool: $DELETE_POOL_ID"
        python "$SCRIPT_DIR/submit_batch_jobs.py" \
            --batch-account-name "$BATCH_ACCOUNT_NAME" \
            --batch-account-key "$BATCH_ACCOUNT_KEY" \
            --batch-account-url "$BATCH_ACCOUNT_URL" \
            --pool-id "$DELETE_POOL_ID" \
            --delete-pool
    fi
    
    log "Cleanup complete"
}

# ==============================================================================
# Main Menu
# ==============================================================================

main() {
    check_credentials
    
    echo ""
    echo "============================================"
    echo "Mussel Azure Batch Example Scripts"
    echo "============================================"
    echo ""
    echo "Select an example to run:"
    echo "  1) Process a single slide"
    echo "  2) Batch process multiple slides"
    echo "  3) Process with tissue filtering (dual extraction)"
    echo "  4) Monitor an existing job"
    echo "  5) Cleanup (delete job/pool)"
    echo "  q) Quit"
    echo ""
    read -p "Enter choice: " choice
    
    case $choice in
        1) example_single_slide ;;
        2) example_batch_processing ;;
        3) example_with_filtering ;;
        4) example_monitor_job ;;
        5) example_cleanup ;;
        q|Q) log "Exiting"; exit 0 ;;
        *) log "Invalid choice"; exit 1 ;;
    esac
}

# Run main menu if script is executed directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main
fi
