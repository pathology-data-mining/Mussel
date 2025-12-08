#!/bin/bash
#
# Upload Azure Batch scripts to Azure Blob Storage
# This allows updating scripts without rebuilding containers
#

set -e

usage() {
    echo "Usage: $0 <storage_account> <container> [blob_prefix]"
    echo ""
    echo "Arguments:"
    echo "  storage_account  Azure Storage account name"
    echo "  container        Blob container name"
    echo "  blob_prefix      Optional blob path prefix (default: scripts/)"
    echo ""
    echo "Environment variables:"
    echo "  AZURE_STORAGE_KEY    Storage account key (required)"
    echo ""
    echo "Example:"
    echo "  export AZURE_STORAGE_KEY='your-key'"
    echo "  $0 myaccount mycontainer scripts/"
    exit 1
}

if [ $# -lt 2 ]; then
    usage
fi

STORAGE_ACCOUNT="$1"
CONTAINER="$2"
BLOB_PREFIX="${3:-scripts/azure_batch/}"

if [ -z "$AZURE_STORAGE_KEY" ]; then
    echo "Error: AZURE_STORAGE_KEY environment variable not set"
    usage
fi

# Ensure blob prefix ends with /
[[ "$BLOB_PREFIX" != */ ]] && BLOB_PREFIX="${BLOB_PREFIX}/"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Uploading Azure Batch scripts to blob storage..."
echo "  Storage account: $STORAGE_ACCOUNT"
echo "  Container: $CONTAINER"
echo "  Blob prefix: $BLOB_PREFIX (scripts will be at ${BLOB_PREFIX}run_tessellate_extract_features.sh)"
echo ""

# List of scripts to upload
SCRIPTS=(
    "run_tessellate_extract_features.sh"
    "persistent_model_cache.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$SCRIPT_DIR/$script" ]; then
        echo "Warning: $script not found, skipping"
        continue
    fi
    
    BLOB_NAME="${BLOB_PREFIX}${script}"
    echo "Uploading $script to $BLOB_NAME..."
    
    az storage blob upload \
        --account-name "$STORAGE_ACCOUNT" \
        --account-key "$AZURE_STORAGE_KEY" \
        --container-name "$CONTAINER" \
        --name "$BLOB_NAME" \
        --file "$SCRIPT_DIR/$script" \
        --overwrite \
        --no-progress
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Uploaded successfully"
    else
        echo "  ✗ Upload failed"
        exit 1
    fi
done

echo ""
echo "All scripts uploaded successfully!"
echo ""
echo "To use these scripts in your batch jobs, add to your config YAML:"
echo ""
echo "script_blob_url: https://${STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER}/${BLOB_PREFIX}"
echo ""
