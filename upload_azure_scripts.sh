#!/bin/bash
#
# Upload Azure Batch scripts to blob storage
# This script ensures scripts are uploaded to the CORRECT path that submit_batch_jobs.py expects
#
# The path MUST be: scripts/azure_batch/run_tessellate_extract_features.sh
# NOT: scripts/run_tessellate_extract_features.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load secrets
if [ -f "$SCRIPT_DIR/secrets.env" ]; then
    source "$SCRIPT_DIR/secrets.env"
else
    echo "Error: secrets.env not found"
    exit 1
fi

# Verify AZURE_STORAGE_KEY is set
if [ -z "$AZURE_STORAGE_KEY" ]; then
    echo "Error: AZURE_STORAGE_KEY not set in secrets.env"
    exit 1
fi

echo "============================================"
echo "Uploading Azure Batch Scripts to Blob Storage"
echo "============================================"
echo ""
echo "Storage Account: mskpdmgen2"
echo "Container: mussel-staging"
echo "Blob Path: scripts/azure_batch/"
echo ""
echo "IMPORTANT: Scripts MUST be at scripts/azure_batch/ path"
echo "  This is hardcoded in submit_batch_jobs.py line 1310"
echo ""

# Upload run_tessellate_extract_features.sh
echo "Uploading run_tessellate_extract_features.sh..."
az storage blob upload \
    --account-name mskpdmgen2 \
    --account-key "$AZURE_STORAGE_KEY" \
    --container-name mussel-staging \
    --name "scripts/azure_batch/run_tessellate_extract_features.sh" \
    --file "$SCRIPT_DIR/scripts/azure_batch/run_tessellate_extract_features.sh" \
    --overwrite \
    --no-progress

echo "✓ Uploaded run_tessellate_extract_features.sh"
echo ""

# Upload persistent_model_cache.sh
echo "Uploading persistent_model_cache.sh..."
az storage blob upload \
    --account-name mskpdmgen2 \
    --account-key "$AZURE_STORAGE_KEY" \
    --container-name mussel-staging \
    --name "scripts/azure_batch/persistent_model_cache.sh" \
    --file "$SCRIPT_DIR/scripts/azure_batch/persistent_model_cache.sh" \
    --overwrite \
    --no-progress

echo "✓ Uploaded persistent_model_cache.sh"
echo ""

echo "============================================"
echo "Upload Complete!"
echo "============================================"
echo ""
echo "Scripts are now available at:"
echo "  https://mskpdmgen2.blob.core.windows.net/mussel-staging/scripts/azure_batch/run_tessellate_extract_features.sh"
echo "  https://mskpdmgen2.blob.core.windows.net/mussel-staging/scripts/azure_batch/persistent_model_cache.sh"
echo ""
echo "Your YAML config should have:"
echo "  script_blob_url: \"https://mskpdmgen2.blob.core.windows.net/mussel-staging/scripts/\""
echo ""
