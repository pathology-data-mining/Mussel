#!/bin/bash
#
# Example: Distributed Batch Processing with Azure Batch
#
# This script demonstrates how to use slide batch feature extraction
# optimization with Azure Batch to process multiple whole-slide images efficiently.
#
# Usage:
#   export AZURE_BATCH_ACCOUNT_NAME="mybatch"
#   export AZURE_BATCH_ACCOUNT_KEY="<key>"
#   export AZURE_BATCH_ACCOUNT_URL="https://mybatch.eastus.batch.azure.com"
#   ./distributed_batch_example_azure.sh
#

set -e

echo "================================================"
echo "Distributed Batch Processing Example (Azure Batch)"
echo "================================================"
echo ""

# Check required environment variables
if [ -z "$AZURE_BATCH_ACCOUNT_NAME" ] || [ -z "$AZURE_BATCH_ACCOUNT_KEY" ] || [ -z "$AZURE_BATCH_ACCOUNT_URL" ]; then
    echo "ERROR: Please set Azure Batch credentials:"
    echo "  export AZURE_BATCH_ACCOUNT_NAME=\"mybatch\""
    echo "  export AZURE_BATCH_ACCOUNT_KEY=\"<key>\""
    echo "  export AZURE_BATCH_ACCOUNT_URL=\"https://mybatch.eastus.batch.azure.com\""
    exit 1
fi

# Configuration
MANIFEST_FILE="example_slides_manifest.csv"
OUTPUT_S3_PREFIX="s3://my-bucket/mussel-results"
SLIDE_MODEL="GIGAPATH_SLIDE"
DISTRIBUTED_BATCH_SIZE=8
POOL_ID="mussel-batch-pool"
JOB_ID="mussel-batch-job-$(date +%Y%m%d-%H%M%S)"

# Create example manifest with S3 paths
echo "Creating example manifest with S3 paths..."
cat > "$MANIFEST_FILE" << 'EOF'
slide_id,slide_path
slide_001,s3://my-bucket/slides/slide_001.svs
slide_002,s3://my-bucket/slides/slide_002.svs
slide_003,s3://my-bucket/slides/slide_003.svs
slide_004,s3://my-bucket/slides/slide_004.svs
slide_005,s3://my-bucket/slides/slide_005.svs
slide_006,s3://my-bucket/slides/slide_006.svs
slide_007,s3://my-bucket/slides/slide_007.svs
slide_008,s3://my-bucket/slides/slide_008.svs
slide_009,s3://my-bucket/slides/slide_009.svs
slide_010,s3://my-bucket/slides/slide_010.svs
slide_011,s3://my-bucket/slides/slide_011.svs
slide_012,s3://my-bucket/slides/slide_012.svs
slide_013,s3://my-bucket/slides/slide_013.svs
slide_014,s3://my-bucket/slides/slide_014.svs
slide_015,s3://my-bucket/slides/slide_015.svs
slide_016,s3://my-bucket/slides/slide_016.svs
EOF

echo "Manifest created: $MANIFEST_FILE"
echo "  Slides: 16"
echo "  Distributed batch size: $DISTRIBUTED_BATCH_SIZE"
echo "  Expected Azure Batch tasks: 2 (16 slides / 8 per batch)"
echo ""

# Submit to Azure Batch with batch processing
echo "Submitting to Azure Batch with slide batch feature extraction..."
echo ""

python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name "$AZURE_BATCH_ACCOUNT_NAME" \
  --batch-account-key "$AZURE_BATCH_ACCOUNT_KEY" \
  --batch-account-url "$AZURE_BATCH_ACCOUNT_URL" \
  --pool-id "$POOL_ID" \
  --create-pool \
  --vm-size Standard_NC6s_v3 \
  --node-count 2 \
  --job-id "$JOB_ID" \
  --create-job \
  --csv-manifest "$MANIFEST_FILE" \
  --output-s3-prefix "$OUTPUT_S3_PREFIX" \
  --aggregation-method model \
  --slide-model-type "$SLIDE_MODEL" \
  --distributed-slide-batch-size "$DISTRIBUTED_BATCH_SIZE" \
  --aws-access-key-id "$AWS_ACCESS_KEY_ID" \
  --aws-secret-access-key "$AWS_SECRET_ACCESS_KEY" \
  --max-retry-count 3 \
  --monitor

echo ""
echo "================================================"
echo "Submission Complete!"
echo "================================================"
echo ""
echo "What happened:"
echo "  1. Azure Batch pool created with 2 nodes"
echo "  2. 16 slides were grouped into 2 batches of 8 slides each"
echo "  3. Each Azure Batch task will:"
echo "     - Download 8 slides from S3"
echo "     - Load the slide encoder model (${SLIDE_MODEL}) ONCE"
echo "     - Process all 8 slides in the batch"
echo "     - Upload results to S3"
echo ""
echo "Benefits:"
echo "  - Model loaded 2 times instead of 16 times (8x reduction)"
echo "  - Faster processing due to batch optimization"
echo "  - Better GPU utilization"
echo "  - Automatic retry on failures"
echo ""
echo "Monitor progress:"
echo "  - Azure Portal: https://portal.azure.com"
echo "  - Or rerun with --monitor flag"
echo ""
echo "Results will be in S3:"
echo "  ${OUTPUT_S3_PREFIX}/GIGAPATH_SLIDE/h5/slide_*.h5"
echo "  ${OUTPUT_S3_PREFIX}/GIGAPATH_SLIDE/pt/slide_*.pt"
echo ""
echo "Cleanup (when done):"
echo "  python scripts/azure_batch/submit_batch_jobs.py \\"
echo "    --batch-account-name \"$AZURE_BATCH_ACCOUNT_NAME\" \\"
echo "    --batch-account-key \"$AZURE_BATCH_ACCOUNT_KEY\" \\"
echo "    --batch-account-url \"$AZURE_BATCH_ACCOUNT_URL\" \\"
echo "    --job-id \"$JOB_ID\" \\"
echo "    --delete-job \\"
echo "    --pool-id \"$POOL_ID\" \\"
echo "    --delete-pool"
echo ""
