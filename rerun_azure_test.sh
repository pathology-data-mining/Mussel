#!/bin/bash
#
# Rerun Azure Batch test with updated Docker image
#

set -e

echo "================================================"
echo "Azure Batch Test Resubmission"
echo "================================================"
echo ""

# Source credentials
if [ -f secrets.env ]; then
    source secrets.env
    echo "✓ Loaded credentials from secrets.env"
else
    echo "ERROR: secrets.env not found"
    exit 1
fi

# Change to scripts/azure_batch directory
cd scripts/azure_batch

# Generate a new job ID
JOB_ID="mussel-test-$(date +%Y%m%d-%H%M%S)"

echo ""
echo "Submitting Azure Batch test..."
echo "  Job ID: $JOB_ID"
echo "  Pool: mussel-pool (will be created)"
echo "  Docker image: mskmind/mussel:latest (freshly pushed)"
echo "  Slides: 9 test slides"
echo "  Output: azfiles://mskpdmgen2/mussel-staging/outputs"
echo ""

# Submit the job
uv run python submit_batch_jobs.py \
  --config ../../azure_test.yaml \
  --csv-manifest ../../test_slides.csv \
  --job-id "$JOB_ID" \
  --monitor

echo ""
echo "================================================"
echo "Submission Complete!"
echo "================================================"
echo ""
echo "The Docker image mskmind/mussel:latest now includes:"
echo "  ✓ CUDA 12.1.1 support"
echo "  ✓ GPU detection and usage"
echo "  ✓ All model paths fixed"
echo "  ✓ Output staging to Azure Files"
echo ""
echo "Results will be in: azfiles://mskpdmgen2/mussel-staging/outputs/"
echo ""
