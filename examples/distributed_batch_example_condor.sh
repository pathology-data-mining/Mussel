#!/bin/bash
#
# Example: Distributed Batch Processing with HTCondor
#
# This script demonstrates how to use slide batch feature extraction
# optimization with HTCondor to process multiple whole-slide images efficiently.
#
# Usage:
#   ./distributed_batch_example_condor.sh
#

set -e

echo "================================================"
echo "Distributed Batch Processing Example (HTCondor)"
echo "================================================"
echo ""

# Configuration
MANIFEST_FILE="example_slides_manifest.csv"
OUTPUT_DIR="/scratch/mussel_results"
SLIDE_MODEL="TITAN_SLIDE"
DISTRIBUTED_BATCH_SIZE=8

# Create example manifest
echo "Creating example manifest..."
cat > "$MANIFEST_FILE" << 'EOF'
slide_id,slide_path
slide_001,/data/slides/slide_001.svs
slide_002,/data/slides/slide_002.svs
slide_003,/data/slides/slide_003.svs
slide_004,/data/slides/slide_004.svs
slide_005,/data/slides/slide_005.svs
slide_006,/data/slides/slide_006.svs
slide_007,/data/slides/slide_007.svs
slide_008,/data/slides/slide_008.svs
EOF

echo "Manifest created: $MANIFEST_FILE"
echo "  Slides: 8"
echo "  Distributed batch size: $DISTRIBUTED_BATCH_SIZE"
echo "  Expected HTCondor tasks: 1 (all slides in one batch)"
echo ""

# Submit to HTCondor with batch processing
echo "Submitting to HTCondor with slide batch feature extraction..."
echo ""

python scripts/condor/submit_condor_jobs.py \
  --csv-manifest "$MANIFEST_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --aggregation-method model \
  --slide-model-type "$SLIDE_MODEL" \
  --distributed-slide-batch-size "$DISTRIBUTED_BATCH_SIZE" \
  --prefilter-model-type RESNET50 \
  --request-cpus 8 \
  --request-memory 64GB \
  --request-gpus 1 \
  --submit

echo ""
echo "================================================"
echo "Submission Complete!"
echo "================================================"
echo ""
echo "What happened:"
echo "  1. All 8 slides were grouped into 1 batch"
echo "  2. The HTCondor task will:"
echo "     - Load the slide encoder model (${SLIDE_MODEL}) ONCE"
echo "     - Process all 8 slides together"
echo "     - Save results for each slide individually"
echo ""
echo "Benefits:"
echo "  - Model loaded 1 time instead of 8 times"
echo "  - Faster processing due to batch optimization"
echo "  - Better GPU utilization"
echo ""
echo "Monitor progress with:"
echo "  condor_q"
echo ""
echo "Check logs in:"
echo "  condor_logs/batch_1_of_1.out"
echo ""
