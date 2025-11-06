#!/bin/bash
#
# Example: Distributed Batch Processing with SLURM
#
# This script demonstrates how to use slide batch feature extraction
# optimization with SLURM to process multiple whole-slide images efficiently.
#
# Usage:
#   ./distributed_batch_example_slurm.sh
#

set -e

echo "================================================"
echo "Distributed Batch Processing Example (SLURM)"
echo "================================================"
echo ""

# Configuration
MANIFEST_FILE="example_slides_manifest.csv"
OUTPUT_DIR="/scratch/mussel_results"
SLIDE_MODEL="GIGAPATH_SLIDE"
DISTRIBUTED_BATCH_SIZE=8
SLIDE_BATCH_SIZE=8

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
slide_009,/data/slides/slide_009.svs
slide_010,/data/slides/slide_010.svs
slide_011,/data/slides/slide_011.svs
slide_012,/data/slides/slide_012.svs
slide_013,/data/slides/slide_013.svs
slide_014,/data/slides/slide_014.svs
slide_015,/data/slides/slide_015.svs
slide_016,/data/slides/slide_016.svs
EOF

echo "Manifest created: $MANIFEST_FILE"
echo "  Slides: 16"
echo "  Distributed batch size: $DISTRIBUTED_BATCH_SIZE"
echo "  Expected SLURM tasks: $((16 / DISTRIBUTED_BATCH_SIZE)) (16 slides / $DISTRIBUTED_BATCH_SIZE per batch)"
echo ""

# Submit to SLURM with batch processing
echo "Submitting to SLURM with slide batch feature extraction..."
echo ""

python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest "$MANIFEST_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --aggregation-method model \
  --slide-model-type "$SLIDE_MODEL" \
  --distributed-slide-batch-size "$DISTRIBUTED_BATCH_SIZE" \
  --slide-batch-size "$SLIDE_BATCH_SIZE" \
  --prefilter-model-type RESNET50 \
  --partition gpu \
  --gres gpu:1 \
  --cpus-per-task 8 \
  --mem 64G \
  --time 04:00:00 \
  --submit

echo ""
echo "================================================"
echo "Submission Complete!"
echo "================================================"
echo ""
echo "What happened:"
echo "  1. 16 slides were grouped into $((16 / DISTRIBUTED_BATCH_SIZE)) batches of $DISTRIBUTED_BATCH_SIZE slides each"
echo "  2. Each SLURM task will:"
echo "     - Load the slide encoder model (${SLIDE_MODEL}) ONCE"
echo "     - Process all $DISTRIBUTED_BATCH_SIZE slides in the batch"
echo "     - Save results for each slide individually"
echo ""
echo "Benefits:"
echo "  - Model loaded 2 times instead of 16 times (8x reduction)"
echo "  - Faster processing due to batch optimization"
echo "  - Better GPU utilization"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  slurm_logs/batch_*_of_*.out"
echo ""
echo "Expected output files in $OUTPUT_DIR:"
echo "  slide_001.features.h5, slide_001.features.pt"
echo "  slide_002.features.h5, slide_002.features.pt"
echo "  ..."
echo "  slide_016.features.h5, slide_016.features.pt"
echo ""
