#!/bin/bash
#
# Example: Batch processing multiple whole-slide images with slide-level aggregation
#
# This script demonstrates how to use tessellate-extract-features (in batch mode) to process
# multiple slides efficiently with GigaPath slide encoder for batch aggregation.
#
# Prerequisites:
# - Mussel installed with gigapath extra: pip install mussel[gigapath]
# - Multiple whole-slide images (.svs, .tiff, etc.)
# - Sufficient GPU memory (adjust slide_batch_size if needed)
#

# Configuration
SLIDES_DIR="/path/to/slides"
OUTPUT_DIR="./batch_features_output"
SLIDE_BATCH_SIZE=8  # Adjust based on GPU memory
NUM_WORKERS=8
BATCH_SIZE=128

# Find all .svs files
SLIDE_PATHS=$(find "$SLIDES_DIR" -name "*.svs" -type f | tr '\n' ',' | sed 's/,$//')

# Run batch processing with GigaPath slide encoder
# Note: tessellate-extract-features automatically operates in batch mode when slide_paths is provided
tessellate_extract_features \
  slide_paths="[$SLIDE_PATHS]" \
  output_dir="$OUTPUT_DIR" \
  aggregation_method=model \
  slide_model_type=GIGAPATH_SLIDE \
  slide_batch_size=$SLIDE_BATCH_SIZE \
  num_workers=$NUM_WORKERS \
  batch_size=$BATCH_SIZE \
  use_gpu=true \
  gpu_device_id=0 \
  seg_config=default \
  keep_intermediate_files=false

echo "Batch processing complete!"
echo "Output directory: $OUTPUT_DIR"
echo "Number of slides processed: $(echo $SLIDE_PATHS | tr ',' '\n' | wc -l)"
