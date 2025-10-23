#!/usr/bin/env bash
#
# Example script demonstrating Mussel Docker usage
# This script processes a whole-slide image using the Docker wrapper
#

set -e

echo "=== Mussel Docker Example ==="
echo ""

# Check if mussel-docker exists
if [ ! -f "./mussel-docker" ]; then
    echo "Error: mussel-docker script not found in current directory"
    echo "Make sure you're running this from the Mussel repository root"
    exit 1
fi

# Check if we have a test slide
TEST_SLIDE="tests/testdata/948176.svs"
if [ ! -f "$TEST_SLIDE" ]; then
    echo "Error: Test slide not found at $TEST_SLIDE"
    exit 1
fi

# Create output directory
OUTPUT_DIR="docker-example-output"
mkdir -p "$OUTPUT_DIR"

echo "This example will:"
echo "1. Build the Docker image (if not already built)"
echo "2. Tessellate (tile) a test whole-slide image"
echo "3. Extract features using CLIP model"
echo "4. Create class embeddings for tissue types"
echo "5. Annotate tiles with tissue types"
echo ""
echo "Output will be saved to: $OUTPUT_DIR/"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Step 1: Build image
echo ""
echo "=== Step 1: Building Docker Image ==="
./mussel-docker build

# Step 2: Tessellate
echo ""
echo "=== Step 2: Tessellating Slide ==="
./mussel-docker tessellate \
    slide_path="$TEST_SLIDE" \
    output_h5_path="$OUTPUT_DIR/tiles.h5" \
    seg_config.segment_threshold=0 \
    num_workers=1

# Step 3: Extract features (using CPU if GPU not available)
echo ""
echo "=== Step 3: Extracting Features ==="
if docker run --rm --gpus all ubuntu:22.04 true &> /dev/null; then
    echo "GPU detected, using GPU for feature extraction"
    ./mussel-docker extract_features \
        slide_path="$TEST_SLIDE" \
        patch_h5_path="$OUTPUT_DIR/tiles.h5" \
        model_type=CLIP \
        output_h5_path="$OUTPUT_DIR/features.h5" \
        output_pt_path="$OUTPUT_DIR/features.pt"
else
    echo "No GPU detected, skipping feature extraction (would be very slow on CPU)"
    echo "You can run it manually with:"
    echo "./mussel-docker extract_features slide_path=$TEST_SLIDE patch_h5_path=$OUTPUT_DIR/tiles.h5 model_type=CLIP output_h5_path=$OUTPUT_DIR/features.h5"
fi

# Step 4: Create class embeddings (if we have features)
if [ -f "$OUTPUT_DIR/features.pt" ]; then
    echo ""
    echo "=== Step 4: Creating Class Embeddings ==="
    ./mussel-docker create_class_embeddings \
        classes='["tumor","stroma","necrosis","lymphocytes"]' \
        output_pt_path="$OUTPUT_DIR/class_embeddings.pt"
    
    # Step 5: Annotate
    echo ""
    echo "=== Step 5: Annotating Tiles ==="
    ./mussel-docker annotate \
        features_pt_path="$OUTPUT_DIR/features.pt" \
        class_embedding_pt_path="$OUTPUT_DIR/class_embeddings.pt" \
        classes='["tumor","stroma","necrosis","lymphocytes"]' \
        output_csv_path="$OUTPUT_DIR/annotations.csv"
    
    echo ""
    echo "=== Complete! ==="
    echo "Results saved to: $OUTPUT_DIR/"
    ls -lh "$OUTPUT_DIR/"
else
    echo ""
    echo "=== Partial Complete ==="
    echo "Tessellation complete. Feature extraction was skipped (no GPU available)."
    echo "Results saved to: $OUTPUT_DIR/"
    ls -lh "$OUTPUT_DIR/"
fi

echo ""
echo "You can explore the results or start an interactive shell with:"
echo "./mussel-docker shell"
