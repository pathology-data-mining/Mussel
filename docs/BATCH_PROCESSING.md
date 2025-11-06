# Batch Processing for Slide-Level Feature Extraction

This document describes the batch processing feature for `tessellate-extract-features` that enables efficient processing of multiple whole-slide images.

## Overview

The new `tessellate-extract-features-batch` command provides optimized batch processing when extracting slide-level features from multiple whole-slide images. This is particularly beneficial when using slide-level aggregation models (e.g., GIGAPATH_SLIDE, TITAN_SLIDE).

## Performance Benefits

Batch processing provides significant performance improvements through:

1. **Model Loading Overhead Reduction**: The slide encoder model is loaded once for all slides, rather than once per slide
2. **Better GPU Utilization**: Multiple slides can be processed in parallel on the GPU
3. **Reduced Memory Transfer Overhead**: Batch tensor operations are more efficient than sequential single-slide operations

### Performance Comparison

**Sequential Processing (per-slide)**:
```
For N slides with model aggregation:
- Load model N times
- Process each slide individually
Total time ≈ N × (model_load_time + inference_time)
```

**Batch Processing**:
```
For N slides with batch_size B:
- Load model once
- Process slides in batches of B
Total time ≈ model_load_time + (N/B) × batch_inference_time
```

Where `batch_inference_time < B × inference_time` due to GPU parallelization.

### Example Timing Estimates

With 100 slides, batch_size=8, using GIGAPATH_SLIDE:
- **Sequential**: 100 × (2s model load + 0.5s inference) = 250s
- **Batch**: 2s model load + (100/8) × 3s batch inference = ~40s
- **Speedup**: ~6.3x

## Usage

### Basic Batch Processing

Process multiple slides without filtering:

```bash
tessellate_extract_features_batch \
  slide_paths="[slide1.svs,slide2.svs,slide3.svs]" \
  output_dir=./output \
  prefilter_model_type=RESNET50 \
  use_gpu=true
```

### With Slide-Level Aggregation (Optimized)

Process multiple slides with slide-level model aggregation:

```bash
tessellate_extract_features_batch \
  slide_paths="[slide1.svs,slide2.svs,slide3.svs]" \
  output_dir=./output \
  aggregation_method=model \
  slide_model_type=GIGAPATH_SLIDE \
  slide_batch_size=8 \
  use_gpu=true
```

The `slide_batch_size` parameter controls how many slides are processed together during slide-level aggregation (default: 8).

### With Filtering

Process multiple slides with tile filtering:

```bash
tessellate_extract_features_batch \
  slide_paths="[slide1.svs,slide2.svs,slide3.svs]" \
  output_dir=./output \
  classifier_pkl=classifier.pkl \
  classifier_threshold=0.75 \
  prefilter_model_type=RESNET50 \
  use_gpu=true
```

### Custom Slide IDs

Specify custom slide identifiers:

```bash
tessellate_extract_features_batch \
  slide_paths="[slide1.svs,slide2.svs]" \
  slide_ids="[patient_001,patient_002]" \
  output_dir=./output \
  prefilter_model_type=RESNET50
```

If not specified, slide IDs are auto-generated from filenames.

## Configuration Parameters

### Core Parameters

- `slide_paths`: List of paths to whole-slide images (required)
- `slide_ids`: Optional list of slide identifiers (defaults to filenames)
- `output_dir`: Directory for output files (required)
- `output_h5_suffix`: Suffix for HDF5 output files (default: "features.h5")
- `output_pt_suffix`: Suffix for PyTorch output files (default: "features.pt")

### Batch Processing Parameters

- `slide_batch_size`: Number of slides to process together during slide-level aggregation (default: 8)
  - Larger values use more GPU memory but may improve throughput
  - Smaller values reduce memory usage
  - Only affects model-based aggregation

### Model Parameters

Same as `tessellate-extract-features`:
- `prefilter_model_type`: Model for initial feature extraction
- `postfilter_model_type`: Model for post-filter extraction (optional)
- `aggregation_method`: "identity", "mean", "max", or "model"
- `slide_model_type`: Slide encoder model (e.g., GIGAPATH_SLIDE, TITAN_SLIDE)
- `slide_model_path`: Path to slide encoder weights (optional)

### Other Parameters

All other parameters from `tessellate-extract-features` are supported:
- Filtering: `classifier_pkl`, `classifier_threshold`
- Segmentation: `seg_config`
- Visualization: `output_mask_suffix`, `output_grid_mask_suffix`, etc.
- Processing: `num_workers`, `batch_size`, `use_gpu`, etc.

## Output Structure

Output files are organized by slide ID in the output directory:

```
output_dir/
├── slide1.features.h5
├── slide1.features.pt
├── slide2.features.h5
├── slide2.features.pt
└── ...
```

Optional outputs (if configured):
```
output_dir/
├── slide1_mask.png
├── slide1_grid.png
├── slide1_thumbnail.png
├── slide1_patches/
│   ├── 0_0.png
│   └── ...
└── ...
```

## When to Use Batch Processing

**Use batch processing when**:
- Processing multiple slides (obviously)
- Using slide-level aggregation with `aggregation_method="model"`
- GPU memory is available
- Throughput is more important than per-slide latency

**Use single-slide processing when**:
- Processing only one slide
- Memory constraints require processing one slide at a time
- Real-time/streaming processing is needed
- Not using slide-level model aggregation

## Implementation Details

The batch processing workflow:

1. **Per-Slide Processing**:
   - Tessellation (parallel)
   - Tile-level feature extraction (parallel)
   - Optional filtering (parallel)

2. **Batch Slide-Level Aggregation**:
   - Load slide encoder model once
   - Process slides in batches during aggregation
   - Save results per slide

For non-model aggregation methods (identity, mean, max), slides are processed sequentially as batch processing provides no benefit.

## Memory Considerations

GPU memory usage depends on:
- `slide_batch_size`: Number of slides processed together
- Number of tiles per slide
- Patch feature dimension
- Slide encoder model size

If encountering out-of-memory errors, reduce `slide_batch_size`.

## Backward Compatibility

The original `tessellate-extract-features` command remains unchanged and continues to process single slides. Use `tessellate-extract-features-batch` for multi-slide workflows.

## Examples

### Process 10 slides with GigaPath

```bash
tessellate_extract_features_batch \
  slide_paths="[slide1.svs,slide2.svs,...,slide10.svs]" \
  output_dir=./gigapath_features \
  aggregation_method=model \
  slide_model_type=GIGAPATH_SLIDE \
  slide_batch_size=4 \
  num_workers=8 \
  batch_size=128 \
  use_gpu=true \
  gpu_device_id=0
```

### Process with filtering and visualization

```bash
tessellate_extract_features_batch \
  slide_paths="[slide1.svs,slide2.svs,slide3.svs]" \
  output_dir=./filtered_features \
  classifier_pkl=tumor_classifier.pkl \
  classifier_threshold=0.8 \
  prefilter_model_type=CTRANSPATH \
  output_grid_mask_suffix="_grid.png" \
  output_thumbnail_suffix="_thumb.png" \
  use_gpu=true
```

## Performance Tuning

To maximize performance:

1. **Tune slide_batch_size**: Start with 8, increase if GPU memory allows
2. **Tune tile batch_size**: Larger values (e.g., 128, 256) for better GPU utilization
3. **Use multi-GPU**: Set `gpu_device_ids=[0,1,2,3]` for multi-GPU processing
4. **Increase num_workers**: More workers for faster data loading (e.g., 8-16)
5. **Keep intermediate files**: Set `keep_intermediate_files=false` to save disk I/O

## Benchmarking

To measure the performance benefit on your hardware:

```python
import time
from mussel.cli import tessellate_extract_features_batch

# Measure batch processing time
start = time.time()
tessellate_extract_features_batch.main(config)
batch_time = time.time() - start

print(f"Batch processing: {batch_time:.2f}s for {n_slides} slides")
print(f"Average: {batch_time/n_slides:.2f}s per slide")
```

Compare with sequential processing using the single-slide command in a loop.
