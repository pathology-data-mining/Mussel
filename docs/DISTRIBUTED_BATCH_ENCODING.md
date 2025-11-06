# Distributed Batch Encoding Optimization

## Overview

The distributed batch processing scripts (HTCondor, SLURM, Azure Batch) now support an optimization for slide-level model aggregation workloads. When processing multiple slides with slide encoders like `GIGAPATH_SLIDE` or `TITAN_SLIDE`, slides can be grouped into batches to significantly reduce model loading overhead.

## Problem

In the original implementation, each slide was processed as an independent task:
- **100 slides** = **100 tasks** = **100× model loads**

When using slide-level model aggregation (e.g., GIGAPATH_SLIDE), each task:
1. Loads the slide encoder model (~2s overhead)
2. Processes one slide
3. Saves results

For 100 slides, this means 100× model loading overhead (~200s total overhead).

## Solution

With batch encoding optimization, slides are grouped into batches:
- **100 slides** with **batch size 8** = **13 tasks** = **13× model loads**

Each batch task:
1. Loads the slide encoder model once (~2s overhead)
2. Processes 8 slides using `aggregate_slide_features_batch()`
3. Saves results for all slides

For 100 slides, this means only 13× model loading overhead (~26s total overhead).

**Result**: ~8x reduction in model loading overhead!

## Usage

### HTCondor

```bash
python -m scripts.condor.submit_condor_jobs \
    --csv-manifest slides.csv \
    --output-s3-prefix s3://bucket/results \
    --aggregation-method model \
    --slide-model-type GIGAPATH_SLIDE \
    --distributed-slide-batch-size 8 \
    --slide-batch-size 8 \
    --submit
```

### SLURM

```bash
python -m scripts.slurm.submit_slurm_jobs \
    --csv-manifest slides.csv \
    --output-dir /scratch/results \
    --aggregation-method model \
    --slide-model-type GIGAPATH_SLIDE \
    --distributed-slide-batch-size 8 \
    --slide-batch-size 8 \
    --submit
```

### Azure Batch

```bash
python -m scripts.azure_batch.submit_batch_jobs \
    --batch-account-name myaccount \
    --batch-account-key mykey \
    --batch-account-url https://myaccount.batch.azure.com \
    --pool-id mypool \
    --job-id myjob \
    --csv-manifest slides.csv \
    --output-s3-prefix s3://bucket/results \
    --distributed-slide-batch-size 8 \
    --create-job \
    --create-pool
```

## Parameters

### `--distributed-slide-batch-size` (default: 1)

Number of slides to group per distributed task for batch encoding optimization.

- **When to use**: Set to 8-16 when using slide-level model aggregation (GIGAPATH_SLIDE, TITAN_SLIDE)
- **When NOT to use**: Leave at 1 (default) for identity aggregation or when not using slide encoders
- **Trade-offs**:
  - **Higher values**: Fewer tasks, less overhead, but less parallelism across compute nodes
  - **Lower values**: More tasks, more overhead, but better parallelism

**Recommended values**:
- GIGAPATH_SLIDE: 8-16
- TITAN_SLIDE: 8-16
- Identity/Mean/Max aggregation: 1 (no benefit)

### `--slide-batch-size` (default: 8)

Number of slides to process in a single batch during slide-level aggregation (used by `aggregate_slide_features_batch()`).

- This parameter controls batching **within** a single task
- Independent of `distributed-slide-batch-size`
- Recommended: 8 for most slide encoders

## How It Works

### Original Workflow (distributed-slide-batch-size=1)

```
CSV Manifest (100 slides)
    ↓
Submit 100 tasks (1 slide each)
    ↓
Task 1: Load model → Process slide 1 → Save
Task 2: Load model → Process slide 2 → Save
...
Task 100: Load model → Process slide 100 → Save
```

### Optimized Workflow (distributed-slide-batch-size=8)

```
CSV Manifest (100 slides)
    ↓
Group into 13 batches (8 slides each, last batch has 4)
    ↓
Submit 13 batch tasks
    ↓
Batch Task 1: Load model → Process slides 1-8 → Save all
Batch Task 2: Load model → Process slides 9-16 → Save all
...
Batch Task 13: Load model → Process slides 97-100 → Save all
```

### Under the Hood

Each batch task receives:
- `SLIDE_PATHS`: Comma-separated list of slide paths
- `SLIDE_IDS`: Comma-separated list of slide IDs
- `OUTPUT_DIR`: Directory for batch outputs

The task script (`run_tessellate_extract_features.sh`) detects batch mode and calls:

```bash
tessellate_extract_features \
    slide_paths=[slide1.svs,slide2.svs,...] \
    slide_ids=[slide1,slide2,...] \
    output_dir=/output \
    aggregation_method=model \
    slide_model_type=GIGAPATH_SLIDE \
    slide_batch_size=8
```

## Automatic Optimization

Batch encoding is **automatically enabled** when:
1. `distributed-slide-batch-size > 1`
2. `aggregation_method=model`
3. `slide_model_type` is specified (e.g., GIGAPATH_SLIDE, TITAN_SLIDE)

If any of these conditions is not met, the script falls back to single-slide tasks (original behavior).

## Performance Benefits

### Model Loading Overhead Reduction

For 100 slides with `distributed-slide-batch-size=8`:
- **Original**: 100 tasks × 2s = 200s overhead
- **Optimized**: 13 tasks × 2s = 26s overhead
- **Savings**: 174s (~87% reduction)

### Total Processing Time

Assuming:
- Model load: 2s
- Slide processing: 0.5s
- 100 slides, batch size 8

**Original**:
- Total: 100 × (2s + 0.5s) = 250s

**Optimized**:
- Total: 13 × 2s + 100 × 0.5s = 26s + 50s = 76s
- **Speedup**: 3.3x faster

Note: Actual speedup depends on:
- Model loading time
- Slide processing time
- Number of slides
- Batch size
- GPU availability

## Backward Compatibility

✅ **Fully backward compatible**

- Default `distributed-slide-batch-size=1` preserves original behavior
- Existing workflows continue to work unchanged
- Single-slide submission still supported
- No breaking changes

## Limitations

1. **Not beneficial for non-model aggregation**: When using `aggregation_method=identity/mean/max`, there's no model loading overhead to reduce.

2. **Reduced parallelism**: Grouping slides into batches means fewer tasks, which may reduce parallelism if you have many compute nodes available.

3. **Azure Files staging**: When using `--stage-to-azure-files` with Azure Batch, incremental staging already provides different optimizations. Batch encoding is not applicable in this mode.

## Examples

### Example 1: Process 100 slides with GIGAPATH_SLIDE on SLURM

```bash
# Create manifest
cat > slides.csv << EOF
slide_id,slide_path
slide001,/data/slide001.svs
slide002,/data/slide002.svs
...
slide100,/data/slide100.svs
EOF

# Submit with batch encoding optimization
python -m scripts.slurm.submit_slurm_jobs \
    --csv-manifest slides.csv \
    --output-dir /scratch/results \
    --aggregation-method model \
    --slide-model-type GIGAPATH_SLIDE \
    --distributed-slide-batch-size 8 \
    --slide-batch-size 8 \
    --partition gpu \
    --gres gpu:1 \
    --mem 32G \
    --time 04:00:00 \
    --submit
```

Output:
```
[Batch Encoding Optimization] Enabled
  Grouping slides into batches of 8
  Slide encoder: GIGAPATH_SLIDE
  This reduces model loading overhead from 100x to 13x

Submitting batch task: batch_1_of_13
  Slides: slide001, slide002, slide003, slide004, slide005, slide006, slide007, slide008
...
Submitted 13 batch tasks
```

### Example 2: Process slides with HTCondor (no optimization needed)

When using identity aggregation, batch encoding provides no benefit:

```bash
python -m scripts.condor.submit_condor_jobs \
    --csv-manifest slides.csv \
    --output-dir ./results \
    --aggregation-method identity \
    --distributed-slide-batch-size 1 \
    --submit
```

The script will submit 100 individual tasks (original behavior).

## Troubleshooting

### Batch task fails

If a batch task fails, check the logs:
- HTCondor: `condor_logs/batch_X_of_Y.err`
- SLURM: `slurm_logs/batch_X_of_Y_<jobid>.err`

Common issues:
- Memory: Increase `--mem` or reduce `distributed-slide-batch-size`
- Time limit: Increase `--time`
- Model download: Ensure models are pre-downloaded or HF_TOKEN is set

### No optimization happening

Check that:
1. `distributed-slide-batch-size > 1`
2. `aggregation-method=model`
3. `slide-model-type` is specified

If any is missing, the script falls back to single-slide tasks.

## See Also

- [Batch Processing README](../README_BATCH_PROCESSING.md) - Overview of batch processing features
- [tessellate-extract-features CLI](../README.md) - Main CLI documentation
- [aggregate_slide_features_batch](../mussel/utils/feature_extract.py) - Core batch processing function
