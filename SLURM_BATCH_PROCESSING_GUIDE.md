# SLURM Batch Processing Guide

## Understanding Batch Processing Modes

The SLURM submission script supports two modes:

### 1. Standard Mode (Job Array)
- **When**: `aggregation_method: identity` (tile features only)
- **Behavior**: Each slide is processed by a separate SLURM array task
- **Use case**: Extracting tile-level features only (e.g., CTRANSPATH, UNI, CONCH)
- **Example**: Your recent test with 9 slides created 9 separate tasks

### 2. Batch Processing Mode (Slide Batching)
- **When**: `aggregation_method: model` + `slide_model_type` specified + `distributed_slide_batch_size > 1`
- **Behavior**: Multiple slides are grouped and processed by a single task
- **Use case**: Slide-level aggregation with models like GIGAPATH_SLIDE or TITAN_SLIDE
- **Benefit**: Reduces slide encoder loading overhead from N×  to ⌈N/batch_size⌉×

## Enabling Batch Processing

To enable batch processing for slide-level aggregation:

### Option 1: Command Line

```bash
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest slides.csv \
  --output-dir outputs/batch_test \
  --aggregation-method model \
  --slide-model-type GIGAPATH_SLIDE \
  --distributed-slide-batch-size 8 \
  --submit
```

### Option 2: Configuration File

```yaml
# config.yaml
prefilter_model_type: CTRANSPATH
aggregation_method: model           # Enable slide-level aggregation
slide_model_type: GIGAPATH_SLIDE   # Specify slide encoder
batch_size: 64
num_workers: 8
use_gpu: true
seg_config:
  group: biopsy
```

Then submit with:

```bash
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest slides.csv \
  --config config.yaml \
  --output-dir outputs/batch_test \
  --distributed-slide-batch-size 8 \  # Group 8 slides per task
  --submit
```

## Batch Size Recommendations

| Slide Encoder | Recommended Batch Size | Notes |
|---------------|------------------------|-------|
| GIGAPATH_SLIDE | 8-16 | Large model, significant loading time |
| TITAN_SLIDE | 8-16 | Large model, significant loading time |
| Identity | N/A | No slide encoder, no benefit from batching |

## Performance Comparison

### Without Batch Processing (distributed_slide_batch_size=1)
```
100 slides × 2 min model loading = 200 minutes loading time
100 slides × 1 min processing = 100 minutes processing time
Total: 300 minutes
```

### With Batch Processing (distributed_slide_batch_size=10)
```
10 batches × 2 min model loading = 20 minutes loading time
100 slides × 1 min processing = 100 minutes processing time
Total: 120 minutes (60% faster!)
```

## Important Notes

1. **Batch processing only applies when**:
   - `aggregation_method` is `"model"` (not `"identity"`, `"mean"`, or `"attention"`)
   - `slide_model_type` is specified (e.g., `GIGAPATH_SLIDE`)
   - `distributed_slide_batch_size` > 1

2. **Your recent SLURM test ran correctly**:
   - Used `aggregation_method: identity` (tile features only)
   - No slide encoder needed
   - Standard job array mode is correct for this configuration
   - Each of 9 slides processed in separate tasks (expected behavior)

3. **Memory considerations**:
   - Larger batch sizes require more memory
   - Each slide in batch needs its tile features in memory during aggregation
   - Monitor memory usage and adjust batch size accordingly

4. **Resource allocation**:
   - Batch tasks may need more memory: use `--mem 64G` or higher
   - Increase time limit for batch tasks: `--time 04:00:00`

## Example: Full Slide-Level Processing

```bash
# Create config with slide-level aggregation
cat > slide_aggregation_config.yaml << 'EOF'
prefilter_model_type: CTRANSPATH
prefilter_model_path: /path/to/ctranspath.pth
aggregation_method: model
slide_model_type: GIGAPATH_SLIDE
batch_size: 64
num_workers: 8
use_gpu: true
seg_config:
  group: biopsy

resources:
  cpus: 8
  memory: 64G
  gpus: 1

slurm:
  time: "04:00:00"
EOF

# Submit with batch processing
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest slides.csv \
  --config slide_aggregation_config.yaml \
  --output-dir outputs/slide_features \
  --distributed-slide-batch-size 10 \
  --submit
```

This will:
1. Group 100 slides into 10 batches of 10 slides each
2. Submit 10 SLURM tasks (instead of 100)
3. Each task loads GIGAPATH_SLIDE once and processes 10 slides
4. Significantly reduces overall processing time

## Verification

Check the submission output:
```
[Batch Encoding Optimization] Enabled
  Grouping slides into batches of 10
  Slide encoder: GIGAPATH_SLIDE
  This reduces model loading overhead from 100x to 10x

Submitting batch task: batch_1_of_10
  Slides: slide001, slide002, ..., slide010
```

If you don't see this message, batch processing is not enabled (which is correct for identity aggregation).
