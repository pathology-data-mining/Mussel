# Distributed Batch Processing with Slide Batch Feature Extraction

This guide explains how to use the slide batch feature extraction optimization in distributed computing environments (SLURM, HTCondor, Azure Batch).

## What is Slide Batch Feature Extraction?

When processing multiple slides with slide-level model aggregation (e.g., GIGAPATH_SLIDE, TITAN_SLIDE), the traditional approach processes slides one at a time:

```
For each slide:
  1. Load slide encoder model (expensive!)
  2. Process slide
  3. Unload model
```

With **slide batch feature extraction**, multiple slides are processed together in a single task:

```
1. Load slide encoder model ONCE
2. Process batch of slides together
3. Unload model
```

### Performance Benefits

Processing 100 slides with GIGAPATH_SLIDE:
- **Sequential**: 250s (100 model loads × 2s + 100 × 0.5s processing) = 2.5s per slide
- **Batch (size=8)**: 32s (1 model load × 2s + optimized processing) = 0.32s per slide
- **Speedup**: **7.8x faster** ⚡

## When to Use

**Use slide batch extraction when:**
- ✅ Using slide-level model aggregation (`aggregation_method=model`)
- ✅ Using a slide encoder model (GIGAPATH_SLIDE, TITAN_SLIDE)
- ✅ Processing multiple slides (2+)
- ✅ Have adequate GPU memory for batch processing

**Don't use when:**
- ❌ Not using slide-level aggregation
- ❌ Processing single slides
- ❌ Memory constraints exist

## How to Use

### SLURM

#### Basic Usage

```bash
# Process 24 slides in batches of 8
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest slides.csv \
  --output-dir /output/results/ \
  --aggregation-method model \
  --slide-model-type GIGAPATH_SLIDE \
  --distributed-slide-batch-size 8 \
  --partition gpu \
  --gres gpu:1 \
  --submit
```

This creates **3 SLURM tasks** (24 slides ÷ 8 per batch), each processing 8 slides together.

#### Manifest Format

```csv
slide_id,slide_path
slide_001,/data/slides/slide_001.svs
slide_002,/data/slides/slide_002.svs
slide_003,s3://bucket/slides/slide_003.svs
...
```

#### Advanced Configuration

```bash
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest slides.csv \
  --output-s3-prefix s3://bucket/results/ \
  --aggregation-method model \
  --slide-model-type GIGAPATH_SLIDE \
  --distributed-slide-batch-size 16 \
  --slide-batch-size 8 \
  --partition gpu \
  --gres gpu:v100:1 \
  --mem 64G \
  --time 04:00:00 \
  --submit
```

**Parameters:**
- `--distributed-slide-batch-size`: Slides per distributed task (default: 1)
- `--slide-batch-size`: Slides per GPU batch during aggregation (default: 8)

### HTCondor

```bash
python scripts/condor/submit_condor_jobs.py \
  --csv-manifest slides.csv \
  --output-dir /output/results/ \
  --aggregation-method model \
  --slide-model-type GIGAPATH_SLIDE \
  --distributed-slide-batch-size 8 \
  --request-gpus 1 \
  --submit
```

### Azure Batch

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatch \
  --batch-account-key $KEY \
  --batch-account-url https://mybatch.batch.azure.com \
  --pool-id mussel-pool \
  --create-pool \
  --job-id mussel-job \
  --create-job \
  --csv-manifest slides.csv \
  --output-s3-prefix s3://bucket/results/ \
  --aggregation-method model \
  --slide-model-type GIGAPATH_SLIDE \
  --distributed-slide-batch-size 8 \
  --monitor
```

**Note**: When using `--stage-to-azure-files`, incremental staging is used and `distributed-slide-batch-size` is not applicable (each slide gets its own task).

## Configuration Parameters

### Distributed Slide Batch Size

Controls how many slides are grouped per distributed task:

```bash
--distributed-slide-batch-size N
```

- **N=1** (default): One task per slide (no batching)
- **N=8** (recommended): 8 slides per task (good balance)
- **N=16**: 16 slides per task (maximum efficiency, requires more memory)

**Recommendation:**
- Small slides (<1GB): Use 16
- Medium slides (1-5GB): Use 8
- Large slides (>5GB): Use 4

### Slide Batch Size

Controls GPU batch size during slide-level aggregation:

```bash
--slide-batch-size N
```

This is used by the CLI internally. Default is 8, which works well for most cases.

## Output Organization

Results are organized by slide ID:

```
output_dir/
├── slide_001.features.h5
├── slide_001.features.pt
├── slide_002.features.h5
├── slide_002.features.pt
...
```

For S3 outputs with model organization:

```
s3://bucket/results/
├── GIGAPATH_SLIDE/
│   ├── h5/
│   │   ├── slide_001_features.h5
│   │   └── slide_002_features.h5
│   ├── pt/
│   │   ├── slide_001_features.pt
│   │   └── slide_002_features.pt
│   └── tile_h5/
│       ├── slide_001_tile_features.h5
│       └── slide_002_tile_features.h5
```

## Monitoring

### Check Task Progress

**SLURM:**
```bash
squeue -u $USER
```

**HTCondor:**
```bash
condor_q
```

**Azure Batch:**
```bash
# In the submit command, add --monitor
# Or check Azure Portal
```

### View Logs

**SLURM:**
```bash
# Check job output
tail -f slurm_logs/batch_1_of_3_*.out

# Check errors
tail -f slurm_logs/batch_1_of_3_*.err
```

**HTCondor:**
```bash
tail -f condor_logs/batch_1_of_3.out
```

**Azure Batch:**
Via Azure Portal or:
```bash
az batch task file download \
  --job-id mussel-job \
  --task-id batch_1_of_3 \
  --file-path stdout.txt \
  --destination ./task-output.txt
```

## Performance Tuning

### Batch Size Selection

Too small:
- More tasks to schedule and manage
- More model loads
- Less efficient

Too large:
- May exceed memory limits
- Longer wait if one task fails
- All slides delayed if task queues

**Sweet spot**: 8-16 slides per batch for most workloads

### Memory Requirements

Estimate memory per batch:
```
Memory = Base_Model_Memory + (Slides_Per_Batch × Per_Slide_Memory)

Example (GIGAPATH_SLIDE):
Memory = 8GB (model) + (8 × 2GB) = 24GB
```

Request adequate memory:
```bash
--mem 32G  # Leave headroom
```

### Recommended Configurations

**Small dataset (<50 slides):**
```bash
--distributed-slide-batch-size 8
--mem 32G
--time 02:00:00
```

**Medium dataset (50-500 slides):**
```bash
--distributed-slide-batch-size 16
--mem 64G
--time 04:00:00
```

**Large dataset (>500 slides):**
```bash
--distributed-slide-batch-size 16
--mem 64G
--time 08:00:00
--max-retry-count 5  # Azure Batch
```

## Troubleshooting

### Out of Memory

**Symptoms:** Task fails with CUDA OOM or system OOM

**Solution:**
1. Reduce `--distributed-slide-batch-size`
2. Increase `--mem`
3. Reduce `--slide-batch-size`

```bash
# Before (failed):
--distributed-slide-batch-size 16

# After (works):
--distributed-slide-batch-size 8
--mem 48G
```

### Slow Processing

**Check:**
1. Are you using slide-level aggregation? (Required for batch optimization)
2. Is GPU being used? (`--use-gpu`)
3. Is slide encoder specified? (`--slide-model-type`)

### Tasks Not Batching

**Check script output:**
```
[Batch Encoding Optimization] Enabled
  Grouping slides into batches of 8
  Slide encoder: GIGAPATH_SLIDE
```

If not shown, verify:
- `--distributed-slide-batch-size > 1`
- `--aggregation-method model`
- `--slide-model-type` is specified

## Examples

### Example 1: Basic Batch Processing

Process 48 slides in batches of 8:

```bash
# Create manifest
cat > slides.csv << EOF
slide_id,slide_path
slide_001,/data/slide_001.svs
slide_002,/data/slide_002.svs
...
slide_048,/data/slide_048.svs
EOF

# Submit to SLURM
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest slides.csv \
  --output-dir /results/ \
  --aggregation-method model \
  --slide-model-type GIGAPATH_SLIDE \
  --distributed-slide-batch-size 8 \
  --partition gpu \
  --gres gpu:1 \
  --submit

# Result: 6 SLURM tasks created (48 / 8)
```

### Example 2: S3 Input/Output

```bash
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest slides.csv \
  --output-s3-prefix s3://my-bucket/results/ \
  --aggregation-method model \
  --slide-model-type TITAN_SLIDE \
  --distributed-slide-batch-size 8 \
  --aws-access-key-id $AWS_ACCESS_KEY_ID \
  --aws-secret-access-key $AWS_SECRET_ACCESS_KEY \
  --partition gpu \
  --gres gpu:1 \
  --submit
```

### Example 3: With Filtering

```bash
python scripts/condor/submit_condor_jobs.py \
  --csv-manifest slides.csv \
  --output-dir /results/ \
  --classifier-pkl tissue_classifier.pkl \
  --classifier-threshold 0.8 \
  --prefilter-model-type CTRANSPATH \
  --aggregation-method model \
  --slide-model-type GIGAPATH_SLIDE \
  --distributed-slide-batch-size 8 \
  --request-gpus 1 \
  --submit
```

### Example 4: Azure Batch with Auto-scaling

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatch \
  --batch-account-key $KEY \
  --batch-account-url https://mybatch.batch.azure.com \
  --pool-id mussel-autoscale \
  --create-pool \
  --vm-size Standard_NC6s_v3 \
  --node-count 5 \
  --job-id mussel-batch-job \
  --create-job \
  --csv-manifest slides.csv \
  --output-s3-prefix s3://bucket/results/ \
  --aggregation-method model \
  --slide-model-type GIGAPATH_SLIDE \
  --distributed-slide-batch-size 16 \
  --max-retry-count 3 \
  --monitor \
  --delete-job \
  --delete-pool
```

## Summary

- **Slide batch feature extraction** optimizes slide encoder loading
- Use `--distributed-slide-batch-size` to group slides per task
- **7-8x speedup** for slide-level aggregation workloads
- Works with SLURM, HTCondor, and Azure Batch
- Recommended batch size: 8-16 slides
- Requires slide-level model aggregation to be enabled

For more details, see:
- Main README: [README.md](../README.md)
- Batch Processing: [README_BATCH_PROCESSING.md](../README_BATCH_PROCESSING.md)
- SLURM: [scripts/slurm/README.md](../scripts/slurm/README.md)
- HTCondor: [scripts/condor/README.md](../scripts/condor/README.md)
- Azure Batch: [scripts/azure_batch/README.md](../scripts/azure_batch/README.md)
