# SLURM Batch Size Guide

## Memory Issues

If you're getting **Out of Memory (OOM)** errors like:
```
slurmstepd: error: Detected 3 oom_kill events in StepId=XXXX.batch. Some of the step tasks have been OOM Killed.
```

This means your batch size is too large for the available GPU memory.

## Solution: Reduce Batch Size

### Quick Fix

When submitting SLURM jobs, use `--batch-size` parameter:

```bash
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest your_slides.csv \
  --batch-size 256 \  # Reduced from default 1024
  --submit
```

### Recommended Batch Sizes by GPU Memory

| GPU Memory | Recommended Batch Size | Max Batch Size (aggressive) |
|------------|------------------------|----------------------------|
| 24GB (A10, RTX 3090/4090) | 256 | 512 |
| 32GB (V100) | 512 | 768 |
| 40GB (A100-40GB) | 768 | 1024 |
| 80GB (A100-80GB) | 1536 | 2048 |

### Model-Specific Batch Sizes (Best Performance)

For even better memory optimization, you can set batch sizes per model in your config YAML:

```yaml
# slurm_config.yaml
batch_size: 512  # Default fallback

# Model-specific overrides (passed via environment to container)
# Note: Currently needs manual addition to job script
model_batch_sizes:
  OPTIMUS: 1024
  VIRCHOW2: 512
  UNI2: 1024
  GIGAPATH: 512
  CONCH1_5: 512
  TITAN_SLIDE: 32
  GIGAPATH_SLIDE: 32
```

## Factors Affecting Memory Usage

1. **Slide Size**: Larger slides = more tiles = more memory needed
2. **Batch Size**: Higher batch size = more tiles processed at once = more memory
3. **Number of Workers**: More workers = more parallel data loading = more memory
4. **Model Size**: Larger models need more GPU memory

## Memory-Constrained Environments

If running with limited memory (e.g., 16GB GPU):

```bash
python scripts/slurm/submit_slurm_jobs.py \
  --batch-size 128 \
  --num-workers 4 \
  --mem 32GB \  # Ensure enough host RAM too
  --submit
```

## Monitoring

Check memory usage in SLURM logs:
```bash
# Check if OOM occurred
sacct -j <JOB_ID> --format=JobID,State,MaxRSS,ReqMem

# Watch GPU memory during execution
nvidia-smi -l 1
```

## Best Practices

1. **Start conservative**: Begin with batch_size=256 and increase if stable
2. **Test first**: Run a single slide to verify settings before batch submission
3. **Monitor logs**: Check `.err` files for memory warnings
4. **Scale gradually**: Increase batch size by 256 at a time

## Example: Safe Configuration

```bash
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest slides.csv \
  --model-types OPTIMUS,VIRCHOW2,UNI2 \
  --slide-model-types TITAN_SLIDE,GIGAPATH_SLIDE \
  --batch-size 256 \
  --num-workers 8 \
  --mem 64GB \
  --time 04:00:00 \
  --gres gpu:1 \
  --distributed-slide-batch-size 8 \
  --submit
```

This configuration should work reliably on most GPUs with 24GB+ memory.
