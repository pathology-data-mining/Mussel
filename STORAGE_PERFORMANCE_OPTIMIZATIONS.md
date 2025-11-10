# Azure Batch Storage Performance Optimizations

## Summary

Implemented comprehensive performance optimizations to address file storage bottlenecks in Azure Batch A100 processing.

## Changes Made

### 1. Increased Worker and Batch Configuration
**File**: `azure_run_no_filter_optimus.yaml`
- **batch_size**: Increased from 64 to 128
- **num_workers**: Increased from 4 to 8
- **Rationale**: A100 GPUs have significantly more compute power than V100s. Doubling workers and batch size improves GPU utilization and reduces I/O wait times.

### 2. Optimized DataLoader Prefetching
**File**: `mussel/utils/feature_extract.py`
- **prefetch_factor**: Increased from 2 to 4 across all DataLoader instances
- **Rationale**: Higher prefetch factor allows more batches to be prepared in advance, reducing GPU idle time when waiting for data

### 3. Use Fast Local SSD for Temporary Storage
**File**: `scripts/azure_batch/run_tessellate_extract_features.sh`
- Changed work directory from `/tmp` (container storage) to `${TMPDIR:-/hosttmp}/mussel_work_$$`
- **Rationale**: 
  - The Docker container mounts host `/tmp` to `/hosttmp` 
  - Host `/tmp` on Azure A100 VMs is backed by fast local SSD (64GB capacity)
  - Container `/tmp` is slower overlayfs storage
  - All slide staging and intermediate files now use fast local SSD

### 4. Environment Variables Set for Cache
**File**: `scripts/azure_batch/submit_batch_jobs.py` (line 755)
- Already configured to use `/hosttmp` for caching:
  - `TORCH_HOME=/hosttmp/torch_cache`
  - `HF_HOME=/hosttmp/hf_cache`
  - `TMPDIR=/hosttmp`

## Performance Impact

### Expected Improvements:
1. **2x faster data loading**: More workers and higher prefetch factor
2. **Faster I/O operations**: Using local SSD vs networked/overlayfs storage
3. **Better GPU utilization**: Larger batch sizes better match A100 memory (40GB)
4. **Reduced bottlenecks**: GPU spends less time waiting for data

### Storage Layout:
- **Fast Local SSD** (`/hosttmp` → host `/tmp`): 64GB
  - Slide staging
  - Work directories
  - Intermediate files
  - Model cache (Torch, HuggingFace)
- **Azure Files** (`/mnt/batch/tasks/fsmounts/azfiles`): ~100MB/s
  - Input slide files (pre-staged)
  - Model files (pre-staged)
  - Final output storage

## Testing

To test these changes, run:
```bash
cd /gpfs/mskmind_ess/limr/repos/Mussel-3
uv run python scripts/azure_batch/submit_batch_jobs.py \
    --csv-manifest test_slides_quick.csv \
    --config azure_run_no_filter_optimus.yaml
```

Monitor performance with:
```bash
# Check job status
python scripts/azure_batch/check_job_status.py

# Get task logs
python scripts/azure_batch/get_task_logs.py
```

## Key Metrics to Monitor

1. **Task duration**: Should decrease significantly
2. **GPU utilization**: Should stay >80% during feature extraction
3. **I/O wait**: Should be minimal with local SSD
4. **Throughput**: Slides processed per hour should increase

## Notes

- A100 VMs have 64GB local temp storage - ensure work files don't exceed this
- With 128 batch size and 8 workers, memory usage will be higher but within A100's 40GB
- Persistent workers mean PyTorch processes stay alive between batches (faster)
