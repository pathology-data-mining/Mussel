# DataLoader Performance Optimization Summary

## Issue
When `num_workers=12` was set in the Azure Batch configuration, PyTorch DataLoader was experiencing performance issues due to:
1. Worker process spawning overhead - creating/destroying 12 processes for each batch
2. No prefetching optimization
3. Excessive resource consumption from repeatedly spawning workers

## Solution
Added two key optimizations to all DataLoader instances:

### 1. `persistent_workers=True`
- Keeps worker processes alive between batches
- Eliminates repeated worker spawn/shutdown overhead
- Only applied when `num_workers > 0`
- Significantly reduces CPU time spent on process management

### 2. `prefetch_factor=2`
- Each worker prefetches 2 batches ahead of time
- Keeps the GPU fed with data, reducing idle time
- Default PyTorch value is 2, but we make it explicit
- Only applied when `num_workers > 0`

## Changes Made
Modified DataLoader instantiation in:
- `mussel/utils/feature_extract.py` (6 locations)
- `mussel/cli/cache_tiles.py` (1 location)

All DataLoaders now include:
```python
loader = DataLoader(
    ...,
    persistent_workers=num_workers > 0,
    prefetch_factor=2 if num_workers > 0 else None,
)
```

## Expected Impact
- Reduced worker process overhead with `num_workers=12`
- Better GPU utilization through prefetching
- Faster overall processing times on A100 nodes
- More stable memory usage patterns

## Configuration
Current Azure Batch configuration:
- `num_workers: 12` (takes advantage of 8 CPUs available)
- `batch_size: 64` (optimized for A100 GPU memory)
- Worker processes are now reused across all tiles/slides

## Commit
- Commit: 8fb570a
- Branch: cdsieng-532
- Pushed to remote: Yes
