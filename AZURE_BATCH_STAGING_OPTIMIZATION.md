# Azure Batch Staging Performance Optimization

## Problem
When submitting Azure Batch jobs with already-staged slides, the script was taking too long because it made individual API calls to check if each slide exists in blob storage (`blob_exists()` for each slide).

For example, with 1000 slides, this meant 2000+ API calls (checking both `slides/` and `revision_slides/` prefixes).

## Solution
Implemented batch blob listing to load all blob names once at the start, then use in-memory set lookups (O(1)) instead of API calls for each slide.

### Changes Made

1. **Added `get_blob_set()` method to `AzureBlobStaging` class** (`scripts/common/azure_blob_staging.py`)
   - Returns a set of all blob names in the container for efficient membership testing
   - Uses existing `list_blobs()` method internally

2. **Updated `submit_batch_jobs.py` staging logic** (`scripts/azure_batch/submit_batch_jobs.py`)
   - Load all existing blobs once at the start: `existing_blobs = submitter.azure_blob_staging.get_blob_set()`
   - Use set membership check instead of API calls: `if blob_name in existing_blobs`

### Performance Improvement

**Before:**
- For N slides: ~2N API calls (checking 2 possible locations per slide)
- Each API call takes ~100-500ms
- Total time for 1000 slides: ~3-10 minutes

**After:**
- 1 API call to list all blobs (returns all blobs in container)
- O(1) in-memory lookups for each slide
- Total time for 1000 slides: ~5-30 seconds

**Speedup: 10-100x faster** depending on number of slides and network latency

### Usage
No changes required to existing scripts. The optimization is transparent and works with existing submission commands:

```bash
./run_pr_test_small2.sh
```

The script will now output:
```
[Azure Blob] Loading existing blob list for fast lookup...
[Azure Blob] Found XXXX existing blobs in storage
```

This indicates the batch loading is working.
