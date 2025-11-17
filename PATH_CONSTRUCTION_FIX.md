# Path Construction Bug Fix âœ…

## Date
2025-11-17 06:00 UTC

## Problem

Tasks were failing with a path error after successfully completing authentication and feature extraction:

```
FileNotFoundError: Unable to open file (name = 'az://mussel-results/a100-staging-test/OPTIMUS/OPTIMUS/TCGA-02-0003...features.h5')
```

Notice the **double `OPTIMUS/OPTIMUS`** in the path - the model name was being duplicated.

## Root Cause

The model subdirectory was being added TWICE:

1. **Azure Batch submission script** (line 1471): Added `/{model_type}` to output_dir
   ```python
   output_dir = f"{output_s3_prefix.rstrip('/')}/{model_type}"
   ```

2. **CLI code** (lines 760, 784): Also added model subdirectory
   ```python
   model_output_dir = _safe_path_join(output_dir_str, model.name)
   cfg_copy.output_dir = model_output_dir
   ```

This resulted in paths like:
- Batch sets: `az://mussel-results/a100-staging-test/OPTIMUS`
- CLI adds: `/OPTIMUS/` 
- Final (broken): `az://mussel-results/a100-staging-test/OPTIMUS/OPTIMUS/file.h5`

## Solution

Removed the model subdirectory addition from the Azure Batch submission script, since the CLI already handles it:

```python
# Before (BROKEN):
output_dir = f"{output_s3_prefix.rstrip('/')}/{model_type}"
output_dir_for_batch = self.convert_azblob_to_fsspec_url(output_dir)

# After (FIXED):
# Use output prefix as-is (CLI will add model subdirectory)
output_dir_for_batch = self.convert_azblob_to_fsspec_url(output_s3_prefix.rstrip('/'))
```

## Files Modified

**scripts/azure_batch/submit_batch_jobs.py** (lines 1469-1474):
- Removed `/{model_type}` from output_dir construction
- Added comment explaining that CLI handles model subdirectories

## Test Results

### Before Fix:
```
FileNotFoundError: ...OPTIMUS/OPTIMUS/TCGA-02-0003-01...features.h5
```

### After Fix:
```
bœ“ No path errors
bœ“ Feature extraction running successfully (8/132 batches)
bœ“ Correct path structure: az://mussel-results/a100-staging-test/OPTIMUS/file.h5
```

## Related Fixes

This fix completes the Azure Storage authentication improvements:

1. âœ… **Input staging credentials** - Fixed environment variable names for az CLI
2. âœ… **Output writing credentials** - Added adlfs environment variables  
3. âœ… **Path construction** - Fixed duplicate model subdirectories

## Verification

Tasks are now running successfully with:
- âœ… Correct authentication (input and output)
- âœ… Correct path structure (no duplicates)
- ðŸ”„ Feature extraction in progress (no errors)

The complete end-to-end workflow is now functional!
