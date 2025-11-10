# Azure Batch Test - Fix Summary

## What Was Fixed

### 1. Docker Image ✅
- **Built**: mskmind/mussel:latest with CUDA 12.1.1 support
- **Pushed**: Successfully pushed to Docker Hub
- Image includes GPU/CUDA support

### 2. Staging Configuration ✅
- **Fixed**: `stage_to_azure_files` wasn't being read from config
- **Fixed**: `AZURE_STORAGE_KEY` environment variable support
- **Fixed**: `cleanup_staged_files` wasn't being read from config
- Slides are now successfully staged to Azure Files

### 3. Output Directory ✅
- **Fixed**: Changed from `/mnt/output` to `/tmp/output` (writable location)
- Output directories are now created successfully

### 4. Entrypoint Permissions ✅
- Updated entrypoint.sh to pre-create output directory

## Current Issue ❌

**Model Path Not Passed to Tasks**

The `prefilter_model_type=CTRANSPATH` is specified but `model_path` is not being passed to the container tasks.

**Root Cause**: Model pre-download caches models locally (`./model_cache/`) on the submission machine, but:
1. These cached files are NOT uploaded to Azure Files or container
2. The model paths are NOT passed as environment variables to tasks
3. CTRANSPATH requires an explicit model_path parameter

**What's Needed**:
Either:
- Model pre-download should upload cached models to Azure Files and pass paths
- OR models should be pre-installed in the Docker image
- OR model paths need to be passed via environment variables from cached locations

**Test Status**: 9/9 tasks fail with:
```
ValueError: model_path must be provided for TransPath model
```

## Files Modified

1. `scripts/azure_batch/submit_batch_jobs.py`:
   - Added config support for `stage_to_azure_files`
   - Added config support for `cleanup_staged_files`
   - Added `AZURE_STORAGE_KEY` fallback
   - Changed default `output_dir` from `/mnt/output` to `/tmp/output`

2. `entrypoint.sh`:
   - Added creation of `/mnt/output` directory (though we ended up using `/tmp/output`)

3. `azure_test.yaml`:
   - Removed `prefilter_model_path` (local GPFS path) to enable pre-download

## Next Steps

Need to implement model staging/path passing for pre-downloaded models so tasks can access them.
