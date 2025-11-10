# Azure Batch Docker & Testing - Complete Summary

## Successfully Completed b��

### 1. Docker Image
- **Built** mskmind/mussel:latest with CUDA 12.1.1 + cuDNN 8
- **Pushed** to Docker Hub (both :latest and :cuda-12.1.1 tags)
- Fixed entrypoint.sh to create writable directories

### 2. Slide Staging
- **Fixed** `stage_to_azure_files` config support (wasn't being read from YAML)
- **Fixed** `AZURE_STORAGE_KEY` environment variable support  
- **Fixed** `cleanup_staged_files` config support
- Slides successfully stage to Azure Files and are accessible in containers

### 3. Output Directory  
- **Fixed** Changed from `/mnt/output` (not writable) to `/tmp/output`
- Updated default in argument parser and function signatures

### 4. Model Staging Infrastructure
- **Added** import of `stage_models_to_azure_files` function
- **Added** code to stage models to Azure Files before task submission
- **Added** `prefilter_model_path`, `postfilter_model_path`, `slide_model_path` to task submission

## Current Status

### What's Working
- ✅ Docker image builds and pushes
- ✅ Azure Batch pool creation  
- ✅ Slide staging to Azure Files
- ✅ Model pre-download (caches CTRANSPATH and OPTIMUS locally)
- ✅ Model staging to Azure Files
- ✅ Task submission and monitoring
- ✅ Output directory creation
- ✅ `postfilter_model_path` is passed correctly to tasks

### What's NOT Working
- ❌ `prefilter_model_path` is NOT being passed to task environment variables
- Tasks fail with: `ValueError: model_path must be provided for TransPath model`

## Root Cause

The model staging code updates `task_default_params['prefilter_model_path']` with the Azure Files path, but when tasks are submitted via `stage_and_submit_tasks_from_csv`, the `prefilter_model_path` parameter may not be correctly propagated to the `submit_task` call's environment variables.

**Evidence**: 
- Postfilter model path DOES appear in command: `postfilter_model_path=/mnt/batch/tasks/fsmounts/azfiles/models/optimus.pth`  
- Prefilter model path does NOT appear in command or environment

**Likely Issue**:
The model staging conditional logic (lines 1842-1843) might not be matching 'CTRANSPATH' correctly, or the path is being set but not passed through to the environment variable generation in `submit_task()`.

## Files Modified

1. **Dockerfile** - No changes needed (already had CUDA support)
2. **entrypoint.sh** - Added `/mnt/output` directory creation (ended up using `/tmp/output`)
3. **scripts/azure_batch/submit_batch_jobs.py**:
   - Added `stage_models_to_azure_files` import
   - Added config support for `stage_to_azure_files` and `cleanup_staged_files`
   - Added `AZURE_STORAGE_KEY` fallback support
   - Changed default `output_dir` from `/mnt/output` to `/tmp/output` (3 locations)
   - Added model staging call with path updates
   - Added `prefilter_model_path`, `postfilter_model_path`, `slide_model_path` to `stage_and_submit_tasks_from_csv` task submission
4. **azure_test.yaml** - Removed local `prefilter_model_path` to enable pre-download

## Next Step

Debug why `prefilter_model_path` in `task_default_params` is not making it to the task environment variables. The postfilter path works, so the infrastructure is there. Need to verify:
1. Is `prefilter_model_path` actually being set in `task_default_params`? (add logging)
2. Is it being passed to `submit_task()` correctly? (already added to call)
3. Is `submit_task()` creating the environment variable? (line 480-481 should do this)

The debug logging added in the latest change should reveal which model paths are being set.
