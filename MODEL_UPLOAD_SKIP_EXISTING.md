# Model Upload Optimization: Skip Existing Files

## Problem
Previously, the model directory upload to Azure Blob had an all-or-nothing approach:
- If ANY blob existed with `model_cache/` prefix → Skip entire upload (even if new models were added)
- If NO blobs existed → Upload ALL files (even if most already existed)

This was inefficient when:
- Adding a new model to an existing directory
- Re-running jobs after adding models

## Solution
Modified the upload logic to check each file individually:

### Changes Made

#### 1. `scripts/common/azure_blob_staging.py`
- Added per-file existence check using `blob_client.get_blob_properties()`
- Skip upload if blob already exists (prints "Skipped ... (already exists)")
- Only upload new/missing files
- Changed `overwrite=True` to `overwrite=False` for safety

#### 2. `scripts/azure_batch/submit_batch_jobs.py`
- Removed the all-or-nothing check (lines 2904-2936)
- Now always calls `upload_directory()` which handles skipping internally
- Simplified logic: just upload with per-file checking

## Behavior Now
- First run: Uploads all models
- Subsequent runs: Only uploads new/changed models
- Prints clear messages:
  - "Uploaded {file} ({size})" - for new files
  - "Skipped {file} (already exists, {size})" - for existing files

## Benefits
- **Faster resubmissions**: Only uploads new models
- **Incremental updates**: Can add models without re-uploading everything
- **Bandwidth savings**: Skips large model files that already exist
- **Clear feedback**: Shows which files were uploaded vs skipped

## Example Output
```
[Pre-Pool] Staging model directory to Azure Blob for pool startup...
  Source: /path/to/models
    [UPLOAD DIR] Uploading directory to Azure Blob...
      Skipped OPTIMUS.pth (already exists, 4432.5 KB)
      Skipped UNI2.pth (already exists, 2654.3 KB)
      Uploaded NEW_MODEL.pth (1234.5 KB)
    [DONE] Directory upload complete
  Model cache staging complete
```
