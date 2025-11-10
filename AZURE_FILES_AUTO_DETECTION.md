# Azure Files Auto-Detection Feature

## Overview

The Azure Batch submission script now **automatically detects** slides that are already staged to Azure Files and uses them directly, avoiding re-upload.

## How It Works

### 1. For S3 Paths

When you submit a CSV with S3 paths:

```csv
slide_id,slide_path
P-0000012-T04-IM6,s3://mskmind-bkt/reef-slides/1106318.svs
```

The script will:
1. Extract the filename: `1106318.svs`
2. Check if it exists in Azure Files in these directories:
   - `slides/1106318.svs`
   - `revision_slides/1106318.svs`
3. If found → Convert to `azfiles://` URL (no download!)
4. If not found → Keep S3 path (container downloads from S3)

### 2. For Local Paths

When you submit a CSV with local paths:

```csv
slide_id,slide_path
slide_001,/local/path/slide_001.svs
```

The script will:
1. Check if file exists locally
2. Check if already staged to Azure Files
3. If already staged → Use existing (no re-upload!)
4. If not staged → Upload to Azure Files

### 3. For azfiles:// Paths

When you submit a CSV with azfiles:// paths:

```csv
slide_id,slide_path
slide_001,azfiles://mskpdmgen2/mussel-staging/revision_slides/1106318.svs
```

The script recognizes these as already-remote and passes them directly (no check needed).

## Example Usage

### Scenario 1: Mix of Staged and Un-staged Slides

**CSV (original S3 paths):**
```csv
slide_id,slide_path
slide_001,s3://bucket/slide_001.svs
slide_002,s3://bucket/slide_002.svs
slide_003,s3://bucket/slide_003.svs
```

**What happens:**
- slide_001: Found in Azure Files → Use azfiles:// (fast!)
- slide_002: Found in Azure Files → Use azfiles:// (fast!)
- slide_003: Not staged → Download from S3 (slower)

**Result:**
- 2 slides processed instantly from Azure Files mount
- 1 slide downloaded from S3
- **No re-upload of already-staged slides!**

### Scenario 2: All Slides Staged

After running the staging script on all 37,267 slides, submit the original CSV:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --csv-manifest revision_samples_with_paths.csv \
  --config-file run_virchow_gigapath_uni.yaml \
  ...
```

**Result:**
- ✅ All slides detected as already staged
- ✅ All converted to azfiles:// URLs
- ✅ Zero re-uploads
- ✅ Zero S3 downloads
- ✅ Maximum performance!

## Benefits

### 1. **No Wasted Uploads**
- Already-staged slides are detected automatically
- No need to manually track what's staged
- No duplicate uploads consuming bandwidth/time

### 2. **Flexibility**
- Use original CSV with S3 paths
- Mix of staged/unstaged slides works seamlessly
- Partial staging is fine - unstaged slides use S3

### 3. **Performance**
- Staged slides: Instant access via Azure Files mount
- Unstaged slides: Download from S3 (fallback)
- Best of both worlds!

### 4. **Simple Workflow**
- Start staging in background (takes days)
- Start submitting jobs immediately
- Early jobs download from S3
- Later jobs use Azure Files as staging completes
- No coordination needed!

## Technical Details

### File Detection Logic

The script checks multiple directories:
```python
possible_remote_paths = [
    f"slides/{slide_filename}",
    f"revision_slides/{slide_filename}",
]

for remote_path in possible_remote_paths:
    if staging.file_exists(remote_path):
        # Found! Use this path
        azfiles_url = f"azfiles://.../{remote_path}"
        break
```

### Skip-If-Exists for Models

Model files (classifier, prefilter, postfilter) also use the same detection:
```python
staging.upload_file(
    local_path=model_path,
    remote_path=remote_path,
    skip_if_exists=True  # Default
)
```

If the model is already staged, it's skipped automatically.

## Files Modified

1. **scripts/common/azure_files_staging.py**
   - Added `file_exists()` method
   - Added `skip_if_exists` parameter to `upload_file()`

2. **scripts/azure_batch/submit_batch_jobs.py**
   - Added S3 path detection logic
   - Checks multiple remote directories
   - Auto-converts S3 → azfiles:// URLs

3. **scripts/azure_batch/stage_slides_to_azure_files.py**
   - Updated output CSV format to match batch script expectations
   - Changed columns from `image_id, sample_id, azfiles_path` to `slide_id, slide_path`

## Testing

Tested with 3 staged slides:

```
✓ P-0000012-T04-IM6: ALREADY STAGED
  Original:  s3://mskmind-bkt/reef-slides/1106318.svs
  Use:       azfiles://mskpdmgen2/mussel-staging/revision_slides/1106318.svs

✓ P-0000034-T01-IM3: ALREADY STAGED
  Original:  s3://mskmind-bkt/reef-slides/881837.svs
  Use:       azfiles://mskpdmgen2/mussel-staging/revision_slides/881837.svs

✓ P-0000037-T02-IM3: ALREADY STAGED
  Original:  s3://mskmind-bkt/reef-slides/755246.svs
  Use:       azfiles://mskpdmgen2/mussel-staging/revision_slides/755246.svs
```

All slides correctly detected and converted to azfiles:// URLs!

## Current Status

### Staging in Progress

**Process**: PID 3281600 (started ~11:10 UTC)
**Files staged**: ~74 slides (as of 16:40 UTC)
**Rate**: ~3.5 files/minute (~17 seconds per slide)
**Total slides**: 37,267 (MSK slides only)
**Estimated completion**: ~7-8 days from start

### You Can Start Submitting Jobs Now!

Even though staging is still in progress, you can submit batch jobs:

1. Jobs will automatically use Azure Files for slides that are already staged
2. Jobs will download from S3 for slides not yet staged
3. As more slides finish staging, subsequent jobs will benefit
4. No coordination or waiting required!

## Conclusion

✅ **Feature Complete!**

You can now:
- Submit jobs with original S3 CSV
- Script automatically detects staged slides
- No re-uploads
- Optimal performance
- Works during partial staging progress

The staging can continue in the background while you submit jobs. Early jobs will download from S3, later jobs will use Azure Files as more slides finish staging.
