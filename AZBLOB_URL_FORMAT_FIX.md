# Azure Blob URL Format Fix

## Problem

Model upload was failing with:
```
ERROR: Failed to upload models to Azure Blob: The specified container does not exist.
ErrorCode:ContainerNotFound
```

## Root Cause

**Incorrect URL format** in model staging:
```python
# WRONG - includes account name in URL
s3_model_prefix = f"azblob://{args.storage_account_name}/{args.staging_container}/models/"
# Result: azblob://mskpdmgen2/mussel-staging/models/
```

The URL parser was treating `mskpdmgen2` (account name) as the container name!

```python
parsed = urlparse("azblob://mskpdmgen2/mussel-staging/models/")
container_name = parsed.netloc  # = "mskpdmgen2" ❌ WRONG!
prefix = parsed.path.lstrip('/') # = "mussel-staging/models/"
```

## Solution

**Correct URL format** - container name only:
```python
# CORRECT - container name in URL, account from environment
s3_model_prefix = f"azblob://{args.staging_container}/models/"
# Result: azblob://mussel-staging/models/
```

Now the parser gets the right container:
```python
parsed = urlparse("azblob://mussel-staging/models/")
container_name = parsed.netloc  # = "mussel-staging" ✅ CORRECT!
prefix = parsed.path.lstrip('/') # = "models/"
```

Account name comes from environment variable `AZURE_STORAGE_ACCOUNT`.

## Azure Blob URL Formats

There are **two valid formats** used in this codebase:

### Format 1: Simple (for model_predownload.py)
```
azblob://container/path
```

Example: `azblob://mussel-staging/models/uni2.pth`

**Usage**: Model upload/download in `model_predownload.py`

**Parsing**:
- Container: `netloc` = `mussel-staging`
- Path: `path` = `models/uni2.pth`
- Account: From `AZURE_STORAGE_ACCOUNT` env var

### Format 2: Full (for Azure Batch tasks)
```
azblob://account.blob.core.windows.net/container/path
```

Example: `azblob://mskpdmgen2.blob.core.windows.net/mussel-staging/slides/slide001.svs`

**Usage**: Slide staging, task parameters

**Parsing**:
- Account: Extracted from `netloc` = `mskpdmgen2.blob.core.windows.net`
- Container: From first path segment
- Path: Remaining path segments

## Changes Made

### File: `scripts/azure_batch/submit_batch_jobs.py`

**Line ~2577**:
```diff
  elif args.staging_container and args.storage_account_name:
      # Use staging container for models (recommended)
-     s3_model_prefix = f"azblob://{args.storage_account_name}/{args.staging_container}/models/"
+     s3_model_prefix = f"azblob://{args.staging_container}/models/"
      print(f"[Model Staging] Using staging container: {s3_model_prefix}")
```

## Before & After

### Before (Broken)
```
[Model Staging] Using staging container: azblob://mskpdmgen2/mussel-staging/models/
[Pre-download] Uploading models to Azure Blob: azblob://mskpdmgen2/mussel-staging/models/
  Uploading VIRCHOW2: ... -> azblob://mskpdmgen2/mussel-staging/models/virchow2.pth
ERROR: The specified container does not exist.
ErrorCode:ContainerNotFound
```

**Problem**: Trying to find container named "mskpdmgen2" (the account!) ❌

### After (Fixed)
```
[Model Staging] Using staging container: azblob://mussel-staging/models/
[Pre-download] Uploading models to Azure Blob: azblob://mussel-staging/models/
  Uploading VIRCHOW2: ... -> azblob://mussel-staging/models/virchow2.pth
  ✓ Uploaded VIRCHOW2
```

**Success**: Found container "mussel-staging" correctly ✅

## Why Two Formats?

### Simple Format (`azblob://container/path`)
**Pros**:
- Cleaner, simpler
- Account from environment (secure)
- Works with model_predownload.py

**Cons**:
- Needs `AZURE_STORAGE_ACCOUNT` env var
- Can't embed account in URL

### Full Format (`azblob://account.blob.core.windows.net/container/path`)
**Pros**:
- Self-contained (account in URL)
- Standard Azure Blob URL format
- Works for external references

**Cons**:
- Longer, more complex
- Requires parsing account from URL

## When to Use Which Format

### Use Simple Format (`azblob://container/path`)
- ✅ Model staging/upload
- ✅ Internal references where account is in environment
- ✅ Scripts that set `AZURE_STORAGE_ACCOUNT`

### Use Full Format (`azblob://account.blob.core.windows.net/container/path`)
- ✅ Slide staging for Azure Batch tasks
- ✅ External references
- ✅ When account needs to be embedded in URL

## Testing

### Test URL Parsing
```python
from urllib.parse import urlparse

# Simple format (for models)
url = "azblob://mussel-staging/models/"
parsed = urlparse(url)
print(f"Container: {parsed.netloc}")  # mussel-staging ✅
print(f"Path: {parsed.path.lstrip('/')}")  # models/

# Full format (for slides)
url = "azblob://mskpdmgen2.blob.core.windows.net/mussel-staging/slides/"
parsed = urlparse(url)
print(f"Account.Host: {parsed.netloc}")  # mskpdmgen2.blob.core.windows.net ✅
print(f"Path: {parsed.path}")  # /mussel-staging/slides/
```

### Test Model Upload
```bash
python scripts/azure_batch/submit_batch_jobs.py \
    --env-file secrets.env \
    --staging-container mussel-staging \
    --config config.yaml \
    --csv-manifest test.csv

# Should see:
# [Model Staging] Using staging container: azblob://mussel-staging/models/
# ✓ Uploaded VIRCHOW2
```

## Summary

b�� **Fixed**: Model upload URL format corrected

b�� **Format**: `azblob://container/path` (simple) for models

b�� **Account**: Comes from `AZURE_STORAGE_ACCOUNT` environment variable

b�� **Verified**: URL parsing works correctly

## Related URLs in Codebase

**Simple format (correct for models)**:
- `azblob://mussel-staging/models/` ✅

**Full format (correct for slides)**:
- `azblob://mskpdmgen2.blob.core.windows.net/mussel-staging/slides/` ✅

**Incorrect (was in code)**:
- `azblob://mskpdmgen2/mussel-staging/models/` ❌ (account/container confusion)
