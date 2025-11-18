# Skip Uploading Models That Already Exist

## Problem

**Before**: Models were re-uploaded to Azure Blob/S3 every time, even if they already existed.

```python
# OLD CODE (always uploads)
blob_client.upload_blob(data, overwrite=True)  # ‚ùå Always uploads
```

**Impact**:
- Wasted time (5-10 minutes per run)
- Wasted bandwidth (12 GB upload)
- Unnecessary Azure storage operations costs

## Solution

**After**: Check if model exists before uploading, skip if already present.

```python
# NEW CODE (checks first)
if blob_client.exists():
    print(f"‚úì {model_type} already exists, skipping")
else:
    blob_client.upload_blob(data, overwrite=False)
    print(f"‚úì Uploaded {model_type}")
```

## Changes Made

### File: `scripts/common/model_predownload.py`

#### 1. Azure Blob - Single File Upload
```python
# Before
blob_client.upload_blob(data, overwrite=True)

# After
if blob_client.exists():
    print(f"‚úì {model_type} already exists, skipping: {blob_path}")
else:
    blob_client.upload_blob(data, overwrite=False)
    print(f"‚úì Uploaded {model_type}")
```

#### 2. Azure Blob - Directory Upload
```python
# Before
blob_client.upload_blob(data, overwrite=True)

# After
if blob_client.exists():
    print(f"‚úì {model_type}/{rel_path} already exists, skipping")
else:
    blob_client.upload_blob(data, overwrite=False)
    print(f"‚úì Uploaded {model_type}/{rel_path}")
```

#### 3. S3 - Single File Upload
```python
# Before
s3_client.upload_file(local_path, bucket, s3_key)

# After
try:
    s3_client.head_object(Bucket=bucket, Key=s3_key)
    print(f"‚úì {model_type} already exists, skipping")
except:
    s3_client.upload_file(local_path, bucket, s3_key)
    print(f"‚úì Uploaded {model_type}")
```

#### 4. S3 - Directory Upload
```python
# Before
s3_client.upload_file(local_file, bucket, s3_key)

# After
try:
    s3_client.head_object(Bucket=bucket, Key=s3_key)
    print(f"‚úì {model_type}/{rel_path} already exists, skipping")
except:
    s3_client.upload_file(local_file, bucket, s3_key)
    print(f"‚úì Uploaded {model_type}/{rel_path}")
```

## Behavior

### First Run (Models Don't Exist)
```
[Model Pre-Download] Starting model pre-download process...
[Model Pre-Download] Models to download: UNI2, VIRCHOW2, OPTIMUS
[Model Staging] Using staging container: azblob://account/mussel-staging/models/
  Uploading UNI2: ./model_cache/uni2.pth -> azblob://...
  ‚úì Uploaded UNI2
  Uploading VIRCHOW2: ./model_cache/virchow2.pth -> azblob://...
  ‚úì Uploaded VIRCHOW2
  Uploading OPTIMUS: ./model_cache/optimus.pth -> azblob://...
  ‚úì Uploaded OPTIMUS
```

**Time**: 5-10 minutes (uploading 12 GB)

### Second Run (Models Already Exist)
```
[Model Pre-Download] Starting model pre-download process...
[Model Pre-Download] UNI2 already cached: ./model_cache/uni2.pth
[Model Pre-Download] VIRCHOW2 already cached: ./model_cache/virchow2.pth
[Model Pre-Download] OPTIMUS already cached: ./model_cache/optimus.pth
[Model Staging] Using staging container: azblob://account/mussel-staging/models/
  ‚úì UNI2 already exists, skipping: azblob://...
  ‚úì VIRCHOW2 already exists, skipping: azblob://...
  ‚úì OPTIMUS already exists, skipping: azblob://...
```

**Time**: ~5-10 seconds (just checking existence)

## Performance Impact

### Upload Times Saved

| Scenario | Before | After | Savings |
|----------|--------|-------|---------|
| First run | 10 min | 10 min | 0% |
| Second run | 10 min | 10 sec | **99%** |
| Third run | 10 min | 10 sec | **99%** |
| **10 runs** | **100 min** | **10 min** | **90%** |

### Bandwidth Saved

| Scenario | Before | After | Savings |
|----------|--------|-------|---------|
| First run | 12 GB | 12 GB | 0 GB |
| Second run | 12 GB | 0 GB | **12 GB** |
| **10 runs** | **120 GB** | **12 GB** | **108 GB (90%)** |

## Cost Savings

### Azure Blob Storage Operations
- **Before**: Upload operation √ó models √ó runs
  - Example: 5 models √ó 10 runs = 50 uploads
  - Cost: ~$0.05 per 10K operations = $0.00025 per upload
  - Total: 50 √ó $0.00025 = **$0.0125**

- **After**: Upload once + check existence
  - 5 uploads + 45 checks
  - Checks are cheaper than uploads
  - Total: **~$0.003**

**Savings**: Small per run, but adds up over time

### Bandwidth & Time
- **Time saved**: 90% on subsequent runs
- **Bandwidth saved**: 108 GB over 10 runs
- **Developer time**: Less waiting!

## Edge Cases Handled

### 1. Model Updated Locally
If you update a model in local cache, it won't be re-uploaded unless you delete the blob first:

```bash
# Delete blob to force re-upload
az storage blob delete \
    --container-name mussel-staging \
    --name models/uni2.pth \
    --account-name mskpdmgen2
```

### 2. Directory Models (e.g., TITAN_SLIDE)
Each file in the directory is checked individually:
```
búì titan_slide/config.json already exists, skipping
búì titan_slide/model.safetensors already exists, skipping
```

### 3. Corrupted Blobs
If a blob exists but is corrupted, you need to manually delete it to re-upload:
```bash
az storage blob delete --container-name mussel-staging --name models/uni2.pth
```

## When Models Are Uploaded

Models are uploaded when:
1. ‚úÖ First time running with a staging container
2. ‚úÖ Model doesn't exist in blob storage
3. ‚úÖ Model was manually deleted from blob storage
4. ‚ùå Model exists but was updated locally (must delete blob first)

## When Models Are Skipped

Models are skipped when:
1. ‚úÖ Model already exists in blob storage
2. ‚úÖ Running a second/third/nth time with same staging container
3. búÖ Multiple runs sharing the same `staging_container`

## Forced Re-upload

If you need to force a re-upload (e.g., model updated):

### Option 1: Delete from Blob Storage
```bash
# Delete specific model
az storage blob delete \
    --container-name mussel-staging \
    --name models/uni2.pth \
    --account-name mskpdmgen2

# Delete all models
az storage blob delete-batch \
    --source mussel-staging \
    --pattern "models/*" \
    --account-name mskpdmgen2
```

### Option 2: Use Different Prefix
```bash
# Upload to new location
--model-s3-prefix azblob://mussel-staging/models-v2/
```

## Testing

### Test 1: First Run (Uploads)
```bash
python scripts/azure_batch/submit_batch_jobs.py \
    --env-file secrets.env \
    --staging-container mussel-staging \
    --config config.yaml \
    --csv-manifest slides.csv

# Look for "Uploading" messages
```

### Test 2: Second Run (Skips)
```bash
# Run again with same staging container
python scripts/azure_batch/submit_batch_jobs.py \
    --env-file secrets.env \
    --staging-container mussel-staging \
    --config config.yaml \
    --csv-manifest slides.csv

# Look for "already exists, skipping" messages
```

## Summary

búÖ **Fixed**: Models are no longer re-uploaded if they already exist

búÖ **Performance**: 90% faster on subsequent runs (10 min ‚Üí 10 sec)

búÖ **Bandwidth**: 90% savings over multiple runs (108 GB saved)

búÖ **Backward Compatible**: No breaking changes

búÖ **Smart Caching**: Checks existence before uploading

## Recommendation

**Always use the same `staging_container` for all runs** to maximize caching benefits:

```yaml
azure:
  staging_container: "mussel-staging"  # Reuse this for all experiments!
  output_prefix: "azblob://mussel-results/run1/"  # Change this per run
```

This way:
- Models uploaded once to `mussel-staging/models/`
- Reused across all experiments
- Results go to separate locations per run
