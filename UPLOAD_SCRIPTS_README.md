# Uploading Azure Batch Scripts

## CRITICAL: Path Requirements

The scripts **MUST** be uploaded to the correct blob path or they won't be found by tasks.

### Required Blob Path
```
scripts/azure_batch/run_tessellate_extract_features.sh
scripts/azure_batch/persistent_model_cache.sh
```

### Why This Path?
This path is **hardcoded** in `scripts/azure_batch/submit_batch_jobs.py` at line 1310:
```python
blob_name = "scripts/azure_batch/run_tessellate_extract_features.sh"
```

## How to Upload Scripts

### Method 1: Use the upload script (RECOMMENDED)

```bash
cd /gpfs/mskmind_ess/limr/repos/Mussel-3
./upload_azure_scripts.sh
```

This script:
- Loads credentials from `secrets.env`
- Uploads to the CORRECT path: `scripts/azure_batch/`
- Shows confirmation of upload
- Displays the URLs to verify

### Method 2: Manual upload (if you must)

```bash
source secrets.env

az storage blob upload \
    --account-name mskpdmgen2 \
    --account-key "$AZURE_STORAGE_KEY" \
    --container-name mussel-staging \
    --name "scripts/azure_batch/run_tessellate_extract_features.sh" \
    --file scripts/azure_batch/run_tessellate_extract_features.sh \
    --overwrite

az storage blob upload \
    --account-name mskpdmgen2 \
    --account-key "$AZURE_STORAGE_KEY" \
    --container-name mussel-staging \
    --name "scripts/azure_batch/persistent_model_cache.sh" \
    --file scripts/azure_batch/persistent_model_cache.sh \
    --overwrite
```

## Verification

After uploading, verify the scripts are at the correct location:

```bash
source secrets.env

az storage blob list \
    --account-name mskpdmgen2 \
    --account-key "$AZURE_STORAGE_KEY" \
    --container-name mussel-staging \
    --prefix "scripts/azure_batch/" \
    --output table
```

You should see:
- `scripts/azure_batch/run_tessellate_extract_features.sh`
- `scripts/azure_batch/persistent_model_cache.sh`

## YAML Configuration

Your batch job YAML config should have:

```yaml
script_blob_url: "https://mskpdmgen2.blob.core.windows.net/mussel-staging/scripts/"
```

Note: The URL ends with `scripts/` not `scripts/azure_batch/` because the code appends `azure_batch/` automatically.

## Common Mistakes

❌ **WRONG**: Uploading to `scripts/run_tessellate_extract_features.sh`
✅ **CORRECT**: Upload to `scripts/azure_batch/run_tessellate_extract_features.sh`

❌ **WRONG**: Using `scripts/azure_batch/upload_scripts_to_blob.sh` with default prefix
✅ **CORRECT**: Use `./upload_azure_scripts.sh` from repo root

## When to Upload Scripts

You need to upload scripts after making changes to:
- `scripts/azure_batch/run_tessellate_extract_features.sh`
- `scripts/azure_batch/persistent_model_cache.sh`

Tasks will automatically download and use the latest version from blob storage.

## Troubleshooting

### Scripts not being downloaded by tasks

Check that:
1. Scripts are at `scripts/azure_batch/` path in blob storage (use verification command above)
2. Your YAML has `script_blob_url` set correctly
3. The `submit_batch_jobs.py` has been updated (container prepull is now always enabled)

### Old script still running

The script is downloaded at task start time. If you uploaded a new version:
1. Submit new tasks
2. Existing running tasks will continue using the old version they downloaded
