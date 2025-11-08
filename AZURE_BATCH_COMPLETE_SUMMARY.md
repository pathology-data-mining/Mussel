# Azure Batch - Complete Implementation Summary

All changes committed to PR #94 (branch: `cdsieng-532`)

## Fixed Issues

### 1. Gen2 VM Image Support ✅
- **Problem**: A100 GPUs require Gen2 hypervisor, original image was Gen1
- **Solution**: 
  - Separated image SKU (`2204`) from node agent SKU ID (`batch.node.ubuntu 22.04`)
  - Image: `microsoft-dsvm/ubuntu-hpc/2204` (Gen2 compatible)
  - Added `--node-agent-sku-id` parameter to submit script

### 2. Container GPU Passthrough ✅
- **Problem**: `--gpus all` Docker flag not supported in Azure Batch
- **Solution**: Removed `--gpus` flag - Azure Batch auto-passes GPUs on GPU VMs

### 3. Batch Processing Support ✅
- **Problem**: Script only handled single slides
- **Solution**: Updated `run_tessellate_extract_features.sh` to:
  - Accept `SLIDE_PATHS` (comma-separated)
  - Support `OUTPUT_H5_PATHS`, `OUTPUT_PT_PATHS`, `SLIDE_IDS`
  - Generate output paths from `OUTPUT_DIR` if not provided
  - Process slides in loop with `process_slide()` function

### 4. Azure Files Staging ✅
- **Problem**: Local slide files not accessible on Azure nodes
- **Solution**: Automatic staging to Azure Files:
  - Stage slides: `mussel-staging/slides/`
  - Stage models: `mussel-staging/models/`
  - Convert to `azfiles://` URLs
  - Resolve to mount path: `/mnt/batch/tasks/fsmounts/azfiles/`

### 5. Model Path Resolution ✅
- **Problem**: Model paths not passed to tasks or resolved
- **Solution**:
  - Stage `classifier_pkl`, `prefilter_model_path`, `postfilter_model_path`
  - Resolve `azfiles://` URLs to local mount paths
  - Add paths to `tessellate_extract_features` command

### 6. Output File Staging ✅
- **Problem**: Results not uploaded back to specified destination
- **Solution**: Support multiple storage backends:
  - **S3**: `s3://bucket/prefix/`
  - **Azure Files**: `azfiles://account/share/path`
  - **Azure Blob**: `https://account.blob.core.windows.net/container/path`
  - **Azure Blob (alt)**: `azblob://account/container/path`
  - Auto-detect storage type and upload accordingly

## Storage Support Matrix

| Feature | S3 | Azure Files | Azure Blob | Local |
|---------|-------|-------------|------------|-------|
| Input slides | ✅ | ✅ | ✅ | ✅ |
| Model files | ✅ | ✅ | ✅ | ✅ |
| Output results | ✅ | ✅ | ✅ | ✅ |
| Auto-staging | ✅ | ✅ | ✅ | N/A |

## Usage Examples

### Example 1: Azure Files for Everything
```bash
uv run python submit_batch_jobs.py \
  --batch-account-name "ocra" \
  --batch-account-key "$AZURE_BATCH_ACCOUNT_KEY" \
  --batch-account-url "https://ocra.eastus2.batch.azure.com" \
  --storage-account-name "mskpdmgen2" \
  --storage-account-key "$AZURE_STORAGE_KEY" \
  --azure-files-share-name "mussel-staging" \
  --config config.yaml \
  --csv-manifest slides.csv \
  --output-dir "azfiles://mskpdmgen2/mussel-staging/outputs" \
  --job-id "mussel-job-001"
```

### Example 2: Azure Blob for Outputs
```bash
uv run python submit_batch_jobs.py \
  ... \
  --output-dir "https://mskpdmgen2.blob.core.windows.net/results/run1" \
  --job-id "mussel-job-002"
```

### Example 3: S3 for Outputs
```bash
uv run python submit_batch_jobs.py \
  ... \
  --output-dir "s3://my-bucket/mussel-results/" \
  --job-id "mussel-job-003"
```

## Configuration

### Config File (`azure_test.yaml`)
```yaml
prefilter_model_path: /gpfs/path/to/ctranspath.pth
classifier_pkl: /gpfs/path/to/classifier.pkl

# Azure Batch settings
azure_batch:
  batch_account_name: "ocra"
  container_image: "mskmind/mussel:latest"
  storage_account_name: "mskpdmgen2"
  azure_files_share_name: "mussel-staging"
  stage_to_azure_files: true
  mount_azure_files: true
  output_dir_azfiles: "azfiles://mskpdmgen2/mussel-staging/outputs"
```

### CSV Manifest (`slides.csv`)
```csv
slide_id,slide_path
1079807,/gpfs/path/to/1079807.svs
1147432,/gpfs/path/to/1147432.svs
```

## Next Steps

1. **Rebuild Docker Image**:
   ```bash
   docker build -t mskmind/mussel:latest .
   docker push mskmind/mussel:latest
   ```

2. **Test End-to-End**:
   ```bash
   source secrets.env
   cd scripts/azure_batch
   uv run python submit_batch_jobs.py [options]
   ```

3. **Monitor Progress**:
   ```python
   from azure.batch import *
   from azure.batch.batch_auth import *
   c = SharedKeyCredentials('account', 'key')
   b = BatchServiceClient(c, 'url')
   
   for task in b.task.list('job-id'):
       print(f"{task.id}: {task.state}")
   ```

4. **Download Results**:
   - Azure Files: Use Azure Portal or `az storage file download`
   - Azure Blob: Use Azure Portal or `az storage blob download`
   - S3: Use `aws s3 cp`

## Files Changed

- `scripts/azure_batch/submit_batch_jobs.py` - Pool creation, task submission, staging
- `scripts/azure_batch/run_tessellate_extract_features.sh` - Batch processing, path resolution, uploads
- `azure_test.yaml` - Test configuration

## PR Details

- **Branch**: `cdsieng-532`
- **PR**: #94
- **Status**: Ready for review and merge
- **Tested**: Pool creation, task submission, staging (end-to-end pending Docker rebuild)
