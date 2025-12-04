# External Data Azure Batch Processing Guide

## Overview

External data slides from `s3://mskmind-bkt/external-data/` have been staged to Azure Blob Storage and are ready for Azure Batch processing.

## Manifest Details

### Source Manifest
- **File**: `external_data_manifest.csv`
- **Format**: `image_id,sample_id,svs_path`
- **Source**: S3 URLs (`s3://mskmind-bkt/external-data/...`)
- **Count**: 3,115 slides (3,116 lines including header)

### Azure Blob Staged Manifest
- **File**: `external_data_staged_manifest.csv`
- **Format**: `slide_id,slide_path`
- **Storage**: Azure Blob (`azblob://mskpdmgen2/mussel-staging/slides/...`)
- **Count**: 3,115 slides (3,116 lines including header)
- **Status**: âœ… Ready for Azure Batch submission

### URL Format

All slides use the format:
```
azblob://mskpdmgen2/mussel-staging/slides/<slide_id>.<ext>
```

Example:
```
a1951e8f-357f-11eb-9252-001a7dda7111,azblob://mskpdmgen2/mussel-staging/slides/a1951e8f-357f-11eb-9252-001a7dda7111.ndpi
```

This format is compatible with Azure Batch incremental staging (account/container/path).

## Submitting to Azure Batch

### Basic Submission

```bash
#!/bin/bash
source secrets.env

export USE_AZCOPY=true
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

python scripts/azure_batch/submit_batch_jobs.py \
  --csv-manifest external_data_staged_manifest.csv \
  --config run_paper_revisions_prod.yaml \
  --pool-id mussel-external-data-pool \
  --create-pool \
  --create-job \
  --job-id external-data-$TIMESTAMP \
  --env-file secrets.env \
  --stage-to-azure-blob \
  --staging-workers 20 \
  --monitor \
  --save-failed-tasks external-data-failed-$TIMESTAMP.csv
```

### Key Parameters

- `--csv-manifest external_data_staged_manifest.csv`: Use the staged manifest
- `--stage-to-azure-blob`: Enable incremental staging (will detect slides already staged)
- `--staging-workers 20`: Use 20 parallel workers for staging check
- `--slides-per-task 8`: Group 8 slides per task (default, can adjust)
- `--pool-id mussel-external-data-pool`: Custom pool name for external data
- `--monitor`: Monitor job progress

### Expected Task Count

With 3,115 slides and `slides_per_task=8`:
- **Total tasks**: ~390 tasks (3,115 / 8)
- Task names: `batch_1_of_390_OPTIMUS_plus4more`, `batch_2_of_390_OPTIMUS_plus4more`, etc.

### Storage Configuration

From `secrets.env`:
- **Storage Account**: `mskpdmgen2`
- **Container**: `mussel-staging`
- **Slide Path**: `slides/`

## Output Structure

Results will be written to Azure Blob Storage:

```
az://mussel-output/external-data-TIMESTAMP/
b”œâ”€â”€ OPTIMUS/
b”‚   â”œâ”€â”€ h5/
b”‚   â”‚   â”œâ”€â”€ a1951e8f-357f-11eb-9252-001a7dda7111.h5
b”‚   â”‚   â””â”€â”€ ...
b”‚   â””â”€â”€ pt/
b”‚       â”œâ”€â”€ a1951e8f-357f-11eb-9252-001a7dda7111.pt
b”‚       â””â”€â”€ ...
b”œâ”€â”€ VIRCHOW2/
b”‚   â”œâ”€â”€ h5/
b”‚   â””â”€â”€ pt/
b”œâ”€â”€ UNI2/
b”‚   â”œâ”€â”€ h5/
b”‚   â””â”€â”€ pt/
b”œâ”€â”€ GIGAPATH_SLIDE/
b”‚   â”œâ”€â”€ h5/
b”‚   â””â”€â”€ pt/
b””â”€â”€ TITAN_SLIDE/
    â”œâ”€â”€ h5/
    â””â”€â”€ pt/
```

## Monitoring

### Check Job Status

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --job-id external-data-TIMESTAMP \
  --monitor-only
```

### Check Task Progress

```python
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials

# From secrets.env
credentials = SharedKeyCredentials(
    account_name="mskocraBatchAccount",
    account_key="<from secrets.env>"
)
client = BatchServiceClient(credentials, "https://mskocrabatchaccount.eastus2.batch.azure.com")

job_id = "external-data-TIMESTAMP"
tasks = list(client.task.list(job_id))
print(f"Total tasks: {len(tasks)}")
print(f"Completed: {sum(1 for t in tasks if t.state == 'completed')}")
print(f"Active: {sum(1 for t in tasks if t.state == 'active')}")
print(f"Failed: {sum(1 for t in tasks if t.state == 'failed')}")
```

## Downloading Results

After job completion, download results:

```bash
# Download all results
python download_azure_simple.py \
  --container mussel-output \
  --prefix external-data-TIMESTAMP/ \
  --output-dir external_data_results/

# Download specific model
python download_azure_simple.py \
  --container mussel-output \
  --prefix external-data-TIMESTAMP/GIGAPATH_SLIDE/ \
  --output-dir external_data_results/GIGAPATH_SLIDE/
```

## Troubleshooting

### Issue: Tasks named `batch_1_of_1`

**Fixed**: This was caused by incorrect batch numbering in incremental submission. Now uses global batch tracking.

### Issue: GIGAPATH_SLIDE empty output

**Fixed**: GIGAPATH_SLIDE model symlink created in `model_cache/`. See `GIGAPATH_SLIDE_FIX.md`.

### Issue: Some slides fail staging

Check failed tasks CSV:
```bash
cat external-data-failed-$TIMESTAMP.csv
```

Resubmit failed slides:
```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --csv-manifest external-data-failed-$TIMESTAMP.csv \
  --config run_paper_revisions_prod.yaml \
  --pool-id mussel-external-data-pool \
  --job-id external-data-retry-$TIMESTAMP \
  --create-job \
  --stage-to-azure-blob \
  --staging-workers 20 \
  --monitor
```

## Cost Estimation

Based on Azure Batch A100 40GB pricing:
- **GPU**: ~$3.00/hour
- **Storage**: ~$0.02/GB/month
- **3,115 slides** Ã— **5 models** = 15,575 extractions
- **Estimated time**: ~20-30 hours (depending on pool size and slide complexity)
- **Estimated cost**: $60-90 for compute + storage costs

## Files

- `external_data_manifest.csv` - Original S3 manifest
- `external_data_staged_manifest.csv` - Azure Blob manifest (ready for batch)
- `EXTERNAL_DATA_AZURE_BATCH_GUIDE.md` - This guide
- `secrets.env` - Azure credentials (not committed)

## Related Documentation

- `AZURE_INCREMENTAL_STAGING_FIX.md` - Fixes for incremental staging issues
- `GIGAPATH_SLIDE_FIX.md` - Fix for GIGAPATH_SLIDE model loading
- `SLURM_BATCH_SIZE_GUIDE.md` - Batch size recommendations (also applies to Azure)
