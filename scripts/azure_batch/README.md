# Azure Batch Scripts for Mussel

This directory contains scripts for running `tessellate-extract-features` on Azure Batch, enabling large-scale processing of whole-slide images in the cloud with support for S3 storage.

## Overview

The Azure Batch integration allows you to:
- Process multiple whole-slide images in parallel
- Scale computation using Azure's cloud infrastructure
- Use GPU-enabled VMs for fast feature extraction
- Manage long-running jobs with automatic retry and monitoring
- Stage slides from S3 and publish results to S3 or local storage
- Process slides from CSV manifests with slide identifiers
- Automatic cleanup of temporary files upon task completion (success or failure)

## Files

- **`submit_batch_jobs.py`**: Python script to submit jobs to Azure Batch
- **`run_tessellate_extract_features.sh`**: Bash script that runs on Azure Batch compute nodes
- **`config_template.json`**: Template for batch job configuration
- **`manifest_template.csv`**: Template for CSV slide manifest
- **`README.md`**: This file

## Prerequisites

### 1. Azure Setup

You need:
- An Azure subscription
- An Azure Batch account
- (Optional) An Azure Storage account for storing slides and results

To create these resources:

```bash
# Create resource group
az group create --name myResourceGroup --location eastus

# Create storage account (optional but recommended)
az storage account create \
  --resource-group myResourceGroup \
  --name mystorageaccount \
  --location eastus \
  --sku Standard_LRS

# Create batch account
az batch account create \
  --name mybatchaccount \
  --resource-group myResourceGroup \
  --location eastus
```

### 2. Python Dependencies

Install the required Python packages:

```bash
pip install azure-batch azure-storage-blob azure-identity
```

### 3. Docker Image

The scripts use the Mussel Docker image. The default image is `mskmind/mussel:latest-torch-gpu`. You can:
- Use the pre-built image from Docker Hub
- Build your own image (see main README)
- Use a custom image by specifying `--container-image`

## Quick Start

### Single Task Submission

Submit a single slide for processing:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatchaccount \
  --batch-account-key <your-batch-key> \
  --batch-account-url https://mybatchaccount.eastus.batch.azure.com \
  --pool-id mussel-pool \
  --create-pool \
  --job-id mussel-job-001 \
  --create-job \
  --task-id slide-001 \
  --slide-path /mnt/data/slide.svs \
  --output-h5-path /mnt/output/slide_features.h5 \
  --output-pt-path /mnt/output/slide_features.pt \
  --monitor
```

### Batch Processing with Configuration File

For processing multiple slides, create a configuration file:

```json
{
  "defaults": {
    "prefilter_model_type": "CTRANSPATH",
    "segment_threshold": 0,
    "patch_size": 256,
    "mpp": 0.5,
    "num_workers": 4,
    "batch_size": 64,
    "use_gpu": true
  },
  "tasks": [
    {
      "task_id": "slide_001",
      "slide_path": "/mnt/data/slides/slide_001.svs",
      "output_h5_path": "/mnt/output/slide_001_features.h5",
      "output_pt_path": "/mnt/output/slide_001_features.pt"
    },
    {
      "task_id": "slide_002",
      "slide_path": "/mnt/data/slides/slide_002.svs",
      "output_h5_path": "/mnt/output/slide_002_features.h5",
      "output_pt_path": "/mnt/output/slide_002_features.pt"
    }
  ]
}
```

Submit the batch:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatchaccount \
  --batch-account-key <your-batch-key> \
  --batch-account-url https://mybatchaccount.eastus.batch.azure.com \
  --pool-id mussel-pool \
  --create-pool \
  --job-id mussel-job-002 \
  --create-job \
  --config-file tasks.json \
  --monitor
```

### Processing Slides from S3 with CSV Manifest

For processing slides stored in S3, use a CSV manifest file:

**manifest.csv:**
```csv
slide_id,slide_path
slide_001,s3://my-bucket/slides/slide_001.svs
slide_002,s3://my-bucket/slides/slide_002.svs
slide_003,s3://my-bucket/slides/slide_003.svs
```

Submit with S3 output:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatchaccount \
  --batch-account-key <your-batch-key> \
  --batch-account-url https://mybatchaccount.eastus.batch.azure.com \
  --pool-id mussel-pool \
  --create-pool \
  --job-id mussel-job-003 \
  --create-job \
  --csv-manifest manifest.csv \
  --output-s3-prefix s3://my-bucket/results/ \
  --aws-access-key-id $AWS_ACCESS_KEY_ID \
  --aws-secret-access-key $AWS_SECRET_ACCESS_KEY \
  --aws-region us-east-1 \
  --monitor
```

Or output to local directory:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatchaccount \
  --batch-account-key <your-batch-key> \
  --batch-account-url https://mybatchaccount.eastus.batch.azure.com \
  --pool-id mussel-pool \
  --job-id mussel-job-004 \
  --csv-manifest manifest.csv \
  --output-dir /mnt/output \
  --aws-access-key-id $AWS_ACCESS_KEY_ID \
  --aws-secret-access-key $AWS_SECRET_ACCESS_KEY \
  --monitor
```

### Single Task with S3 Input/Output

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatchaccount \
  --batch-account-key <your-batch-key> \
  --batch-account-url https://mybatchaccount.eastus.batch.azure.com \
  --pool-id mussel-pool \
  --job-id mussel-job-005 \
  --task-id slide-s3-001 \
  --slide-path s3://my-bucket/slides/slide.svs \
  --output-h5-path s3://my-bucket/results/slide_features.h5 \
  --output-pt-path s3://my-bucket/results/slide_features.pt \
  --aws-access-key-id $AWS_ACCESS_KEY_ID \
  --aws-secret-access-key $AWS_SECRET_ACCESS_KEY \
  --monitor
```

## Configuration

### Pool Configuration

Control the compute resources:

- `--vm-size`: VM size (default: `Standard_NC6s_v3` with 1 GPU)
  - GPU VMs: `Standard_NC6s_v3`, `Standard_NC12s_v3`, `Standard_NC24s_v3`
  - CPU VMs: `Standard_D4s_v3`, `Standard_D8s_v3`
- `--node-count`: Number of nodes in the pool
- `--container-image`: Docker image to use

### Task Configuration

Each task can specify:

- **Required:**
  - `slide_path`: Path to the whole-slide image
  - `output_h5_path`: Path for output HDF5 file
  - `output_pt_path`: Path for output PyTorch file

- **Optional:**
  - `classifier_pkl`: Path to classifier for filtering
  - `classifier_threshold`: Threshold for filtering (default: 0.75)
  - `prefilter_model_type`: Model for pre-filter extraction (default: CTRANSPATH)
  - `postfilter_model_type`: Model for post-filter extraction
  - `segment_threshold`: Tissue segmentation threshold (default: 0)
  - `patch_size`: Patch size in pixels (default: 256)
  - `mpp`: Microns per pixel (default: 0.5)
  - `num_workers`: Number of workers (default: 4)
  - `batch_size`: Batch size for feature extraction (default: 64)
  - `use_gpu`: Whether to use GPU (default: true)
  - `keep_intermediate_files`: Keep intermediate files (default: false)
  - `hf_token`: HuggingFace token for gated models

### Available Models

Pre-filter and post-filter model types:
- `RESNET50`: ResNet-50
- `CTRANSPATH`: TransPath (default)
- `CLIP`: OpenCLIP (QuiltNet)
- `VIRCHOW`: Virchow
- `VIRCHOW2`: Virchow2
- `HOPTIMUS0`: H-Optimus-0
- `GIGAPATH`: Prov-GigaPath
- `CONCH`: Conch v1.5

## Data Management

### Using S3 Storage (Recommended for Large Datasets)

The scripts natively support S3 for input slides and output results:

**Features:**
- Automatic staging: slides are downloaded from S3 to local temp storage before processing
- Automatic publishing: results are uploaded to S3 after processing completes
- Mixed paths: slides can be from S3 or local, outputs can go to S3 or local
- AWS CLI integration: uses standard AWS credentials

**Setup:**
1. Install AWS CLI in your Docker image or ensure it's available
2. Provide AWS credentials via command-line arguments:
   - `--aws-access-key-id`
   - `--aws-secret-access-key`
   - `--aws-region` (default: us-east-1)

**Example with S3:**
```bash
# Input from S3, output to S3
--slide-path s3://my-bucket/slides/slide.svs \
--output-h5-path s3://my-bucket/results/slide.h5 \
--output-pt-path s3://my-bucket/results/slide.pt

# Input from S3, output local
--slide-path s3://my-bucket/slides/slide.svs \
--output-h5-path /mnt/output/slide.h5 \
--output-pt-path /mnt/output/slide.pt

# Input local, output to S3
--slide-path /mnt/data/slide.svs \
--output-h5-path s3://my-bucket/results/slide.h5 \
--output-pt-path s3://my-bucket/results/slide.pt
```

### CSV Manifest Format

For batch processing from a slide list:

```csv
slide_id,slide_path
slide_001,s3://bucket/path/slide_001.svs
slide_002,s3://bucket/path/slide_002.svs
slide_003,/local/path/slide_003.svs
```

- **slide_id**: Unique identifier for the slide (used for task ID and output filenames)
- **slide_path**: Path to slide (can be S3 URL or local path)

When using `--csv-manifest`, outputs are automatically named as `{slide_id}_features.h5` and `{slide_id}_features.pt`.

### Using Azure Storage

Mount Azure Storage as a file share on Batch nodes:

1. Create a file share:
```bash
az storage share create \
  --account-name mystorageaccount \
  --name myshare
```

2. Upload slides:
```bash
az storage file upload-batch \
  --account-name mystorageaccount \
  --destination myshare/slides \
  --source ./local_slides/
```

3. Configure mount in your tasks (modify the submission script to add mount configuration)

### Local Files

Alternatively, you can:
- Use Azure Blob Storage and download files in the task script
- Bake slides into a custom Docker image (for small datasets)
- Use resource files in Azure Batch

## Monitoring

### Monitor Tasks

The `--monitor` flag enables real-time monitoring:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatchaccount \
  --batch-account-key <your-batch-key> \
  --batch-account-url https://mybatchaccount.eastus.batch.azure.com \
  --pool-id mussel-pool \
  --job-id mussel-job-001 \
  --monitor
```

### Check Status via Azure Portal

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your Batch account
3. Click on "Jobs" to see all jobs
4. Click on a job to see individual tasks
5. Click on a task to view logs and output

### Azure CLI

Check job status:
```bash
az batch job show --job-id mussel-job-001
```

List tasks:
```bash
az batch task list --job-id mussel-job-001 --output table
```

Get task output:
```bash
az batch task file download \
  --job-id mussel-job-001 \
  --task-id slide-001 \
  --file-path stdout.txt \
  --destination ./task-output.txt
```

## Automatic File Cleanup

The task execution script automatically cleans up temporary files when tasks complete:

**What gets cleaned up:**
- Staged slide files downloaded from S3 (stored in `/tmp/mussel_work_*`)
- Temporary output files (when outputs are uploaded to S3)
- Any intermediate work directories created during processing

**When cleanup happens:**
- On successful task completion (after uploading results to S3 if applicable)
- On task failure (to free up disk space)
- On script interruption (via EXIT, INT, or TERM signals)

**Cleanup is automatic and requires no configuration.** The trap mechanism ensures cleanup occurs even if the script exits unexpectedly.

**Note:** Only temporary files are removed. Final output files stored in local directories (non-S3 paths) are preserved.

## Cleanup

### Delete Job

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatchaccount \
  --batch-account-key <your-batch-key> \
  --batch-account-url https://mybatchaccount.eastus.batch.azure.com \
  --job-id mussel-job-001 \
  --delete-job
```

### Delete Pool

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatchaccount \
  --batch-account-key <your-batch-key> \
  --batch-account-url https://mybatchaccount.eastus.batch.azure.com \
  --pool-id mussel-pool \
  --delete-pool
```

### Using Azure CLI

```bash
# Delete job
az batch job delete --job-id mussel-job-001 --yes

# Delete pool
az batch pool delete --pool-id mussel-pool --yes
```

## Advanced Usage

### Using Gated Models

Some models (e.g., Prov-GigaPath, GooglePath, Virchow) require HuggingFace authentication:

1. Create a HuggingFace token at https://huggingface.co/settings/tokens
2. Accept the model license on HuggingFace
3. Pass the token in your configuration:

```json
{
  "tasks": [
    {
      "task_id": "slide_001",
      "slide_path": "/mnt/data/slide.svs",
      "output_h5_path": "/mnt/output/features.h5",
      "output_pt_path": "/mnt/output/features.pt",
      "prefilter_model_type": "GIGAPATH",
      "hf_token": "hf_..."
    }
  ]
}
```

### Custom Docker Image

Build and use a custom image:

```bash
# Build image
docker build -t myregistry.azurecr.io/mussel:custom .

# Push to Azure Container Registry
az acr login --name myregistry
docker push myregistry.azurecr.io/mussel:custom

# Use in submission
python scripts/azure_batch/submit_batch_jobs.py \
  --container-image myregistry.azurecr.io/mussel:custom \
  ...
```

### Auto-scaling

Enable auto-scaling for dynamic workloads:

```python
# Modify submit_batch_jobs.py to use auto-scale formula
auto_scale_formula = """
    maxNumberofVMs = 10;
    pendingTaskSamplePercent = $PendingTasks.GetSamplePercent(180 * TimeInterval_Second);
    pendingTaskSamples = pendingTaskSamplePercent < 70 ? startingNumberOfVMs : avg($PendingTasks.GetSample(180 * TimeInterval_Second));
    $TargetDedicatedNodes = min(maxNumberofVMs, pendingTaskSamples);
"""
```

## Troubleshooting

### Task Failures

1. Check task logs in Azure Portal or via CLI
2. Look for error messages in stdout.txt and stderr.txt
3. Common issues:
   - Slide file not found: Check mount paths
   - GPU not available: Verify VM size has GPU
   - Out of memory: Reduce batch_size or num_workers
   - Model access denied: Verify HF_TOKEN is set correctly

### Pool Issues

1. Verify pool is active: `az batch pool show --pool-id mussel-pool`
2. Check node state: `az batch node list --pool-id mussel-pool --output table`
3. If nodes are unusable, delete and recreate pool

### Container Issues

1. Verify Docker image exists and is accessible
2. Check container registry authentication
3. Test image locally: `docker run mskmind/mussel:latest-torch-gpu tessellate_extract_features --help`

## Cost Optimization

1. **Use low-priority VMs**: Reduce costs by up to 80%
2. **Auto-scale**: Scale down when idle
3. **Delete pools**: Remove pools when not in use
4. **Batch size**: Optimize batch_size and num_workers for your VM
5. **Region selection**: Choose regions with lower costs

## Example Workflows

### Process 100 slides with filtering

```json
{
  "defaults": {
    "prefilter_model_type": "CTRANSPATH",
    "classifier_pkl": "/mnt/data/tissue_classifier.pkl",
    "classifier_threshold": 0.75,
    "postfilter_model_type": "CLIP",
    "batch_size": 128,
    "num_workers": 8
  },
  "tasks": [
    {
      "task_id": "slide_001",
      "slide_path": "/mnt/data/slides/slide_001.svs",
      "output_h5_path": "/mnt/output/slide_001_features.h5",
      "output_pt_path": "/mnt/output/slide_001_features.pt"
    }
    // ... 99 more slides
  ]
}
```

### Mixed workload (different models per slide)

```json
{
  "tasks": [
    {
      "task_id": "lung_slide_001",
      "slide_path": "/mnt/data/lung/slide_001.svs",
      "output_h5_path": "/mnt/output/lung_001_features.h5",
      "output_pt_path": "/mnt/output/lung_001_features.pt",
      "prefilter_model_type": "VIRCHOW"
    },
    {
      "task_id": "breast_slide_001",
      "slide_path": "/mnt/data/breast/slide_001.svs",
      "output_h5_path": "/mnt/output/breast_001_features.h5",
      "output_pt_path": "/mnt/output/breast_001_features.pt",
      "prefilter_model_type": "GIGAPATH"
    }
  ]
}
```

## Additional Resources

- [Azure Batch Documentation](https://docs.microsoft.com/en-us/azure/batch/)
- [Azure Batch Python SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/batch)
- [Mussel Documentation](../../README.md)
- [Docker Hub - Mussel Images](https://hub.docker.com/r/mskmind/mussel)

## Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/pathology-data-mining/Mussel/issues)
- Check the main [Mussel documentation](../../README.md)
- Review [Azure Batch troubleshooting](https://docs.microsoft.com/en-us/azure/batch/batch-common-errors)
