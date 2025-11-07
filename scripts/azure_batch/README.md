# Azure Batch Scripts for Mussel

This directory contains scripts for running `tessellate-extract-features` on Azure Batch, enabling large-scale processing of whole-slide images in the cloud with support for S3 and Azure Files storage.

## Overview

The Azure Batch integration allows you to:
- Process multiple whole-slide images in parallel
- Scale computation using Azure's cloud infrastructure
- Use GPU-enabled VMs for fast feature extraction
- Manage long-running jobs with automatic retry and monitoring
- Stage slides from S3 or local storage to Azure Files for preprocessing
- Mount Azure Files to batch nodes for direct access (eliminating download overhead)
- Publish results to S3, Azure Storage, or local directories
- Process slides from CSV manifests with slide identifiers
- Automatic cleanup of staged files and temporary files upon task completion

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

Install the required Python packages. If using `uv`:

```bash
uv sync --extra distributed
```

Or with `pip`:

```bash
pip install azure-batch azure-storage-blob azure-storage-file-share azure-identity boto3
```

The `distributed` extra includes all dependencies needed for Azure Batch submission scripts and S3 integration.

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

### Slide Batch Processing (Optimized)

**NEW**: Process multiple slides together to optimize slide encoder loading:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatchaccount \
  --batch-account-key <your-batch-key> \
  --batch-account-url https://mybatchaccount.eastus.batch.azure.com \
  --pool-id mussel-pool \
  --create-pool \
  --job-id mussel-job-batch \
  --create-job \
  --csv-manifest manifest.csv \
  --output-s3-prefix s3://my-bucket/results/ \
  --aggregation-method model \
  --slide-model-type GIGAPATH_SLIDE \
  --distributed-slide-batch-size 8 \
  --aws-access-key-id $AWS_ACCESS_KEY_ID \
  --aws-secret-access-key $AWS_SECRET_ACCESS_KEY \
  --monitor
```

**What this does:**
- Groups 8 slides per Azure Batch task
- Loads slide encoder model ONCE per task
- **7-8x speedup** for slide-level aggregation workloads

**When to use:**
- Processing 2+ slides with slide-level model aggregation
- Using GIGAPATH_SLIDE, TITAN_SLIDE, or similar models
- NOT using --stage-to-azure-files (incompatible with batching)

**Note:** When using `--stage-to-azure-files`, incremental staging creates one task per slide, and `--distributed-slide-batch-size` is not used.

See [examples/distributed_batch_processing.md](../../examples/distributed_batch_processing.md) for detailed guide.

### Azure Files Staging (Preprocessing)

For optimal performance when processing many slides, you can stage input files to Azure Files before processing. Azure Files can be mounted directly to batch nodes, eliminating download overhead during task execution.

**Benefits:**
- **Incremental processing**: Tasks start as soon as slides are staged (no waiting for full batch)
- **Faster task startup**: No per-task download time
- **Reduced egress costs**: Lower S3 egress costs
- **Better resource utilization**: Batch nodes can start processing immediately
- **Automatic per-task cleanup**: Staged files removed after each task completes, freeing storage space progressively
- **Centralized storage**: All batch nodes access same files
- **Lower storage costs**: Per-task cleanup means you only pay for storage during active processing

**Setup:**

1. Create an Azure Files share (or use existing):
```bash
az storage share create \
  --name mussel-staging \
  --account-name mystorageaccount \
  --account-key $STORAGE_KEY
```

2. Stage slides and run with Azure Files:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatchaccount \
  --batch-account-key <your-batch-key> \
  --batch-account-url https://mybatchaccount.eastus.batch.azure.com \
  --storage-account-name mystorageaccount \
  --storage-account-key <your-storage-key> \
  --azure-files-share-name mussel-staging \
  --pool-id mussel-pool \
  --create-pool \
  --mount-azure-files \
  --job-id mussel-job-006 \
  --create-job \
  --csv-manifest manifest.csv \
  --stage-to-azure-files \
  --output-s3-prefix s3://my-bucket/results/ \
  --aws-access-key-id $AWS_ACCESS_KEY_ID \
  --aws-secret-access-key $AWS_SECRET_ACCESS_KEY \
  --monitor \
  --cleanup-staged-files
```

**Workflow:**
1. Pool is created with Azure Files mount at `/mnt/batch/tasks/fsmounts/azfiles/`
2. For each slide in the manifest:
   - Slide is staged to Azure Files share
   - Task is immediately submitted to process the staged slide
   - Processing can start as soon as the first slide is staged (no need to wait for all slides)
   - **After task completes successfully, the staged file is automatically cleaned up**
3. Tasks access slides directly from mounted Azure Files (no download needed)
4. Cleanup happens per-task, so storage space is freed as tasks complete

**Incremental Processing:**
When `--stage-to-azure-files` is enabled, slides are staged and tasks are submitted one by one. This means:
- Processing starts as soon as the first slide is uploaded
- No need to wait for all slides to be staged before processing begins
- Better utilization of batch resources (nodes can start working immediately)
- Faster overall throughput for large batches
- **Automatic per-task cleanup**: Each task cleans up its staged file after successful completion

**Key Arguments:**
- `--azure-files-share-name`: Name of Azure Files share for staging
- `--stage-to-azure-files`: Enable incremental staging of input files to Azure Files
- `--mount-azure-files`: Mount Azure Files share to batch pool nodes
- `--cleanup-staged-files`: (Optional) Remove any remaining staged files after all tasks complete (normally not needed with incremental staging)

## Configuration

### Pool Configuration

Control the compute resources:

- `--vm-size`: VM size (default: `Standard_NC6s_v3` with 1 GPU)
  - GPU VMs (V100): `Standard_NC6s_v3`, `Standard_NC12s_v3`, `Standard_NC24s_v3`
  - GPU VMs (A100): `Standard_NC24ads_A100_v4`, `Standard_NC48ads_A100_v4`, `Standard_NC96ads_A100_v4`
  - GPU VMs (H100): `Standard_NC40ads_H100_v5`, `Standard_NC80ads_H100_v5`
  - CPU VMs: `Standard_D4s_v3`, `Standard_D8s_v3`
- `--node-count`: Number of nodes in the pool (or initial/min count for auto-scaling)
- `--use-gpu`: Enable GPU support for pool nodes (default: True)
- `--no-gpu`: Disable GPU support for pool nodes (for CPU-only workloads)
- `--container-image`: Docker image to use
  - For GPU workloads: `mskmind/mussel:latest-torch-gpu` (default)
  - For CPU workloads: `mskmind/mussel:latest-torch-cpu`

#### Auto-Scaling Configuration

Enable auto-scaling to dynamically adjust pool size based on workload:

- `--enable-auto-scale`: Enable auto-scaling based on pending tasks
- `--min-node-count`: Minimum number of nodes (defaults to `--node-count`)
- `--max-node-count`: Maximum number of nodes (required if auto-scaling is enabled)
- `--auto-scale-evaluation-interval`: Evaluation interval in minutes (default: 15)

**How auto-scaling works:**
- Pool starts with `--min-node-count` nodes (or `--node-count` if min not specified)
- Scales up to `--max-node-count` based on pending tasks
- Evaluates workload every `--auto-scale-evaluation-interval` minutes
- Automatically scales down when tasks complete
- **Handles unusable nodes**: Automatically accounts for and replaces nodes that become unusable, maintaining pool capacity

**Example: Create a GPU pool with fixed size**
```bash
--pool-id mussel-gpu-pool \
--create-pool \
--vm-size Standard_NC6s_v3 \
--node-count 2 \
--use-gpu \
--container-image mskmind/mussel:latest-torch-gpu
```

**Example: Create an auto-scaling GPU pool**
```bash
--pool-id mussel-autoscale-pool \
--create-pool \
--vm-size Standard_NC6s_v3 \
--node-count 1 \
--enable-auto-scale \
--min-node-count 1 \
--max-node-count 10 \
--use-gpu \
--container-image mskmind/mussel:latest-torch-gpu
```

**Example: Create a CPU-only pool**
```bash
--pool-id mussel-cpu-pool \
--create-pool \
--vm-size Standard_D4s_v3 \
--node-count 4 \
--no-gpu \
--container-image mskmind/mussel:latest-torch-cpu
```

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
  - `intermediate_h5_path`: Path for intermediate tile-level features (for slide-level aggregation)
  - `aggregation_method`: Aggregation method: identity (default), mean, max, model
  - `slide_model_type`: Slide model type for aggregation_method=model
  - `segment_threshold`: Tissue segmentation threshold (default: 0)
  - `patch_size`: Patch size in pixels (default: 256)
  - `mpp`: Microns per pixel (default: 0.5)
  - `num_workers`: Number of workers (default: 4)
  - `batch_size`: Batch size for feature extraction (default: 64)
  - `use_gpu`: Whether to use GPU (default: true)
  - `keep_intermediate_files`: Keep intermediate files (default: false)
  - `hf_token`: HuggingFace token for gated models
  - `max_retry_count`: Maximum number of retry attempts for failed tasks (default: 3)

### Running Multiple Postfilter Models

Process each slide with multiple postfilter models **sequentially in the same task**:

**Usage:**
```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --csv-manifest slides.csv \
  --output-s3-prefix s3://bucket/results/ \
  --postfilter-models CTRANSPATH,CLIP,VIRCHOW \
  --monitor
```

**How it works:**
- Each slide is processed with all specified postfilter models sequentially
- Models run **within the same task** (not separate tasks)
- Slide is read once and cached in memory for efficiency
- Each model's results are saved to model-specific subdirectories
- Task IDs remain as `{slide_id}` (one task per slide)

**Example output structure:**
```
s3://bucket/results/
├── CTRANSPATH/
│   ├── h5/slide_001_features.h5
│   └── pt/slide_001_features.pt
├── CLIP/
│   ├── h5/slide_001_features.h5
│   └── pt/slide_001_features.pt
└── VIRCHOW/
    ├── h5/slide_001_features.h5
    └── pt/slide_001_features.pt
```

**Benefits:**
- **Efficient**: Slide read once and reused for all models
- **Sequential processing**: Models run one after another in same task
- **Organized outputs**: Each model's results in dedicated subdirectories
- **Single job**: One task per slide, regardless of model count
- **Cost effective**: Fewer task overhead, better resource utilization

**Available models:** RESNET50, CTRANSPATH, CLIP, VIRCHOW, VIRCHOW2, HOPTIMUS0, GIGAPATH, CONCH

### Retry Configuration and Failure Handling

Tasks automatically retry on failure up to the specified maximum attempts:

**Configuration:**
- `--max-retry-count`: Set maximum retry attempts (default: 3)
- `--save-failed-tasks`: Save list of failed tasks to CSV file for resubmission

**Automatic Retry:**
Azure Batch automatically retries failed tasks up to `max_retry_count` times. This handles transient failures like:
- Network interruptions
- Temporary resource unavailability
- Sporadic compute errors

**Saving Failed Tasks:**
After job completion, save tasks that failed all retry attempts:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatch \
  --batch-account-key $KEY \
  --batch-account-url https://mybatch.batch.azure.com \
  --pool-id mussel-pool \
  --job-id mussel-job-001 \
  --save-failed-tasks failed_tasks.csv
```

The failed tasks CSV contains:
- `task_id`: Task identifier
- `slide_path`: Original slide path
- `output_h5_path`, `output_pt_path`: Output paths
- `state`: Task final state
- `exit_code`: Exit code from failed task

**Resubmitting Failed Tasks:**
Use the failed tasks CSV to resubmit:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatch \
  --batch-account-key $KEY \
  --batch-account-url https://mybatch.batch.azure.com \
  --pool-id mussel-pool \
  --job-id mussel-job-retry \
  --create-job \
  --csv-manifest failed_tasks.csv \
  --max-retry-count 5 \
  --monitor
```

### Results Manifest Generation

Generate a comprehensive manifest of all successfully completed result files:

**Generate Manifest:**
```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatch \
  --batch-account-key $KEY \
  --batch-account-url https://mybatch.batch.azure.com \
  --job-id mussel-job-001 \
  --generate-manifest results_manifest.csv
```

**Manifest Contents:**
The results manifest CSV includes:
- `task_id`: Task identifier (slide ID)
- `slide_path`: Input slide path
- `output_h5_path`: Slide-level HDF5 features path
- `output_pt_path`: Slide-level PyTorch features path
- `intermediate_h5_path`: Tile-level features path (if aggregation was used)
- `model_type`: Model used for feature extraction (e.g., CTRANSPATH, CLIP)
- `file_type`: File type (h5, pt, tile_h5)
- `state`: Task state (completed)
- `exit_code`: Exit code (0 for success)

**Use Cases:**
- Track all generated result files for downstream processing
- Document which model was used for each slide
- Verify completeness of batch processing
- Generate inventory for data management

**Example with monitoring and manifest:**
```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --csv-manifest slides.csv \
  --output-s3-prefix s3://bucket/results/ \
  --monitor \
  --generate-manifest results_manifest.csv \
  --save-failed-tasks failed.csv
```

This workflow monitors progress, generates a manifest of successful results, and saves any failures for resubmission.

### Slide-Level Aggregation

When performing slide-level aggregation (e.g., `aggregation_method=mean`), the workflow produces both:
1. **Tile-level features** (patch embeddings) - saved to `intermediate_h5_path`
2. **Slide-level features** (aggregated from tiles) - saved to `output_h5_path` and `output_pt_path`

Both feature files are automatically published to the specified output directory or S3 prefix.

**Example with aggregation:**
```python
# In config JSON or CSV defaults
{
  "aggregation_method": "mean",
  "intermediate_h5_path": "s3://bucket/results/slide_001_tile_features.h5",
  "output_h5_path": "s3://bucket/results/slide_001_slide_features.h5",
  "output_pt_path": "s3://bucket/results/slide_001_slide_features.pt"
}
```

When using CSV manifests, tile-level features are automatically named as `{slide_id}_tile_features.h5`.

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
- **Organized output structure**: Results are automatically organized by model type and file type

**Output Directory Structure:**

When using CSV manifests, results are organized into subdirectories:
```
{output_dir}/
├── {model_type}/           # e.g., CTRANSPATH, CLIP, VIRCHOW
│   ├── h5/                 # Slide-level HDF5 features
│   │   └── slide_001_features.h5
│   ├── pt/                 # Slide-level PyTorch features
│   │   └── slide_001_features.pt
│   └── tile_h5/            # Tile-level HDF5 features (when using aggregation)
│       └── slide_001_tile_features.h5
```

Example S3 structure:
```
s3://bucket/results/
├── CTRANSPATH/
│   ├── h5/slide_001_features.h5
│   ├── pt/slide_001_features.pt
│   └── tile_h5/slide_001_tile_features.h5
```

**Setup:**
1. Install AWS CLI in your Docker image or ensure it's available
2. Provide AWS credentials via command-line arguments:
   - `--aws-access-key-id`
   - `--aws-secret-access-key`
   - `--aws-region` (default: us-east-1)
   - `--aws-endpoint-url` (optional, for S3-compatible storage like MinIO or Ceph)

**Example with S3:**
```bash
# Input from S3, output to S3 (organized structure)
--slide-path s3://my-bucket/slides/slide.svs \
--output-h5-path s3://my-bucket/results/CTRANSPATH/h5/slide.h5 \
--output-pt-path s3://my-bucket/results/CTRANSPATH/pt/slide.pt

# Input from S3, output local
--slide-path s3://my-bucket/slides/slide.svs \
--output-h5-path /mnt/output/slide.h5 \
--output-pt-path /mnt/output/slide.pt

# Input local, output to S3
--slide-path /mnt/data/slide.svs \
--output-h5-path s3://my-bucket/results/slide.h5 \
--output-pt-path s3://my-bucket/results/slide.pt

# Using custom S3 endpoint (e.g., MinIO)
--slide-path s3://my-bucket/slides/slide.svs \
--output-h5-path s3://my-bucket/results/slide.h5 \
--output-pt-path s3://my-bucket/results/slide.pt \
--aws-endpoint-url http://minio.example.com:9000
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

### Automatic Cleanup After Completion

When you want to automatically delete resources after all tasks complete, use the cleanup flags with `--monitor`:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatchaccount \
  --batch-account-key <your-batch-key> \
  --batch-account-url https://mybatchaccount.eastus.batch.azure.com \
  --pool-id mussel-pool \
  --create-pool \
  --job-id mussel-job-001 \
  --create-job \
  --csv-manifest slides.csv \
  --output-dir /mnt/output \
  --monitor \
  --delete-job \
  --delete-pool
```

**How it works:**
- The script will monitor tasks until all complete
- After monitoring completes, the job is deleted (if `--delete-job` is specified)
- Then the pool is deleted (if `--delete-pool` is specified)
- This ensures resources are cleaned up only after processing finishes

**Note:** If you use `--delete-job` or `--delete-pool` without `--monitor`, the resources will be deleted immediately, which may terminate running tasks. Always use these flags together with `--monitor` when you want cleanup after completion.

### Manual Cleanup

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
