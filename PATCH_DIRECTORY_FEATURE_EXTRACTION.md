# Patch Directory Feature Extraction Guide

## Overview

The `mussel extract_features` CLI now clearly documents three distinct input modes for feature extraction. This guide focuses on **Patch Directory Mode**, which processes pre-extracted patch images.

## Three Input Modes

### 1. Single Slide Mode
Processes one slide from patch coordinates in HDF5 file:
```bash
extract_features \
    patch_h5_path=/path/to/patches.h5 \
    slide_path=/path/to/slide.svs \
    output_h5_path=/path/to/output.h5 \
    output_pt_path=/path/to/output.pt \
    model_type=OPTIMUS
```
**Output**: Single H5 and PT file with features for one slide

### 2. Batch Slides Mode
Processes multiple slides efficiently (batch processing):
```bash
extract_features \
    patch_h5_paths=[/path/to/slide1.h5,/path/to/slide2.h5] \
    slide_paths=[/path/to/slide1.svs,/path/to/slide2.svs] \
    output_dir=/path/to/output_dir \
    model_type=OPTIMUS
```
**Output**: Multiple H5/PT files (one per slide) in output_dir

### 3. Patch Directory Mode (Focus of this Guide)
Processes pre-extracted patch images from a directory:
```bash
extract_features \
    patch_path=/path/to/patches_dir \
    output_h5_path=/path/to/output.h5 \
    output_pt_path=/path/to/output.pt \
    slide_path=None \
    patch_h5_path=None \
    model_type=OPTIMUS
```
**Output**: Single H5 and PT file containing features for ALL patches in directory

**Important Note**: The output is **aggregated** - one H5 file and one PT file for all patches in the directory, NOT separate files per patch image.

## SLURM Batch Processing for Patch Directories

A complete SLURM submission infrastructure exists for processing multiple patch directories in parallel.

### Quick Start

1. **Prepare CSV manifest** with slide_id and patch_dir columns:
```csv
slide_id,patch_dir
slide001,/path/to/slide001/patches
slide002,s3://bucket/slide002/patches
slide003,/data/slide003/patches
```

2. **Submit jobs** (dry-run first, then add `--submit`):
```bash
python scripts/slurm/submit_patch_extract_jobs.py \
    --csv-manifest patches.csv \
    --output-dir /path/to/output \
    --model-type OPTIMUS \
    --model-dir /path/to/models \
    --partition gpu \
    --time 4:00:00 \
    --mem 32G \
    --cpus-per-task 8 \
    --gpus 1 \
    --submit
```

### Key Features

- **S3 Support**: Patch directories can be S3 paths (s3://bucket/prefix/)
- **Automatic Staging**: S3 directories are downloaded locally before processing
- **Environment File**: Load credentials from `.env` file using `--env-file secrets.env`
- **Apptainer Support**: Use containerized execution with `--use-apptainer`
- **Dry-run Mode**: Test without submitting (default behavior without `--submit`)

### Configuration Options

#### Input/Output
- `--csv-manifest`: CSV file with slide_id,patch_dir columns (required)
- `--output-dir`: Output directory for features (required)

#### Model Configuration
- `--model-type`: Model type (default: OPTIMUS)
- `--model-path`: Optional path to model weights
- `--model-dir`: Directory where models are cached

#### Processing
- `--batch-size`: Batch size for feature extraction (default: 64)
- `--num-workers`: Number of data loading workers (default: 4)
- `--use-gpu`: Use GPU true/false (default: true)

#### AWS S3 Configuration
- `--aws-access-key-id`: AWS access key (or AWS_ACCESS_KEY_ID env var)
- `--aws-secret-access-key`: AWS secret key (or AWS_SECRET_ACCESS_KEY env var)
- `--aws-region`: AWS region (default: us-east-1)
- `--aws-endpoint-url`: Custom S3 endpoint (e.g., for MinIO)

#### SLURM Configuration
- `--partition`: SLURM partition (default: gpu)
- `--time`: Job time limit (default: 4:00:00)
- `--mem`: Memory per job (default: 32G)
- `--cpus-per-task`: CPUs per task (default: 8)
- `--gpus`: Number of GPUs (default: 1)
- `--gpu-type`: GPU type constraint (e.g., a100, v100)

#### Job Configuration
- `--job-name`: Job name prefix (default: patch-extract)
- `--log-dir`: Directory for SLURM logs (default: slurm_logs)
- `--use-apptainer`: Use Apptainer/Singularity container
- `--apptainer-image`: Path to Apptainer/Singularity image

#### Execution
- `--submit`: Submit jobs to SLURM (default: dry-run without this flag)
- `--max-jobs`: Maximum number of jobs to submit

### Configuration File

Instead of command-line arguments, you can use a YAML config file:

```yaml
# slurm_patch_config.yaml
csv_manifest: patches.csv
output_dir: /path/to/output
model_type: OPTIMUS
model_dir: /path/to/models
partition: gpu
time: "4:00:00"
mem: 32G
cpus_per_task: 8
gpus: 1
submit: true
env_file: secrets.env
```

Then run:
```bash
python scripts/slurm/submit_patch_extract_jobs.py --config slurm_patch_config.yaml
```

### Example Workflows

#### 1. Local Patch Directories
```bash
python scripts/slurm/submit_patch_extract_jobs.py \
    --csv-manifest local_patches.csv \
    --output-dir /data/features \
    --model-type UNI \
    --model-dir /data/models \
    --submit
```

#### 2. S3 Patch Directories
```bash
python scripts/slurm/submit_patch_extract_jobs.py \
    --csv-manifest s3_patches.csv \
    --output-dir /data/features \
    --model-type GIGAPATH \
    --env-file secrets.env \
    --submit
```

#### 3. Using Apptainer Container
```bash
python scripts/slurm/submit_patch_extract_jobs.py \
    --csv-manifest patches.csv \
    --output-dir /data/features \
    --model-type OPTIMUS \
    --use-apptainer \
    --apptainer-image /path/to/mussel.sif \
    --submit
```

## Output Format

For each patch directory (slide), the output consists of:

1. **HDF5 File** (`{slide_id}.h5`):
   - Contains feature embeddings for all patches
   - Includes metadata like image paths and class labels

2. **PyTorch File** (`{slide_id}.pt`):
   - Contains feature tensor for all patches
   - Can be loaded with `torch.load()`

**Important**: These are **aggregated outputs** containing features for ALL patch images in the directory, not separate files per patch image.

## Monitoring Jobs

After submission, monitor your jobs:

```bash
# Check job status
squeue -u $USER

# Check specific jobs
squeue -j job_id1,job_id2

# Check logs (in slurm_logs directory by default)
tail -f slurm_logs/slide001_*.out
tail -f slurm_logs/slide001_*.err
```

## Troubleshooting

### S3 Access Issues
Ensure AWS credentials are set:
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
# Or use --env-file secrets.env
```

### Missing Patch Images
Check the patch directory contains supported formats:
- PNG (.png)
- JPEG (.jpg, .jpeg)
- TIFF (.tif, .tiff)

### GPU Memory Issues
Reduce batch size:
```bash
--batch-size 32  # or smaller
```

### Model Download Issues
Pre-download models to a shared directory and use `--model-dir`:
```bash
--model-dir /shared/models
```

## File Structure

```
scripts/
├── slurm/
│   └── submit_patch_extract_jobs.py    # SLURM submission script
└── common/
    └── run_extract_features_patches.sh  # Task execution script
```

## Related Documentation

- Main README: [README.md](README.md)
- SLURM Batch Processing: [README_BATCH_PROCESSING.md](README_BATCH_PROCESSING.md)
- Extract Features CLI: Run `extract_features --help` for all options
