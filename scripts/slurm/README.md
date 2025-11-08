# SLURM Backend for Mussel Feature Extraction

This directory contains scripts for running `tessellate-extract-features` on SLURM clusters.

## Overview

SLURM support enables distributed processing of whole-slide images on HPC clusters using the SLURM workload manager.

**Key Features:**
- **Slide batch feature extraction** for optimized slide encoder loading (7-8x speedup)
- Job arrays for efficient batch processing
- CSV manifest processing
- Partition and QOS selection
- GPU resource allocation
- Automatic retry via job dependencies
- S3 staging and publishing
- Multi-model optimization (filter-tessellate + extract-features)

## Requirements

- SLURM installed and configured
- Access to SLURM submit node
- Python 3.7+
- Mussel environment installed on compute nodes
- (Optional) For S3 support: Install the `distributed` extra with `uv sync --extra distributed` or `pip install boto3`

## Quick Start

### Single Slide

```bash
python scripts/slurm/submit_slurm_jobs.py \
  --job-name slide-001 \
  --slide-path /data/slides/slide_001.svs \
  --output-h5-path /output/slide_001_features.h5 \
  --output-pt-path /output/slide_001_features.pt \
  --partition gpu \
  --gres gpu:1 \
  --submit
```

### Batch Processing with Job Array

```bash
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest manifest.csv \
  --output-dir /output/results/ \
  --partition gpu \
  --gres gpu:1 \
  --submit
```

### Multi-Model Processing

```bash
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest slides.csv \
  --output-s3-prefix s3://bucket/results/ \
  --postfilter-models CTRANSPATH,CLIP,VIRCHOW \
  --partition gpu \
  --gres gpu:1 \
  --submit
```

### Slide Batch Processing (Optimized)

**NEW**: Process multiple slides together to optimize model loading and GPU utilization:

```bash
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest slides.csv \
  --output-dir /output/results/ \
  --distributed-slide-batch-size 8 \
  --partition gpu \
  --gres gpu:1 \
  --mem 64G \
  --submit
```

**What this does:**
- Groups 8 slides per SLURM task
- Loads patch encoder model ONCE per task (not per slide)
- If using slide-level aggregation, also loads slide encoder model ONCE per task
- **Significant speedup** for all multi-slide processing (not just slide-level aggregation)
- Reduces model loading overhead from N times to N/8 times

**Performance benefits:**
- **Tile/patch extraction**: 2-3x speedup (patch encoder loaded once)
- **With slide-level aggregation**: 6-8x speedup (both encoders loaded once)

**When to use:**
- Processing 2+ slides (always beneficial for batch processing)
- Both with and without slide-level model aggregation
- Have adequate GPU memory (recommend 32-64GB for larger batches)

**Auto-enabled by default** when processing multiple slides. Use `--distributed-slide-batch-size 1` to disable.

See [examples/distributed_batch_processing.md](../../examples/distributed_batch_processing.md) for detailed guide.

## Configuration

### Resource Requirements

```bash
--partition gpu           # SLURM partition
--cpus-per-task 8        # CPUs per task
--mem 32G                # Memory per task
--time 04:00:00          # Time limit (HH:MM:SS)
--gres gpu:1             # Generic resources (GPUs)
--qos high               # Quality of service
```

### Job Arrays

By default, batch submissions use SLURM job arrays for efficiency:

```bash
# Use job array (default)
python scripts/slurm/submit_slurm_jobs.py --csv-manifest manifest.csv --submit

# Submit individual jobs
python scripts/slurm/submit_slurm_jobs.py --csv-manifest manifest.csv --no-array --submit
```

### GPU Support

```bash
--use-gpu                # Enable GPU (default)
--no-gpu                 # Disable GPU
--gres gpu:1             # Request 1 GPU
--gres gpu:v100:2        # Request 2 V100 GPUs
```

## CSV Manifest Format

```csv
slide_id,slide_path
slide_001,/data/slides/slide_001.svs
slide_002,s3://bucket/slides/slide_002.svs
```

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### View specific job
```bash
squeue -j <job_id>
```

### Cancel job
```bash
scancel <job_id>
```

### Job array status
```bash
squeue -j <array_job_id>
```

### View log files
```bash
# Job array logs
tail -f slurm_logs/mussel_array_<job_id>_<array_index>.out
tail -f slurm_logs/mussel_array_<job_id>_<array_index>.err

# Individual job logs
tail -f slurm_logs/slide-001_<job_id>.out
tail -f slurm_logs/slide-001_<job_id>.err
```

## S3 Integration

Process slides from S3 and publish results to S3:

```bash
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest manifest.csv \
  --output-s3-prefix s3://bucket/results/ \
  --aws-access-key-id $AWS_ACCESS_KEY_ID \
  --aws-secret-access-key $AWS_SECRET_ACCESS_KEY \
  --partition gpu \
  --gres gpu:1 \
  --submit
```

**Using custom S3 endpoint (e.g., MinIO, Ceph):**

```bash
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest manifest.csv \
  --output-s3-prefix s3://bucket/results/ \
  --aws-access-key-id $AWS_ACCESS_KEY_ID \
  --aws-secret-access-key $AWS_SECRET_ACCESS_KEY \
  --aws-endpoint-url http://minio.example.com:9000 \
  --partition gpu \
  --gres gpu:1 \
  --submit
```

## Advanced Usage

### Environment Modules

Edit the generated batch script to load required modules:

```bash
# In slurm_job_*.sbatch or slurm_array_job.sbatch
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0
```

### Tissue Filtering

```bash
--classifier-pkl /path/to/classifier.pkl \
--classifier-threshold 0.8
```

### Slide-Level Aggregation

```bash
--aggregation-method mean
```

### Custom Parameters

```bash
--patch-size 512 \
--mpp 0.25 \
--batch-size 128 \
--num-workers 8
```

### Job Dependencies

For complex workflows, use SLURM dependencies:

```bash
# Submit first job
job1=$(sbatch slurm_job_1.sbatch | awk '{print $4}')

# Submit second job dependent on first
sbatch --dependency=afterok:$job1 slurm_job_2.sbatch
```

## Troubleshooting

### Jobs pending (PD)

Check reason:
```bash
squeue -j <job_id> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

Common reasons:
- `Resources` - Waiting for resources
- `Priority` - Lower priority than other jobs
- `QOSMaxJobsPerUserLimit` - User job limit reached

### Jobs failed

Check exit code:
```bash
sacct -j <job_id> --format=JobID,JobName,State,ExitCode
```

View error logs:
```bash
cat slurm_logs/slide-001_<job_id>.err
```

### Check partition info

```bash
sinfo -p <partition_name>
```

### Check account limits

```bash
sacctmgr show assoc user=$USER format=user,account,partition,qos,maxjobs
```

## File Organization

Generated files:
- `slurm_job_<job_name>.sbatch` - Individual job batch scripts
- `slurm_array_job.sbatch` - Job array batch script
- `slurm_array_manifest.csv` - Manifest for job array
- `slurm_logs/` - Job output and error files
- Output features organized by model type (when using `--output-s3-prefix`)

## Comparison with Other Backends

| Feature | SLURM | HTCondor | Azure Batch |
|---------|-------|----------|-------------|
| Best for | HPC clusters | Throughput computing | Cloud bursting |
| Scaling | Manual | Manual/auto | Auto-scale |
| Cost | Fixed infrastructure | Fixed infrastructure | Pay-per-use |
| Setup | Easy | Moderate | Complex |
| Arrays | Native | DAGMan | Task collections |

## Customization

### Modify batch script template

The generated batch scripts use defaults suitable for most clusters. To customize:

1. Generate script without submitting:
   ```bash
   python scripts/slurm/submit_slurm_jobs.py ... --no-submit
   ```

2. Edit generated `.sbatch` file

3. Submit manually:
   ```bash
   sbatch modified_script.sbatch
   ```

### Site-specific configuration

Create a wrapper script with your cluster's defaults:

```bash
#!/bin/bash
python scripts/slurm/submit_slurm_jobs.py \
  --partition gpu-partition \
  --qos normal \
  --time 08:00:00 \
  --mem 64G \
  --gres gpu:a100:1 \
  "$@"
```

## Support

For issues specific to SLURM configuration, consult your cluster administrator.
For Mussel-specific issues, see the main repository documentation.
