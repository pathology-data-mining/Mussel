# HTCondor Backend for Mussel Feature Extraction

This directory contains scripts for running `tessellate-extract-features` on HTCondor clusters.

## Overview

HTCondor support enables distributed processing of whole-slide images on HPC clusters using the HTCondor workload management system.

**Key Features:**
- DAGMan workflows for complex job dependencies
- CSV manifest processing for batch submissions
- Automatic retry configuration
- S3 staging and publishing
- Multi-model optimization (filter-tessellate + extract-features)
- GPU scheduling support

## Requirements

- HTCondor installed and configured
- Access to HTCondor submit node
- Python 3.7+
- Mussel environment installed on worker nodes

## Quick Start

### Single Slide

```bash
python scripts/condor/submit_condor_jobs.py \
  --task-id slide-001 \
  --slide-path /data/slides/slide_001.svs \
  --output-h5-path /output/slide_001_features.h5 \
  --output-pt-path /output/slide_001_features.pt \
  --submit
```

### Batch Processing from CSV

```bash
python scripts/condor/submit_condor_jobs.py \
  --csv-manifest manifest.csv \
  --output-dir /output/results/ \
  --submit
```

### Multi-Model Processing

```bash
python scripts/condor/submit_condor_jobs.py \
  --csv-manifest slides.csv \
  --output-s3-prefix s3://bucket/results/ \
  --postfilter-models CTRANSPATH,CLIP,VIRCHOW \
  --submit
```

## Configuration

### Resource Requirements

Specify compute resources via command-line arguments:

```bash
--request-cpus 8          # CPUs per task
--request-memory 32GB     # Memory per task
--request-gpus 1          # GPUs per task (if --use-gpu)
```

### Retry Configuration

```bash
--max-retries 5           # Maximum retry attempts
```

### GPU Support

```bash
--use-gpu                 # Enable GPU (default)
--no-gpu                  # Disable GPU
--request-gpus 1          # Number of GPUs
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
condor_q
```

### View specific job
```bash
condor_q <job_id>
```

### Check job history
```bash
condor_history <job_id>
```

### View log files
```bash
tail -f condor_logs/slide-001.log
tail -f condor_logs/slide-001.out
tail -f condor_logs/slide-001.err
```

## S3 Integration

Process slides from S3 and publish results to S3:

```bash
python scripts/condor/submit_condor_jobs.py \
  --csv-manifest manifest.csv \
  --output-s3-prefix s3://bucket/results/ \
  --aws-access-key-id $AWS_ACCESS_KEY_ID \
  --aws-secret-access-key $AWS_SECRET_ACCESS_KEY \
  --submit
```

## Advanced Usage

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

## Troubleshooting

### Jobs on hold

Check why jobs are on hold:
```bash
condor_q -hold
```

Release held jobs:
```bash
condor_release <job_id>
```

### Failed jobs

Check exit codes:
```bash
condor_history -l <job_id> | grep ExitCode
```

View error logs:
```bash
cat condor_logs/slide-001.err
```

### Check worker node capacity

```bash
condor_status
```

## File Organization

Generated files:
- `condor_submit_<task_id>.sub` - HTCondor submit files
- `condor_logs/` - Job output, error, and log files
- Output features organized by model type (when using `--output-s3-prefix`)

## Comparison with Other Backends

| Feature | HTCondor | SLURM | Azure Batch |
|---------|----------|-------|-------------|
| Best for | Throughput computing | HPC clusters | Cloud bursting |
| Scaling | Manual/auto | Manual | Auto-scale |
| Cost | Fixed infrastructure | Fixed infrastructure | Pay-per-use |
| Setup | Moderate | Easy | Complex |

## Support

For issues specific to HTCondor configuration, consult your cluster administrator.
For Mussel-specific issues, see the main repository documentation.
