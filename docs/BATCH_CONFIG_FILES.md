# Configuration Files for Batch Processing

## Overview

All distributed batch submission scripts (Azure Batch, HTCondor, SLURM) now support configuration files in both **JSON** and **YAML** formats. Configuration files provide a structured way to define tasks with default parameters, making it easier to manage large-scale batch processing jobs.

You can use configuration files in two ways:
1. **Standalone**: Config file with both task definitions and parameters
2. **With CSV**: CSV manifest for slide paths + config file for parameters only

## Features

- **Dual Format Support**: Use either JSON or YAML format (detected automatically by file extension)
- **Default Parameters**: Define common parameters once in a `defaults` section
- **Task-Specific Overrides**: Override defaults for individual tasks as needed
- **CSV + Config Combination**: Use CSV for slide manifest and YAML/JSON for parameters
- **Command-Line Overrides**: All command-line arguments override config file parameters for maximum flexibility
- **Security**: Sensitive fields (credentials, tokens) are automatically filtered from output manifests
- **Configuration Tracking**: Non-sensitive configuration is saved to result manifests for reproducibility

## Usage Modes

### Mode 1: Standalone Config File (with tasks)

Use a single config file that contains both task definitions and parameters.

```bash
# Azure Batch
python scripts/azure_batch/submit_batch_jobs.py \
  --config-file batch_config.yaml \
  --pool-id my-pool --job-id my-job

# HTCondor
python scripts/condor/submit_condor_jobs.py \
  --task-config batch_config.yaml --submit

# SLURM
python scripts/slurm/submit_slurm_jobs.py \
  --task-config batch_config.yaml --submit
```

### Mode 2: CSV Manifest + Config File (for parameters)

Use a simple CSV for slide IDs and paths, and a config file for all processing parameters.

**Benefits**:
- Simple slide manifest (just ID and path)
- All parameters in one place
- Easy to update parameters without touching slide list
- Cleaner separation of concerns
- Backend-specific parameters (SLURM partition, HTCondor resources, etc.) can be included

```bash
# Azure Batch
python scripts/azure_batch/submit_batch_jobs.py \
  --csv-manifest slides.csv \
  --config params.yaml \
  --pool-id my-pool --job-id my-job

# HTCondor
python scripts/condor/submit_condor_jobs.py \
  --csv-manifest slides.csv \
  --config params.yaml \
  --submit

# SLURM
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest slides.csv \
  --config params.yaml \
  --submit
```

**CSV format** (slides.csv):
```csv
slide_id,slide_path
slide_001,s3://bucket/slides/slide_001.svs
slide_002,s3://bucket/slides/slide_002.svs
```

**Config format with standardized resources** (params.yaml):
```yaml
# General processing parameters
prefilter_model_type: CTRANSPATH
batch_size: 64
num_workers: 4
use_gpu: true

# Standardized resource requirements (works across all backends)
resources:
  cpus: 8
  memory: 32G
  gpus: 1

# AWS S3 configuration (optional)
aws:
  region: us-east-1
  endpoint_url: https://s3.example.com  # Optional: for S3-compatible storage

# Backend-specific parameters (optional, for backend-only settings)
slurm:
  partition: gpu
  time: "04:00:00"

condor:
  max_retries: 3
```

### Command-Line Parameter Overrides

**All command-line arguments override configuration file parameters.** This allows you to:
- Define baseline parameters in a config file
- Override specific parameters for different runs without modifying the config file
- Test different parameter combinations quickly

**Priority order** (lowest to highest):
1. Config file defaults
2. Command-line arguments (override config)
3. Task-specific config (in standalone mode only)

**Example**: Override batch size and partition:
```bash
# Config file has batch_size: 64 and partition: gpu
# Command-line overrides both
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest slides.csv \
  --config params.yaml \
  --batch-size 128 \
  --partition cpu \
  --submit

# Result: batch_size=128 (not 64), partition=cpu (not gpu)
```

**Common override use cases**:
```bash
# Override resource requirements for a larger job
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest slides.csv \
  --config params.yaml \
  --cpus-per-task 16 \
  --mem 64G \
  --submit

# Override model type for testing
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest slides.csv \
  --config params.yaml \
  --prefilter-model-type UNI \
  --submit

# Override AWS region
python scripts/slurm/submit_slurm_jobs.py \
  --csv-manifest slides.csv \
  --config params.yaml \
  --aws-region us-west-2 \
  --submit
```

**Note**: Sensitive parameters (credentials, tokens) should always be passed via command-line or environment variables for security, not stored in config files.

## Configuration File Formats

### Mode 1: Standalone Config (with tasks)

A configuration file has two main sections:

1. **`defaults`**: Common parameters applied to all tasks
2. **`tasks`**: List of individual task definitions

### YAML Example

```yaml
# batch_config.yaml
defaults:
  prefilter_model_type: CTRANSPATH
  batch_size: 64
  num_workers: 4
  patch_size: 256
  mpp: 0.5
  use_gpu: true
  segment_threshold: 0
  classifier_threshold: 0.75
  max_retry_count: 3

tasks:
  - task_id: slide_001
    slide_path: s3://my-bucket/slides/slide_001.svs
    output_h5_path: s3://my-bucket/results/CTRANSPATH/h5/slide_001_features.h5
    output_pt_path: s3://my-bucket/results/CTRANSPATH/pt/slide_001_features.pt
    
  - task_id: slide_002
    slide_path: s3://my-bucket/slides/slide_002.svs
    output_h5_path: s3://my-bucket/results/CTRANSPATH/h5/slide_002_features.h5
    output_pt_path: s3://my-bucket/results/CTRANSPATH/pt/slide_002_features.pt
    batch_size: 128  # Override default for this task
    
  - task_id: slide_003
    slide_path: /local/path/slide_003.svs
    output_h5_path: /output/slide_003_features.h5
    output_pt_path: /output/slide_003_features.pt
    postfilter_model_type: VIRCHOW  # Use different model
```

### Mode 2: Parameters-Only Config (for use with CSV)

When using CSV manifest mode, the config file only needs parameters (no `tasks` section).

**YAML Example** (params.yaml):
```yaml
# Processing parameters only (no tasks)
prefilter_model_type: CTRANSPATH
batch_size: 64
num_workers: 4
patch_size: 256
mpp: 0.5
use_gpu: true
segment_threshold: 0
classifier_threshold: 0.75
max_retry_count: 3

# Optional: aggregation settings
aggregation_method: identity

# Optional: slide-level aggregation
# aggregation_method: model
# slide_model_type: GIGAPATH_SLIDE
# slide_batch_size: 8
```

**JSON Example** (params.json):
```json
{
  "prefilter_model_type": "CTRANSPATH",
  "batch_size": 64,
  "num_workers": 4,
  "use_gpu": true,
  "segment_threshold": 0
}
```

**CSV Example** (slides.csv):
```csv
slide_id,slide_path
slide_001,s3://my-bucket/slides/slide_001.svs
slide_002,s3://my-bucket/slides/slide_002.svs
slide_003,/local/path/slide_003.svs
```

### JSON Example

```json
{
  "defaults": {
    "prefilter_model_type": "CTRANSPATH",
    "batch_size": 64,
    "num_workers": 4,
    "patch_size": 256,
    "mpp": 0.5,
    "use_gpu": true
  },
  "tasks": [
    {
      "task_id": "slide_001",
      "slide_path": "s3://my-bucket/slides/slide_001.svs",
      "output_h5_path": "s3://my-bucket/results/h5/slide_001_features.h5",
      "output_pt_path": "s3://my-bucket/results/pt/slide_001_features.pt"
    }
  ]
}
```

## Usage

### Azure Batch

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name <account> \
  --batch-account-key <key> \
  --batch-account-url <url> \
  --pool-id my-pool \
  --create-pool \
  --job-id my-job \
  --create-job \
  --config-file batch_config.yaml \
  --aws-access-key-id <key> \
  --aws-secret-access-key <secret> \
  --generate-manifest results_manifest.csv
```

### HTCondor

```bash
python scripts/condor/submit_condor_jobs.py \
  --config-file batch_config.yaml \
  --aws-access-key-id <key> \
  --aws-secret-access-key <secret> \
  --submit
```

### SLURM

```bash
python scripts/slurm/submit_slurm_jobs.py \
  --config-file batch_config.yaml \
  --partition gpu \
  --gres gpu:1 \
  --aws-access-key-id <key> \
  --aws-secret-access-key <secret> \
  --submit
```

## Configuration Parameters

### Required Fields (per task)

- `task_id` or `job_name`: Unique identifier for the task
- `slide_path`: Path to the whole-slide image (local path or s3://)
- `output_h5_path`: Path for output HDF5 feature file
- `output_pt_path`: Path for output PyTorch feature file

### Common Optional Parameters

- `prefilter_model_type`: Model for tile-level feature extraction (default: "CTRANSPATH")
- `postfilter_model_type`: Additional postfilter model
- `aggregation_method`: "identity", "mean", "max", or "model"
- `slide_model_type`: Model for slide-level aggregation (e.g., "GIGAPATH_SLIDE")
- `batch_size`: Batch size for tile processing (default: 64)
- `num_workers`: Number of data loading workers (default: 4)
- `patch_size`: Size of patches to extract (default: 256)
- `mpp`: Microns per pixel (default: 0.5)
- `use_gpu`: Enable GPU processing (default: true)
- `segment_threshold`: Tissue segmentation threshold (default: 0)
- `classifier_pkl`: Path to classifier pickle file for filtering
- `classifier_threshold`: Classifier threshold (default: 0.75)
- `max_retry_count`: Maximum retry attempts (default: 3)

### AWS S3 Parameters

AWS parameters should be grouped under the `aws:` section in your configuration file. The config loader automatically maps these to the expected `aws_*` parameter names.

**AWS configuration structure:**
```yaml
aws:
  region: us-east-1
  endpoint_url: https://s3.example.com  # Optional: for S3-compatible storage
  # access_key_id and secret_access_key should be provided via command-line for security
```

**Parameters:**
- `region`: AWS region (default: "us-east-1") - mapped to `aws_region`
- `endpoint_url`: Custom S3 endpoint URL for S3-compatible storage like MinIO, Wasabi - mapped to `aws_endpoint_url`
- `access_key_id`: AWS access key (not saved to manifest) - mapped to `aws_access_key_id`
- `secret_access_key`: AWS secret key (not saved to manifest) - mapped to `aws_secret_access_key`

**Note:** For security, AWS credentials (`access_key_id`, `secret_access_key`) should typically be provided via command-line arguments or environment variables rather than in the config file.

### Standardized Resource Parameters

Resource requirements (CPU, memory, GPU) should be specified in the top-level `resources:` section using standardized parameter names. The config loader automatically maps these to backend-specific parameter names:

**Standardized resource parameters:**
- `cpus`: Number of CPUs (integer)
- `memory`: Memory amount with unit (e.g., "32G", "64GB")
- `gpus`: Number of GPUs (integer)

**Backend mapping:**
- **SLURM**: `cpus` → `cpus_per_task`, `memory` → `mem`, `gpus` → `gres` (as `gpu:N`)
- **HTCondor**: `cpus` → `request_cpus`, `memory` → `request_memory`, `gpus` → `request_gpus`
- **Azure**: Uses values as-is

**Example with standardized resources for all backends:**
```yaml
prefilter_model_type: CTRANSPATH
batch_size: 64

resources:
  cpus: 8
  memory: 32G
  gpus: 1

# AWS S3 configuration (optional, for S3-compatible storage)
aws:
  region: us-east-1
  endpoint_url: https://s3.example.com

# SLURM-specific parameters
slurm:
  partition: gpu
  time: "04:00:00"

# HTCondor-specific parameters
condor:
  max_retries: 3

# Azure Batch-specific parameters
azure:
  container_image: "mskmind/mussel:latest-torch-gpu"
  storage_account_name: "mystorageaccount"
```

### Backend-Specific Parameters

Backend-specific parameters (non-resource settings) should be grouped under their respective backend sections (`slurm:`, `condor:`, or `azure:`). This keeps configuration organized and allows a single config file to work with multiple backends.

#### SLURM Parameters

SLURM-specific parameters (resources are defined in the top-level `resources:` section):

- `partition`: SLURM partition (default: "batch")
- `time`: Time limit in HH:MM:SS format (default: "02:00:00")
- `qos`: Quality of service

**Example SLURM config:**
```yaml
prefilter_model_type: CTRANSPATH
batch_size: 64

resources:
  cpus: 8
  memory: 32G
  gpus: 1

slurm:
  partition: gpu
  time: "04:00:00"
  qos: high
```

#### HTCondor Parameters

HTCondor-specific parameters (resources are defined in the top-level `resources:` section):

- `max_retries`: Maximum retry attempts (default: 3)

**Example HTCondor config:**
```yaml
prefilter_model_type: CTRANSPATH
batch_size: 64

resources:
  cpus: 8
  memory: 32GB
  gpus: 1

condor:
  max_retries: 3
```

### Azure-Specific Parameters

When using Azure Batch, include these parameters under the `azure:` or `azure_batch:` section:

- `storage_account_name`: Azure storage account
- `storage_account_key`: Azure storage key (not saved to manifest)
- `container_image`: Docker container image (default: "mskmind/mussel:latest-torch-gpu")

**Example Azure config:**
```yaml
prefilter_model_type: CTRANSPATH
batch_size: 64

resources:
  cpus: 8
  memory: 32G

azure:
  container_image: "mskmind/mussel:latest-torch-gpu"
```

## Security and Privacy

### Sensitive Fields

The following fields are automatically **excluded** from result manifests:

- `aws_access_key_id`
- `aws_secret_access_key`
- `hf_token`
- `batch_account_key`
- `storage_account_key`
- `azure_files_share_name`

All other configuration parameters are saved to the manifest for reproducibility.

### Best Practices

1. **Store credentials separately**: Pass credentials via command-line arguments or environment variables
2. **Use config files for non-sensitive data**: Task definitions, model parameters, paths
3. **Version control config files**: Track changes to processing parameters over time
4. **Review manifests**: Verify that no sensitive data leaked into result manifests

## Output Manifests

When using `--generate-manifest` (Azure Batch), the output CSV will include:

- Task metadata (task_id, state, exit_code)
- Output file paths (slide_path, output_h5_path, output_pt_path)
- Configuration parameters (prefixed with `config_`)
- Model type and file type information

Example manifest output:

```csv
task_id,state,exit_code,slide_path,output_h5_path,config_batch_size,config_prefilter_model_type
slide_001,completed,0,s3://bucket/slide_001.svs,s3://bucket/results/slide_001.h5,64,CTRANSPATH
slide_002,completed,0,s3://bucket/slide_002.svs,s3://bucket/results/slide_002.h5,128,CTRANSPATH
```

## Advantages Over CSV Manifests

| Feature | CSV Manifest | Config File |
|---------|--------------|-------------|
| Default parameters | ❌ Not supported | ✅ Supported |
| Per-task overrides | ❌ Not supported | ✅ Supported |
| Human-readable | ⚠️ Limited | ✅ Excellent (YAML) |
| Comments | ❌ Not supported | ✅ Supported (YAML) |
| Complex nested data | ❌ Limited | ✅ Supported |
| Configuration tracking | ❌ Not tracked | ✅ Tracked in manifests |

## When to Use

### Use Configuration Files When:

- Processing tasks with many shared parameters
- Need to override specific parameters for individual tasks
- Want to track configuration alongside results
- Prefer human-readable, self-documenting format
- Working with complex nested configurations

### Use CSV Manifests When:

- Simple slide ID and path mapping is sufficient
- All tasks use identical parameters (passed via command-line)
- Need to generate manifests programmatically from simple data
- Integrating with existing CSV-based workflows

## Examples

See the `examples/` directory for complete examples:

- `examples/batch_config_example.yaml`: YAML configuration example
- `examples/batch_config_example.json`: JSON configuration example

## Testing

To validate your configuration file:

```bash
# Test with dry-run (doesn't submit)
python scripts/condor/submit_condor_jobs.py \
  --task-config my_config.yaml

# Or use Python to load and inspect
python3 -c "
from scripts.common.config_loader import load_config
config = load_config('my_config.yaml')
print('Tasks:', len(config['tasks']))
print('Defaults:', config['defaults'])
"
```

## Troubleshooting

### "Unsupported configuration file format" error

**Cause**: File extension is not .json, .yaml, or .yml

**Solution**: Rename your file with the correct extension:
```bash
mv config.txt config.yaml
```

### "PyYAML is required to load YAML configuration files"

**Cause**: PyYAML library not installed

**Solution**: Install PyYAML:
```bash
pip install pyyaml
```

### Tasks not inheriting defaults

**Cause**: Task-specific parameters override defaults completely

**Solution**: Only specify parameters you want to override in task definitions. Other parameters will inherit from defaults.

### Sensitive data in manifest

**Cause**: Using custom field names not in the sensitive fields list

**Solution**: Sensitive fields are automatically filtered. If you're using custom field names for secrets, avoid including them in config or add them to `SENSITIVE_FIELDS` in `config_loader.py`.

## Migration from JSON

Converting existing JSON configurations to YAML:

```bash
# Install yq if not available
pip install yq

# Convert JSON to YAML
cat config.json | yq -y '.' > config.yaml
```

Or manually convert using Python:

```python
import json
import yaml

with open('config.json') as f:
    config = json.load(f)

with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
```
