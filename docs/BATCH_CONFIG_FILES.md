# Configuration Files for Batch Processing

## Overview

All distributed batch submission scripts (Azure Batch, HTCondor, SLURM) now support configuration files in both **JSON** and **YAML** formats. Configuration files provide a structured way to define tasks with default parameters, making it easier to manage large-scale batch processing jobs.

## Features

- **Dual Format Support**: Use either JSON or YAML format (detected automatically by file extension)
- **Default Parameters**: Define common parameters once in a `defaults` section
- **Task-Specific Overrides**: Override defaults for individual tasks as needed
- **Security**: Sensitive fields (credentials, tokens) are automatically filtered from output manifests
- **Configuration Tracking**: Non-sensitive configuration is saved to result manifests for reproducibility

## Configuration File Format

### Structure

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

### S3 Parameters

- `aws_access_key_id`: AWS access key (not saved to manifest)
- `aws_secret_access_key`: AWS secret key (not saved to manifest)
- `aws_region`: AWS region (default: "us-east-1")
- `aws_endpoint_url`: Custom S3 endpoint URL

### Azure-Specific Parameters

- `storage_account_name`: Azure storage account
- `storage_account_key`: Azure storage key (not saved to manifest)
- `container_image`: Docker container image (default: "mskmind/mussel:latest-torch-gpu")

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
  --config-file my_config.yaml

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
