# Using --env-file to Load Credentials

## Quick Start

Use the `--env-file` parameter to load credentials from a file:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
    --env-file secrets.env \
    --config run_paper_revisions.yaml \
    --csv-manifest slides.csv
```

## What It Does

The `--env-file` parameter loads environment variables from a file (like `secrets.env`) before processing other arguments. This replaces the need to manually `source` the file in your shell.

## Format

The env file should contain `export VAR=value` or `VAR=value` lines:

```bash
# Azure Batch credentials
export AZURE_BATCH_ACCOUNT_NAME="ocra"
export AZURE_BATCH_ACCOUNT_KEY="your-key-here"
export AZURE_BATCH_ACCOUNT_URL="https://ocra.eastus2.batch.azure.com"

# Azure Storage credentials
export AZURE_STORAGE_ACCOUNT="mystorageaccount"
export AZURE_STORAGE_KEY="your-storage-key"

# AWS credentials (optional)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_ENDPOINT_URL="http://pmindecs.mskcc.org:9020"

# HuggingFace token (optional)
export HF_TOKEN="hf_xxxxxxxxxxxxx"

# Container registry (optional)
export AZURE_CONTAINER_REGISTRY_SERVER="myregistry.azurecr.io"
```

## Priority Order

Credentials are loaded in this order (first wins):

1. **Command-line arguments** (highest priority)
2. **--env-file** parameter
3. **Environment variables** (already set in shell)
4. **Config file** (YAML)

Example:
```bash
# The CLI argument overrides secrets.env
python submit_batch_jobs.py \
    --env-file secrets.env \
    --batch-account-name override-account \
    ...
```

## Advantages Over `source`

### Before (manual sourcing):
```bash
source secrets.env
python scripts/azure_batch/submit_batch_jobs.py --config myconfig.yaml ...
```

### After (automatic loading):
```bash
python scripts/azure_batch/submit_batch_jobs.py \
    --env-file secrets.env \
    --config myconfig.yaml \
    ...
```

**Benefits:**
- ✅ Works in scripts without needing `source`
- ✅ No shell state pollution
- ✅ Explicit about which env file is used
- ✅ Works in CI/CD pipelines
- ✅ Can use different env files per run

## Usage Examples

### Basic Usage
```bash
python scripts/azure_batch/submit_batch_jobs.py \
    --env-file secrets.env \
    --config run_paper_revisions.yaml \
    --csv-manifest slides.csv
```

### With Different Env Files
```bash
# Development
python submit_batch_jobs.py --env-file secrets.dev.env ...

# Production
python submit_batch_jobs.py --env-file secrets.prod.env ...
```

### In Scripts
```bash
#!/bin/bash
# No need to source secrets.env!

python scripts/azure_batch/submit_batch_jobs.py \
    --env-file secrets.env \
    --config run_paper_revisions.yaml \
    --csv-manifest slides.csv
```

### In CI/CD
```yaml
# GitHub Actions example
- name: Submit Azure Batch job
  run: |
    python scripts/azure_batch/submit_batch_jobs.py \
        --env-file ${{ secrets.ENV_FILE_PATH }} \
        --config config.yaml \
        --csv-manifest slides.csv
```

## Error Handling

### File Not Found
```bash
$ python submit_batch_jobs.py --env-file nonexistent.env
ERROR: Environment file not found: nonexistent.env
```

### Invalid Format
The script tolerates various formats:
- `export VAR=value`
- `VAR=value`
- `VAR="value"`
- `VAR='value'`
- Comments with `#`
- Empty lines

## Testing

Test if the env file is loaded correctly:

```bash
# Should show loading message and loaded count
python scripts/azure_batch/submit_batch_jobs.py --env-file secrets.env

# Should show error
python scripts/azure_batch/submit_batch_jobs.py --env-file nonexistent.env
```

## Security

1. **Never commit secrets.env** to git (already in .gitignore)
2. **Use restrictive permissions**: `chmod 600 secrets.env`
3. **Use different files for dev/prod**: `secrets.dev.env`, `secrets.prod.env`
4. **Store production secrets securely**: Use secret management services

## Comparison

| Method | Pros | Cons |
|--------|------|------|
| **`--env-file`** | ✅ Explicit<br>✅ Script-friendly<br>✅ No shell pollution | - |
| **`source secrets.env`** | ✅ Traditional | ❌ Requires shell<br>❌ Pollutes environment |
| **CLI arguments** | ✅ Most explicit | b�� Verbose<br>❌ Secrets in command history |
| **Config YAML** | ✅ Clean | ❌ Not for secrets |

## Recommendation

**Use `--env-file` for all automated scripts and production runs.**

```bash
python scripts/azure_batch/submit_batch_jobs.py \
    --env-file secrets.env \
    --config run_paper_revisions.yaml \
    --csv-manifest slides.csv
```

Simple, explicit, and safe!
