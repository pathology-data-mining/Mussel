# Staging Container Changes - Summary

## What Changed

### 1. Renamed Parameter: `--azure-blob-container` → `--staging-container`

**Old name**: `--azure-blob-container`
**New name**: `--staging-container`

**Reason**: More intuitive name that reflects its purpose as a unified staging location.

### 2. Unified Staging Location

**Before**: Models and slides staged to different locations
- Slides: `azure_blob_container/slides/`
- Models: `output_prefix/models/`

**After**: Both staged to same container
- Slides: `staging_container/slides/`
- Models: `staging_container/models/` (new default!)

## Benefits

### 1. **Clearer Organization**
```
staging_container/
b��── slides/              # Input WSI files
b��   ├── slide001.svs
b��   └── slide002.svs
b��── models/              # Model weights (NEW!)
    ├── uni.pth
    ├── virchow2.pth
    └── gigapath_slide.pth

output_prefix/           # Results only
b��── slide001_uni.h5
b��── slide002_uni.h5
```

### 2. **No Model Duplication**
- Models stored **once** in `staging_container/models/`
- Reused across all runs
- Results stored separately in `output_prefix/`

### 3. **Cost Savings**
**Before (10 runs)**:
```
output1/models/  (12 GB)
output2/models/  (12 GB)
output3/models/  (12 GB)
...
= 120 GB of duplicate models!
```

**After (10 runs)**:
```
staging_container/models/  (12 GB, shared)
output1/                   (results only)
output2/                   (results only)
...
= 12 GB total for models!
```

**Savings**: ~90% storage reduction for models!

## Usage

### New Parameter (Recommended)
```bash
python scripts/azure_batch/submit_batch_jobs.py \
    --env-file secrets.env \
    --staging-container mussel-staging \
    --stage-to-azure-blob \
    --config config.yaml \
    --csv-manifest slides.csv
```

### Old Parameter (Still Works)
```bash
python scripts/azure_batch/submit_batch_jobs.py \
    --env-file secrets.env \
    --azure-blob-container mussel-staging \  # Deprecated but still works
    --stage-to-azure-blob \
    --config config.yaml \
    --csv-manifest slides.csv
```

## Configuration

### In YAML Config
```yaml
# Old name (still works)
azure:
  azure_blob_container: "mussel-staging"

# New name (recommended)
azure:
  staging_container: "mussel-staging"
```

## Model Staging Logic

### Priority Order:
1. **`--model-s3-prefix`** (highest priority)
   - If specified, use this exact location
   
2. **`--staging-container`** (new default)
   - If specified, use `staging_container/models/`
   
3. **`output_prefix/models/`** (legacy fallback)
   - Only used if no staging container specified

### Examples

#### Example 1: Default Behavior (New)
```yaml
azure:
  staging_container: "mussel-staging"
  output_prefix: "azblob://mussel-results/run1/"
```

**Result:**
- Models: `azblob://account/mussel-staging/models/`
- Slides: `azblob://account/mussel-staging/slides/`
- Results: `azblob://mussel-results/run1/`

#### Example 2: Custom Model Location
```yaml
azure:
  staging_container: "mussel-staging"  # For slides
  model_s3_prefix: "azblob://mussel-models/v2/"  # Custom models location
  output_prefix: "azblob://mussel-results/run1/"
```

**Result:**
- Models: `azblob://mussel-models/v2/`
- Slides: `azblob://account/mussel-staging/slides/`
- Results: `azblob://mussel-results/run1/`

#### Example 3: Legacy (No Staging Container)
```yaml
azure:
  output_prefix: "azblob://mussel-results/run1/"
```

**Result:**
- Models: `azblob://mussel-results/run1/models/` (old behavior)
- Slides: Not staged (must be accessible directly)
- Results: `azblob://mussel-results/run1/`

## Migration Guide

### For Existing Scripts

**Option 1**: Update to new parameter name (recommended)
```diff
- --azure-blob-container mussel-staging
+ --staging-container mussel-staging
```

**Option 2**: Do nothing (backward compatible)
```bash
# Old parameter still works
--azure-blob-container mussel-staging
```

### For Existing YAML Configs

**Option 1**: Update key name (recommended)
```diff
  azure:
-   azure_blob_container: "mussel-staging"
+   staging_container: "mussel-staging"
```

**Option 2**: Do nothing (backward compatible)
```yaml
azure:
  azure_blob_container: "mussel-staging"  # Still works
```

## Breaking Changes

**None!** The changes are fully backward compatible:
- ✅ Old parameter `--azure-blob-container` still works
- ✅ Old config key `azure_blob_container` still works
- ✅ Model staging logic improved but gracefully falls back

## What You Should Do

### For New Projects
Use the new parameter:
```bash
--staging-container mussel-staging
```

### For Existing Projects
Either:
1. **Update parameter names** (recommended for clarity)
2. **Keep using old names** (works fine)

## Testing

All changes tested and verified:

```bash
# Test new parameter
python submit_batch_jobs.py --staging-container mussel-staging

# Test old parameter (backward compatibility)
python submit_batch_jobs.py --azure-blob-container mussel-staging

# Both work identically!
```

## Summary

### What Changed:
1. ✅ Renamed `--azure-blob-container` → `--staging-container`
2. ✅ Models now staged to `staging_container/models/` by default
3. ✅ Unified staging location for slides and models
4. ✅ Full backward compatibility maintained

### Benefits:
1. ✅ Clearer parameter naming
2. ✅ No duplicate models across runs
3. ✅ ~90% storage cost reduction
4. ✅ Better organization

### Migration:
- **Required**: None (backward compatible)
- **Recommended**: Update to `--staging-container` for clarity

### Example Command:
```bash
python scripts/azure_batch/submit_batch_jobs.py \
    --env-file secrets.env \
    --staging-container mussel-staging \
    --stage-to-azure-blob \
    --config run_paper_revisions.yaml \
    --csv-manifest slides.csv
```

This gives you:
- Slides in `mussel-staging/slides/`
- Models in `mussel-staging/models/` (shared!)
- Results in `output_prefix/` (per run)
