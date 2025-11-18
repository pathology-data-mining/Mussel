# Model Staging Fix - Verified

## Problem
Models were still being staged to `output_prefix/models/` even when `staging_container` was specified.

## Root Cause
The condition was checking `if args.output_prefix and upload_models_to_s3:` first, which always evaluated to true when output_prefix was set, bypassing the staging_container logic.

## Solution
Reordered the logic to check for staging locations in the correct priority order:

```python
# OLD (broken)
if args.output_prefix and upload_models_to_s3:
    if args.model_s3_prefix:
        ...
    elif args.staging_container:
        ...
    else:
        # output_prefix/models/

# NEW (fixed)
if upload_models_to_s3:
    if args.model_s3_prefix:
        ...
    elif args.staging_container and args.storage_account_name:
        ...
    elif args.output_prefix:
        ...
```

## Priority Order (Fixed)

1. **`--model-s3-prefix`** (Highest Priority)
   - Explicit custom location
   - Example: `azblob://custom-models/v1/`

2. **`--staging-container`** (New Default) âœ…
   - Unified staging location
   - Example: `azblob://account/mussel-staging/models/`

3. **`output_prefix/models/`** (Legacy Fallback)
   - Only if no staging container specified
   - Example: `azblob://mussel-results/run1/models/`

4. **Local cache only** (No Remote)
   - If no remote storage configured at all

## Verification

### Test 1: With staging_container (NEW DEFAULT)
```bash
python scripts/azure_batch/submit_batch_jobs.py \
    --staging-container mussel-staging \
    --output-prefix azblob://mussel-results/run1/ \
    ...
```

**Output:**
```
[Model Staging] Using staging container: azblob://mskpdmgen2/mussel-staging/models/
```

bœ… **Models staged to:** `azblob://mskpdmgen2/mussel-staging/models/`

### Test 2: Without staging_container (LEGACY)
```bash
python scripts/azure_batch/submit_batch_jobs.py \
    --output-prefix azblob://mussel-results/run1/ \
    ...
```

**Output:**
```
[Model Staging] Using output prefix (legacy): azblob://mussel-results/run1/models/
```

bœ… **Models staged to:** `azblob://mussel-results/run1/models/`

### Test 3: With custom model_s3_prefix (HIGHEST)
```bash
python scripts/azure_batch/submit_batch_jobs.py \
    --model-s3-prefix azblob://custom-models/v1/ \
    --staging-container mussel-staging \
    ...
```

**Output:**
```
[Model Staging] Using custom model prefix: azblob://custom-models/v1/
```

bœ… **Models staged to:** `azblob://custom-models/v1/`

## Updated Behavior

### Scenario 1: Typical Production Setup
```yaml
azure:
  staging_container: "mussel-staging"
  output_prefix: "azblob://mussel-results/run1/"
```

**Result:**
- âœ… Slides: `azblob://account/mussel-staging/slides/`
- âœ… Models: `azblob://account/mussel-staging/models/` â† **FIXED!**
- âœ… Results: `azblob://mussel-results/run1/`

### Scenario 2: Legacy Setup (No Staging Container)
```yaml
azure:
  output_prefix: "azblob://mussel-results/run1/"
```

**Result:**
- âŒ Slides: Not staged (must be accessible directly)
- âœ… Models: `azblob://mussel-results/run1/models/`
- âœ… Results: `azblob://mussel-results/run1/`

### Scenario 3: Custom Model Location
```yaml
azure:
  staging_container: "mussel-staging"
  model_s3_prefix: "azblob://shared-models/v1/"
  output_prefix: "azblob://mussel-results/run1/"
```

**Result:**
- bœ… Slides: `azblob://account/mussel-staging/slides/`
- âœ… Models: `azblob://shared-models/v1/` â† Custom location
- âœ… Results: `azblob://mussel-results/run1/`

## Benefits

### 1. No Model Duplication
**Before (broken):**
```
run1/models/  (12 GB)
run2/models/  (12 GB)
run3/models/  (12 GB)
= 36 GB
```

**After (fixed):**
```
mussel-staging/models/  (12 GB, shared)
run1/                   (results only)
run2/                   (results only)
= 12 GB total
```

**Savings: 66% for 3 runs, 90% for 10 runs!**

### 2. Cleaner Organization
```
staging_container/
b”œâ”€â”€ slides/              # Input files
b””â”€â”€ models/              # Model weights (shared)

output_prefix/           # Results only (per run)
b”œâ”€â”€ slide001_uni.h5
b””â”€â”€ slide002_uni.h5
```

### 3. Faster Subsequent Runs
- First run: Downloads models to staging container
- Subsequent runs: Reuse models from staging container
- No re-upload needed!

## Testing Commands

```bash
# Test with staging container
python scripts/azure_batch/submit_batch_jobs.py \
    --env-file secrets.env \
    --staging-container mussel-staging \
    --config config.yaml \
    --csv-manifest slides.csv \
    2>&1 | grep "Model Staging"

# Should output:
# [Model Staging] Using staging container: azblob://account/mussel-staging/models/
```

## Summary

bœ… **Fixed**: Models now correctly staged to `staging_container/models/` when specified

bœ… **Backward Compatible**: Legacy behavior still works if no staging container

bœ… **Priority Correct**: model_s3_prefix > staging_container > output_prefix

bœ… **Verified**: All test cases pass

## Migration

**For existing configs**, just add `staging_container`:

```diff
  azure:
+   staging_container: "mussel-staging"
    output_prefix: "azblob://mussel-results/run1/"
```

**Result**: Models will now go to `mussel-staging/models/` instead of `mussel-results/run1/models/`!
