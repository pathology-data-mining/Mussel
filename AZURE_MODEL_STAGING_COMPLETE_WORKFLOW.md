# Azure Batch Model Staging Complete Workflow

## TL;DR - How to Specify Prefilter and Classifier

Add to your config YAML file (e.g., `my_config.yaml`):

```yaml
# Prefiltering with CTRANSPATH
prefilter_model_type: CTRANSPATH

# Classifier for filtering (auto-detected from model_dir)
classifier_pkl: classifier.pkl
classifier_threshold: 0.75

# Ensure classifier.pkl is in your local model_cache/ directory
```

Then submit:
```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --config my_config.yaml \
  --csv-manifest manifest.csv \
  --model-dir ./model_cache \
  --mount-azure-files \
  --azure-files-share-name mussel-models \
  --env-file secrets.env
```

**The script will automatically:**
1. Upload `CTRANSPATH/` and `classifier.pkl` from `model_cache/` to Azure Files
2. Mount Azure Files to pool nodes
3. Rsync to persistent local cache `/mnt/batch/tasks/cache/`
4. Find prefilter model and classifier in the cache automatically

---

## Complete Workflow Overview

```
Local models → Azure Files → Azure Files Mount → Persistent Cache → tessellate_extract_features
   (upload)       (stage)         (mount)            (rsync)              (discover)
```

### Step 1: Local Model Directory Structure

```bash
model_cache/
├── OPTIMUS/             # Main models
├── VIRCHOW2/
├── UNI2/
├── TITAN_SLIDE/         # Slide models
├── GIGAPATH_SLIDE/
├── CTRANSPATH/          # ← Prefilter model
└── classifier.pkl       # ← Classifier for filtering
```

### Step 2: Upload to Azure Files (Automatic)

When you run submit script with `--mount-azure-files` and `--model-dir`:

```python
# scripts/azure_batch/submit_batch_jobs.py:1678-1748

if self.azure_files_staging and local_model_dir:
    # Upload each model directory
    for model_dir in [OPTIMUS, VIRCHOW2, UNI2, TITAN_SLIDE, GIGAPATH_SLIDE, CTRANSPATH]:
        azure_files_staging.upload(
            local_path=f"model_cache/{model_dir}",
            remote_path=f"models/{model_dir}"
        )
    
    # Upload classifier.pkl if present
    azure_files_staging.upload(
        local_path="model_cache/classifier.pkl",
        remote_path="models/classifier.pkl"
    )
```

**Result in Azure Files:**
```
mussel-models/  (share)
└── models/
    ├── OPTIMUS/
    ├── VIRCHOW2/
    ├── UNI2/
    ├── TITAN_SLIDE/
    ├── GIGAPATH_SLIDE/
    ├── CTRANSPATH/
    └── classifier.pkl
```

### Step 3: Mount Azure Files (Pool Configuration)

Pool is configured to mount Azure Files at startup:

```python
mount_config = {
    "azure_file_share_configuration": {
        "account_name": "mskpdmgen2",
        "azure_file_url": "https://mskpdmgen2.file.core.windows.net/mussel-models",
        "relative_mount_path": "azfiles"
    }
}
```

**Result on each node:**
```
/mnt/batch/tasks/fsmounts/azfiles/
└── models/
    ├── OPTIMUS/
    ├── VIRCHOW2/
    ├── UNI2/
    ├── TITAN_SLIDE/
    ├── GIGAPATH_SLIDE/
    ├── CTRANSPATH/
    └── classifier.pkl
```

### Step 4: Rsync to Persistent Cache (Start Task)

Pool start task (runs once per node):

```bash
# scripts/azure_batch/submit_batch_jobs.py:483-506

# Executed on node startup
rsync -av --progress \
  /mnt/batch/tasks/fsmounts/azfiles/models/ \
  /mnt/batch/tasks/cache/
```

**Why rsync to local cache?**
- ✅ Faster access (local SSD vs network storage)
- ✅ Persistent across all tasks on node
- ✅ Shared by all tasks (no duplicate downloads)
- ✅ Reliable (no Azure Files throttling)

**Result:**
```
/mnt/batch/tasks/cache/
├── OPTIMUS/
├── VIRCHOW2/
├── UNI2/
├── TITAN_SLIDE/
├── GIGAPATH_SLIDE/
├── CTRANSPATH/          # ← Prefilter model
└── classifier.pkl       # ← Classifier
```

### Step 5: Use in Tasks (Automatic Discovery)

Task execution script points to cache:

```bash
# scripts/azure_batch/run_tessellate_extract_features.sh:224-232

if [ -n "$MODEL_DIR" ]; then
    PERSISTENT_CACHE_DIR="/mnt/batch/tasks/cache"
    CMD_ARGS+=("model_dir=$PERSISTENT_CACHE_DIR")
fi
```

tessellate_extract_features discovers models:

```python
# mussel/cli/tessellate_extract_features.py:75-160

# Prefilter model discovery
def get_model_path_from_dir(model_dir, model_type):
    # Looks for: /mnt/batch/tasks/cache/CTRANSPATH/
    model_subdir = Path(model_dir) / model_type.name
    if model_subdir.exists():
        logger.info(f"✓ Using local model file: {model_type.name} -> {model_subdir}")
        return str(model_subdir)
    return None

# Classifier discovery
def get_classifier_pkl_from_model_dir(model_dir, classifier_pkl):
    if classifier_pkl is not None:
        return classifier_pkl  # Use explicit path
    
    # Auto-detect: /mnt/batch/tasks/cache/classifier.pkl
    classifier_file = Path(model_dir) / "classifier.pkl"
    if classifier_file.exists():
        logger.info(f"✓ Using local classifier file: {classifier_file}")
        return str(classifier_file)
    return None
```

## Configuration Examples

### Example 1: Config File with Prefilter + Classifier

Create `my_config.yaml`:

```yaml
# Prefiltering configuration
prefilter_model_type: CTRANSPATH
classifier_pkl: classifier.pkl  # Auto-detected from model_dir
classifier_threshold: 0.75

# Main models
model_types:
  - OPTIMUS
  - VIRCHOW2
  - UNI2

# Slide models
slide_model_types:
  - TITAN_SLIDE
  - GIGAPATH_SLIDE

# Processing parameters
batch_size: 256
slide_batch_size: 12
slides_per_task: 4
num_workers: 16

# Model-specific batch sizes
model_batch_sizes:
  OPTIMUS: 1024
  VIRCHOW2: 512
  UNI2: 1024
  CTRANSPATH: 512  # Prefilter model batch size

# Azure configuration
azure:
  batch_account_name: "ocra"
  pool_id: "mussel-prod"
  create_pool: true
  vm_size: "Standard_NC24ads_A100_v4"
  mount_azure_files: true
  azure_files_share_name: "mussel-models"
  output_prefix: "azblob://mussel-results/my-experiment"
```

Submit:
```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --config my_config.yaml \
  --csv-manifest external_data_staged_manifest.csv \
  --model-dir ./model_cache \
  --env-file secrets.env \
  --monitor
```

### Example 2: Without Prefilter (No Classifier)

```yaml
# No prefiltering - just extract features
model_types:
  - OPTIMUS
  - VIRCHOW2

# No prefilter_model_type specified
# No classifier_pkl specified

azure:
  # ... same as above
```

### Example 3: Prefilter Only (No Classifier)

```yaml
# Use prefilter model but no classification filtering
prefilter_model_type: CTRANSPATH

# No classifier_pkl specified - no filtering step

model_types:
  - OPTIMUS
  - VIRCHOW2
```

### Example 4: Explicit Classifier Path

```yaml
prefilter_model_type: CTRANSPATH
classifier_pkl: /path/to/specific/model-1727990346535.pkl
classifier_threshold: 0.8  # Custom threshold
```

## What Happens During Execution

### Job Submission
```
✓ Loading config: my_config.yaml
✓ Staging models to Azure Files share: mussel-models
  Uploading models to models/:
    - OPTIMUS/ (2.5GB) ✓
    - VIRCHOW2/ (1.8GB) ✓
    - UNI2/ (1.2GB) ✓
    - TITAN_SLIDE/ (3.1GB) ✓
    - GIGAPATH_SLIDE/ (2.8GB) ✓
    - CTRANSPATH/ (800MB) ✓
    - classifier.pkl (50MB) ✓
✓ Created pool: mussel-prod
✓ Submitted 779 tasks
```

### Node Startup (Once Per Node)
```
[NODE] Starting pool node...
[MODEL_CACHE] Staging models from Azure Files to persistent cache...
[MODEL_CACHE] Source: /mnt/batch/tasks/fsmounts/azfiles/models/
[MODEL_CACHE] Destination: /mnt/batch/tasks/cache/
[MODEL_CACHE] Starting rsync...
[MODEL_CACHE] OPTIMUS/ ✓
[MODEL_CACHE] VIRCHOW2/ ✓
[MODEL_CACHE] UNI2/ ✓
[MODEL_CACHE] TITAN_SLIDE/ ✓
[MODEL_CACHE] GIGAPATH_SLIDE/ ✓
[MODEL_CACHE] CTRANSPATH/ ✓
[MODEL_CACHE] classifier.pkl ✓
[MODEL_CACHE] ✓ Models copied successfully (12.2GB)
[NODE] Ready for tasks
```

### Task Execution
```
[TASK] Starting tessellate-extract-features...
[TASK] Using model_dir: /mnt/batch/tasks/cache
[TASK] Found prefilter model: CTRANSPATH
[TASK]   Path: /mnt/batch/tasks/cache/CTRANSPATH/
[TASK] Found classifier: classifier.pkl
[TASK]   Path: /mnt/batch/tasks/cache/classifier.pkl
[TASK]   Threshold: 0.75
[TASK]
[TASK] Processing slide: slide_001.svs
[TASK] Step 1: Tessellate → 12,456 patches
[TASK] Step 2: Extract prefilter features (CTRANSPATH) → 12,456 features
[TASK] Step 3: Apply classifier filtering
[TASK]   Before: 12,456 patches
[TASK]   After: 8,234 patches (66% retained)
[TASK] Step 4: Extract main model features
[TASK]   OPTIMUS: 8,234 patches ✓
[TASK]   VIRCHOW2: 8,234 patches ✓
[TASK]   UNI2: 8,234 patches ✓
[TASK] Step 5: Extract slide-level features
[TASK]   TITAN_SLIDE: 1 slide ✓
[TASK]   GIGAPATH_SLIDE: 1 slide ✓
[TASK] ✓ Complete!
```

## Verifying the Setup

### 1. Check Local Models Before Submission

```bash
ls -1 model_cache/
# Expected:
# OPTIMUS/
# VIRCHOW2/
# UNI2/
# TITAN_SLIDE/
# GIGAPATH_SLIDE/
# CTRANSPATH/
# classifier.pkl
```

### 2. Check Azure Files After Upload

```bash
az storage file list \
  --account-name mskpdmgen2 \
  --share-name mussel-models \
  --path models \
  --output table

# Expected to see:
# OPTIMUS/
# VIRCHOW2/
# UNI2/
# TITAN_SLIDE/
# GIGAPATH_SLIDE/
# CTRANSPATH/
# classifier.pkl
```

### 3. Check Task Logs for Model Discovery

```bash
# Get logs from a task
python scripts/azure_batch/submit_batch_jobs.py \
  --job-id pr-job-prod-20251204_142812 \
  --get-task-logs batch_1_of_779

# Look for:
# ✓ Using local model file: CTRANSPATH -> /mnt/batch/tasks/cache/CTRANSPATH/
# ✓ Using local classifier file: /mnt/batch/tasks/cache/classifier.pkl
```

## Troubleshooting

### Models Not Found in Cache

**Symptom:** Task logs show "Model CTRANSPATH not found in model_dir, will download from HuggingFace"

**Solution:**
1. Check model was uploaded to Azure Files
2. Check start task logs for rsync errors
3. Verify model directory name matches exactly (case-sensitive)

### Classifier Not Auto-Detected

**Symptom:** No filtering step in logs

**Solution:**
1. Ensure `classifier.pkl` exists in `model_cache/`
2. Verify `classifier_pkl: classifier.pkl` is in config
3. Check classifier was uploaded to Azure Files

### Azure Files Mount Failed

**Symptom:** Start task shows "No models found in Azure Files mount"

**Solution:**
1. Verify `--mount-azure-files` flag is set
2. Check `--azure-files-share-name` matches share name
3. Ensure share exists and has correct permissions

## Summary

✅ **Complete Workflow**: Local → Azure Files → Mount → Cache → Discovery
✅ **Prefilter**: Specify `prefilter_model_type: CTRANSPATH` in config
✅ **Classifier**: Place `classifier.pkl` in `model_cache/`, auto-detected
✅ **Performance**: Local SSD cache for fast access
✅ **Efficiency**: Shared cache across all tasks on node

**Just ensure your `model_cache/` has the models and add the config - everything else is automatic!**
