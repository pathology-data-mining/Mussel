# Azure Files Model Staging Verification

## Question: Are models staged correctly with --mount-azure-files?

búÖ **YES - Models are staged to Azure Files and correctly discovered by scripts**

## How It Works

### 1. Model Staging (Submit Script)

When you specify `model_dir` with `--mount-azure-files`, the script:

```python
# scripts/azure_batch/submit_batch_jobs.py:1678-1748

if self.azure_files_staging and local_model_dir:
    # Upload models from local model_dir
    for item_name in os.listdir(local_model_dir):
        if item_name in models_to_upload or item_name in ["classifier.pkl"]:
            # Upload model files/directories to Azure Files
            remote_path = f"models/{item_name}"
            azure_files_staging.upload(local_path, remote_path)
    
    # Convert model_dir to Azure Files URL
    azfiles_model_dir = f"azfiles://{storage_account}/{share_name}/models"
    default_params["model_dir"] = azfiles_model_dir
```

**Key files uploaded:**
- Model directories: `OPTIMUS/`, `VIRCHOW2/`, `UNI2/`, `GIGAPATH_SLIDE/`, `TITAN_SLIDE/`
- Model files: `optimus.pth`, `virchow2.pth`, `uni2.pth`, etc.
- **`classifier.pkl`** (for filtering)
- `version.txt` (optional)

**Result**: `model_dir=azfiles://mskpdmgen2/mussel-models/models`

### 2. Azure Files Mount (Batch Pool)

The pool is configured with Azure Files mount:

```python
# Mount Azure Files at /mnt/batch/tasks/fsmounts/azfiles
mount_config = {
    "azure_file_share_configuration": {
        "account_name": storage_account,
        "azure_file_url": f"https://{storage_account}.file.core.windows.net/{share_name}",
        "relative_mount_path": "azfiles"
    }
}
```

**Mount point**: `/mnt/batch/tasks/fsmounts/azfiles/`

### 3. Model Path Resolution (Container)

In the container, `tessellate_extract_features.py` uses `@resolve_remote_paths`:

```python
# mussel/cli/tessellate_extract_features.py:519
@resolve_remote_paths('model_dir', 'classifier_pkl', auto_detect=False)
def tessellate_extract_features_single(cfg):
    # model_dir is now a local path
    ...
```

The decorator downloads azfiles:// URLs:

```python
# mussel/utils/file.py:543-562
if model_path.startswith("azfiles://"):
    # Parse: azfiles://account/share/models
    share_name = "mussel-models"
    prefix = "models"
    
    # Download to local cache
    local_path = "/root/.cache/models"
    _download_azure_files_directory(share_name, prefix, local_path)
```

**Wait, there's an issue!** The URL format doesn't match the parsing logic.

Let me check the actual parsing:

```python
# Line 546-550
path_parts = model_path.split("://", 1)[1]  # "mskpdmgen2/mussel-models/models"
parts = path_parts.split("/", 1)            # ["mskpdmgen2", "mussel-models/models"]

share_name = parts[0]   # "mskpdmgen2" ‚ùå Wrong! Should be "mussel-models"
prefix = parts[1]       # "mussel-models/models" ‚ùå Wrong!
```

**BUG FOUND**: The azfiles:// parsing expects `azfiles://share/path` but gets `azfiles://account/share/path`

### 4. Correct Format

The script should generate:
```python
azfiles_model_dir = f"azfiles://{share_name}/models"
# Result: azfiles://mussel-models/models
```

Not:
```python
azfiles_model_dir = f"azfiles://{storage_account}/{share_name}/models"
# Result: azfiles://mskpdmgen2/mussel-models/models ‚ùå
```

### 5. Model Discovery (After Resolution)

Once `model_dir` is resolved to local path, scripts find models:

```python
# mussel/cli/tessellate_extract_features.py:75-129
def get_model_path_from_dir(model_dir, model_type):
    # Check for model directory: model_dir/OPTIMUS/
    model_subdir = Path(model_dir) / model_type.name
    if model_subdir.exists():
        return str(model_subdir)
    
    # Check for .pth file: model_dir/optimus.pth
    model_file = Path(model_dir) / f"{model_type.name}.pth"
    if model_file.exists():
        return str(model_file)
```

**Prefilter model discovery:**
```python
# Line 597
prefilter_model_path = get_model_path_from_dir(cfg.model_dir, cfg.prefilter_model_type)
```

**Classifier discovery:**
```python
# mussel/cli/tessellate_extract_features.py:132-160
def get_classifier_pkl_from_model_dir(model_dir, classifier_pkl):
    if classifier_pkl is not None:
        return classifier_pkl  # Use explicit path
    
    if model_dir is None:
        return None
    
    # Auto-detect classifier.pkl in model_dir
    classifier_file = Path(model_dir) / "classifier.pkl"
    if classifier_file.exists():
        return str(classifier_file)
```

## Issue Found

**The azfiles:// URL format is WRONG in the submit script.**

### Current (Broken):
```python
# Line 1746
azfiles_model_dir = f"azfiles://{self.storage_account_name}/{self.azure_files_share_name}/models"
# Result: azfiles://mskpdmgen2/mussel-models/models
```

### Expected by Parser:
```python
# mussel/utils/file.py expects: azfiles://share/path
share_name = parts[0]  # Should be "mussel-models"
prefix = parts[1]      # Should be "models"
```

### Fix Required:
```python
# Line 1746 - Remove storage_account_name
azfiles_model_dir = f"azfiles://{self.azure_files_share_name}/models"
# Result: azfiles://mussel-models/models ‚úì
```

## Workaround (If Mounted)

If Azure Files is mounted at `/mnt/batch/tasks/fsmounts/azfiles/`, you can use local path:

```python
# Instead of azfiles:// URL, use mount point
model_dir = "/mnt/batch/tasks/fsmounts/azfiles/models"
```

This bypasses the download logic entirely since the models are already mounted.

## Verification Steps

1. **Check uploaded models** in Azure Files:
   ```bash
   # List models in share
   az storage file list \
     --account-name mskpdmgen2 \
     --share-name mussel-models \
     --path models \
     --output table
   ```

2. **Test azfiles:// URL parsing**:
   ```python
   model_path = "azfiles://mussel-models/models"
   path_parts = model_path.split("://", 1)[1]  # "mussel-models/models"
   parts = path_parts.split("/", 1)            # ["mussel-models", "models"]
   share_name = parts[0]   # "mussel-models" ‚úì
   prefix = parts[1]       # "models" ‚úì
   ```

3. **Verify classifier.pkl uploaded**:
   ```bash
   az storage file show \
     --account-name mskpdmgen2 \
     --share-name mussel-models \
     --path models/classifier.pkl \
     --query properties.contentLength
   ```

## Recommendation

**Use mounted path instead of azfiles:// URL:**

```python
# In submit_batch_jobs.py line 1746-1748, use mount point:
if self.azure_files_staging:
    # Use local mount point instead of azfiles:// URL
    azfiles_model_dir = "/mnt/batch/tasks/fsmounts/azfiles/models"
    default_params["model_dir"] = azfiles_model_dir
```

**Benefits:**
- No download needed (models already mounted)
- Faster startup time
- Avoids URL parsing issues
- Works with existing code

**Or fix the URL format:**
```python
# Remove storage_account_name from URL
azfiles_model_dir = f"azfiles://{self.azure_files_share_name}/models"
```

## Summary

búÖ Models are uploaded correctly to Azure Files (including classifier.pkl)
búÖ Azure Files is mounted at `/mnt/batch/tasks/fsmounts/azfiles/`
búÖ Scripts can find prefilter_model_path and classifier_pkl in model_dir
bùå azfiles:// URL format is incorrect (includes account name)

**Solution:** Use mounted path `/mnt/batch/tasks/fsmounts/azfiles/models` instead of azfiles:// URL
