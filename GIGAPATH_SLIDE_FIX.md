# GIGAPATH_SLIDE Fix

## Problem

GIGAPATH_SLIDE model was not saved properly in `model_cache/`, causing the following error:

```
ERROR | Error during batch aggregation: We have no connection or you passed local_files_only, so force_download is not an accepted option.
```

This resulted in:
- âœ“ OPTIMUS, VIRCHOW2, UNI2 patches extracted successfully
- âœ“ TITAN_SLIDE aggregation completed  
- âœ— GIGAPATH_SLIDE aggregation failed
- Empty `local_prod_run/GIGAPATH_SLIDE/` directory

## Root Cause

The model loading code looks for GIGAPATH_SLIDE in these locations (in order):
1. `model_cache/GIGAPATH_SLIDE/` (directory)
2. `model_cache/gigapath_slide/` (directory)
3. `model_cache/GIGAPATH_SLIDE.pth` (file)
4. `model_cache/gigapath_slide.pth` (file)

But only `model_cache/gigapath.pth` (the patch encoder) existed, not the slide-level model directory.

## Solution

### Quick Fix (Symlink to HF Cache)

Since GIGAPATH_SLIDE was already downloaded to the HuggingFace cache, create a symlink:

```bash
cd model_cache
ln -s .cache/huggingface/hub/models--prov-gigapath--prov-gigapath/snapshots/eba85dd46097c3eedfcc2a3a9a930baecb6bcc19 GIGAPATH_SLIDE
```

Verify:
```bash
ls -la model_cache/ | grep GIGAPATH
# Should show:
# lrwxrwxrwx ... GIGAPATH_SLIDE -> .cache/huggingface/hub/models--prov-gigapath--prov-gigapath/snapshots/eba85dd46097c3eedfcc2a3a9a930baecb6bcc19
```

### Proper Fix (Save Model Properly)

For future setups, save GIGAPATH_SLIDE properly using the `save_model` CLI:

```bash
# Using apptainer
make apptainer-save-slide-models MODEL_DIR=model_cache

# Or specifically for GIGAPATH_SLIDE
make apptainer-save-models MODELS="GIGAPATH_SLIDE" MODEL_DIR=model_cache

# Or using uv directly
uv run save_model model_types=[GIGAPATH_SLIDE] model_dir=model_cache
```

This will create `model_cache/GIGAPATH_SLIDE/` directory with:
- `config.json`
- `pytorch_model.bin`
- `.ready` marker file

## Verification

After applying the fix, check that GIGAPATH_SLIDE loads correctly:

```bash
# Test with apptainer
apptainer exec --nv \
  --bind $(pwd):/workspace \
  --env MODEL_DIR=/workspace/model_cache \
  mussel_fastattn.sif \
  python -c "from mussel.cli.tessellate_extract_features import get_model_path_from_dir; from mussel.models import ModelType; print(get_model_path_from_dir('/workspace/model_cache', ModelType.GIGAPATH_SLIDE))"
```

Expected output:
```
bœ“ Using local model file: GIGAPATH_SLIDE -> /workspace/model_cache/GIGAPATH_SLIDE
/workspace/model_cache/GIGAPATH_SLIDE
```

## Files Modified

- `model_cache/GIGAPATH_SLIDE` (symlink created)

## Related Issues

This same issue could affect:
- TITAN_SLIDE (also a HuggingFace model directory)
- Any future slide-level models that use HuggingFace transformers

Always verify slide-level models are saved as directories with `save_model` CLI, not just patch encoders (.pth files).

## Testing

After applying the fix, rerun the failed batch:

```bash
# Example: Resubmit batch 4 that failed
sbatch slurm_job_batch_4_of_244.sbatch
```

Check logs for:
```
bœ“ Using local model file: GIGAPATH_SLIDE -> /workspace/model_cache/GIGAPATH_SLIDE
...
=== Phase 3: Batch aggregating 8 slides (aggregation_method=model) ===
Slide model: GIGAPATH_SLIDE
...
bœ“ Saved slide features to /workspace/output/GIGAPATH_SLIDE/...
```

And verify output:
```bash
ls local_prod_run/GIGAPATH_SLIDE/h5/
ls local_prod_run/GIGAPATH_SLIDE/pt/
```

Should contain `.h5` and `.pt` files for each processed slide.
