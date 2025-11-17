# Output Directory Structure

## Overview

Output files are now organized by **both model type AND file type** for better organization, especially with 40K+ slides.

## Directory Structure

```
output_dir/
b��── OPTIMUS/
b��   ├── h5/
b��   │   ├── SLIDE_ID.features.h5
b��   │   └── ...
b��   ├── pt/
b��   │   ├── SLIDE_ID.features.pt
b��   │   └── ...
b��   └── tile_h5/
b��       ├── SLIDE_ID.patch.h5
b��       └── ...
b��── VIRCHOW2/
b��   ├── h5/
b��   ├── pt/
b��   └── tile_h5/
b��── UNI2/
b��   ├── h5/
b��   ├── pt/
b��   └── tile_h5/
b��── TITAN_SLIDE/
b��   ├── h5/
b��   ├── pt/
b��   └── tile_h5/
b��── GIGAPATH_SLIDE/
    ├── h5/
    ├── pt/
    └── tile_h5/
```

## File Types

### h5/ - Aggregated Features (HDF5)
Contains slide-level aggregated features in HDF5 format.
- **Patch models:** Aggregated from patch-level features
- **Slide models:** Slide-level encoding from aggregated patches
- **File pattern:** `{slide_id}.features.h5`

### pt/ - Aggregated Features (PyTorch)
Contains slide-level aggregated features in PyTorch format.
- Same content as h5/, different format
- **File pattern:** `{slide_id}.features.pt`

### tile_h5/ - Patch-Level Features (HDF5)
Contains intermediate patch/tile-level features from the patch encoder.
- **File pattern:** `{slide_id}.patch.h5`
- Created for ALL models (both patch and slide-level)
- **Patch models (OPTIMUS, VIRCHOW2, UNI2):** Patch encoder outputs
- **Slide models (TITAN_SLIDE, GIGAPATH_SLIDE):** Patch encoder outputs before slide-level aggregation

## Changes From Previous Version

### Before (Flat Structure):
```
output_dir/
b��── OPTIMUS/
    ├── 1000326.features.h5
    ├── 1000326.features.pt
    ├── 1000326.patch.h5
    ├── 1000331.features.h5
    ├── 1000331.features.pt
    ├── 1000331.patch.h5
    └── ... (40,000+ files mixed together)
```

### After (Organized Structure):
```
output_dir/
b��── OPTIMUS/
    ├── h5/
    │   ├── 1000326.features.h5
    │   ├── 1000331.features.h5
    │   └── ... (40,000 h5 files)
    ├── pt/
    │   ├── 1000326.features.pt
    │   ├── 1000331.features.pt
    │   └── ... (40,000 pt files)
    └── tile_h5/
        ├── 1000326.patch.h5
        ├── 1000331.patch.h5
        └── ... (40,000 tile files)
```

## Benefits

1. **Cleaner organization** with 40K+ slides
2. **Easier to navigate** - files grouped by type
3. **Better for downstream tools** - can target specific file types
4. **Clearer separation** between aggregated features and tile-level features
5. **Scalable** - works well with millions of files

## Examples

### 40K Slides with 5 Models

Total directory structure:
```
output_dir/
b��── OPTIMUS/         (h5/, pt/, tile_h5/)
b��── VIRCHOW2/        (h5/, pt/, tile_h5/)
b��── UNI2/            (h5/, pt/, tile_h5/)
b��── TITAN_SLIDE/     (h5/, pt/, tile_h5/)
b��── GIGAPATH_SLIDE/  (h5/, pt/, tile_h5/)
```

Total files:
- 40K × 5 models × 2 formats = 400,000 aggregated feature files (h5 + pt)
- 40K × 5 models = 200,000 tile-level feature files (tile_h5)
- **Total: 600,000 files** organized into 15 subdirectories

### With Remote Storage (Azure Blob)

When using Azure Blob Storage (`azblob://`), paths like:
```
azblob://account/container/output/OPTIMUS/h5/1000326.features.h5
azblob://account/container/output/OPTIMUS/pt/1000326.features.pt
azblob://account/container/output/OPTIMUS/tile_h5/1000326.patch.h5
```

The subdirectory structure is created automatically by the blob path.

## Implementation Details

Changes in `mussel/cli/tessellate_extract_features.py`:

```python
# H5 files go in h5/ subdirectory
result['output_h5_path'] = _safe_path_join(output_dir_str, "h5", f"{slide_id}.{cfg.output_h5_suffix}")

# PT files go in pt/ subdirectory  
result['output_pt_path'] = _safe_path_join(output_dir_str, "pt", f"{slide_id}.{cfg.output_pt_suffix}")

# Tile H5 files go in tile_h5/ subdirectory (for ALL models)
intermediate_h5_paths = [
    _safe_path_join(output_dir_str, "tile_h5", f"{r['slide_id']}.patch.h5") 
    for r in slide_results
]
```

Directories are created automatically with `mkdir(parents=True, exist_ok=True)`.

## Two-Step Processing

All models use two-step processing:

1. **Step 1: Extract patch features**
   - Patch encoder extracts features from tiles
   - Saved to `tile_h5/SLIDE_ID.patch.h5`

2. **Step 2: Aggregate to slide level**
   - **Patch models:** Simple aggregation (mean, max, etc.)
   - **Slide models:** Model-based aggregation (TITAN_SLIDE, GIGAPATH_SLIDE)
   - Saved to `h5/SLIDE_ID.features.h5` and `pt/SLIDE_ID.features.pt`

### Patch Encoder Compatibility

Slide models require specific patch encoders:
- **TITAN_SLIDE** → Uses **CONCH1_5** patch encoder (512px patches)
- **GIGAPATH_SLIDE** → Uses **GIGAPATH** patch encoder (256px patches)

The patch encoder outputs are saved in `tile_h5/` before slide-level aggregation.
   - Saved to `h5/SLIDE_ID.features.h5` and `pt/SLIDE_ID.features.pt`

The only exception is when `aggregation_method = "identity"`, which writes patch features directly to output (not used in multi-model mode).

## Migration

Existing code using the old flat structure can be updated by:

1. **Path changes:**
   - Old: `{model}/SLIDE_ID.features.h5`
   - New: `{model}/h5/SLIDE_ID.features.h5`

2. **Glob patterns:**
   - Old: `OPTIMUS/*.features.h5`
   - New: `OPTIMUS/h5/*.features.h5`

3. **Remote paths:**
   - Old: `azblob://account/container/OPTIMUS/SLIDE_ID.features.h5`
   - New: `azblob://account/container/OPTIMUS/h5/SLIDE_ID.features.h5`

## Backward Compatibility

b��️ **Breaking Change**: This is a breaking change to the output structure.

Existing downstream tools expecting the flat structure will need to be updated to use the new subdirectory paths.

---

**Commits:**
- `e9a0932` - Fix output filename formatting (added dots)
- `716c4c3` - Organize output files into subdirectories by file type
