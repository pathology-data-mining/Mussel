# Output Directory Structure

## Overview

Output files are now organized by **model name AND file type** for optimal organization with 40K+ slides.

**Key Feature:** Patch encoder and slide encoder features are saved in **separate directories** for slide-level models.

## Directory Structure

```
output_dir/
b��── OPTIMUS/
b��   ├── h5/SLIDE_ID.features.h5
b��   ├── pt/SLIDE_ID.features.pt
b��   └── tile_h5/SLIDE_ID.patch.h5
b��── VIRCHOW2/
b��   ├── h5/SLIDE_ID.features.h5
b��   ├── pt/SLIDE_ID.features.pt
b��   └── tile_h5/SLIDE_ID.patch.h5
b��── UNI2/
b��   ├── h5/SLIDE_ID.features.h5
b��   ├── pt/SLIDE_ID.features.pt
b��   └── tile_h5/SLIDE_ID.patch.h5
b��── TITAN_SLIDE/
b��   ├── h5/SLIDE_ID.features.h5    <- TITAN slide encoder output
b��   └── pt/SLIDE_ID.features.pt    <- TITAN slide encoder output
b��── CONCH1_5/
b��   ├── h5/SLIDE_ID.features.h5    <- CONCH1_5 patch encoder aggregated
b��   ├── pt/SLIDE_ID.features.pt    <- CONCH1_5 patch encoder aggregated
b��   └── tile_h5/SLIDE_ID.patch.h5  <- CONCH1_5 tile features
b��── GIGAPATH_SLIDE/
b��   ├── h5/SLIDE_ID.features.h5    <- GigaPath slide encoder output
b��   └── pt/SLIDE_ID.features.pt    <- GigaPath slide encoder output
b��── GIGAPATH/
    ├── h5/SLIDE_ID.features.h5    <- GigaPath patch encoder aggregated
    ├── pt/SLIDE_ID.features.pt    <- GigaPath patch encoder aggregated
    └── tile_h5/SLIDE_ID.patch.h5  <- GigaPath tile features
```

## Model Directories

### Patch-Level Models (3)
Each saves its own features in 3 subdirectories:

1. **OPTIMUS** (224px patches)
2. **VIRCHOW2** (224px patches)
3. **UNI2** (256px patches)

### Slide-Level Models (2)
Each saves slide-level features AND patch encoder features separately:

4. **TITAN_SLIDE** (512px patches via CONCH1_5)
   - TITAN_SLIDE/ - Slide encoder output
   - CONCH1_5/ - Patch encoder output

5. **GIGAPATH_SLIDE** (256px patches via GIGAPATH)
   - GIGAPATH_SLIDE/ - Slide encoder output
   - GIGAPATH/ - Patch encoder output

### Patch Encoders for Slide Models (2)
These are automatically created when processing slide-level models:

6. **CONCH1_5** - Patch encoder for TITAN_SLIDE
7. **GIGAPATH** - Patch encoder for GIGAPATH_SLIDE

**Total: 7 model directories** (3 patch + 2 slide + 2 patch encoders)

## File Types

### h5/ - Aggregated Features (HDF5)
Contains slide-level aggregated features in HDF5 format.
- **Patch models:** Mean/max aggregated from patch-level features
- **Slide models:** Slide encoder output (model-based aggregation)
- **Patch encoders:** Mean aggregated patches (for slide model's patch encoder)
- **File pattern:** `{slide_id}.features.h5`

### pt/ - Aggregated Features (PyTorch)
Contains slide-level aggregated features in PyTorch format.
- Same content as h5/, different format
- **File pattern:** `{slide_id}.features.pt`

### tile_h5/ - Patch-Level Features (HDF5)
Contains intermediate patch/tile-level features from the patch encoder.
- **File pattern:** `{slide_id}.patch.h5`
- Created for patch models and patch encoders
- NOT created for slide encoders (TITAN_SLIDE, GIGAPATH_SLIDE)

## Feature Processing Flow

### Patch Models (OPTIMUS, VIRCHOW2, UNI2)
```
Slide → Tessellation → Patch Encoder → tile_h5/
                                     ↓
                              Aggregation (mean)
                                     ↓
                               h5/ + pt/
```

### Slide Models (TITAN_SLIDE, GIGAPATH_SLIDE)
```
Slide → Tessellation → Patch Encoder → PATCH_ENCODER/tile_h5/
                                     ↓
                              Aggregation (mean) → PATCH_ENCODER/h5/ + pt/
                                     ↓
                          Slide Encoder (model-based) → SLIDE_ENCODER/h5/ + pt/
```

Example for TITAN_SLIDE:
- CONCH1_5/tile_h5/SLIDE_ID.patch.h5 (tiles)
- CONCH1_5/h5/SLIDE_ID.features.h5 (aggregated patches)
- TITAN_SLIDE/h5/SLIDE_ID.features.h5 (slide encoding)

## Total Files for 40K Slides

### 7 Models × 40K Slides:

**Patch Models (3):**
- OPTIMUS: 120K files (40K × 3 types)
- VIRCHOW2: 120K files (40K × 3 types)
- UNI2: 120K files (40K × 3 types)

**Slide Models (2):**
- TITAN_SLIDE: 80K files (40K × 2 types)
- GIGAPATH_SLIDE: 80K files (40K × 2 types)

**Patch Encoders for Slide Models (2):**
- CONCH1_5: 120K files (40K × 3 types)
- GIGAPATH: 120K files (40K × 3 types)

**Total: 760,000 files**
- 560K aggregated (h5 + pt) = 40K × 7 models × 2 formats
- 200K tile-level (tile_h5) = 40K × 5 encoders (3 patch + 2 patch encoders)

## Patch Encoder Compatibility

Slide models require specific patch encoders:
- **TITAN_SLIDE** → **CONCH1_5** (512px patches)
- **GIGAPATH_SLIDE** → **GIGAPATH** (256px patches)

Source: `mussel/models/model_factory.py`
```python
SLIDE_ENCODER_COMPATIBILITY = {
    ModelType.GIGAPATH_SLIDE: ModelType.GIGAPATH,
    ModelType.TITAN_SLIDE: ModelType.CONCH1_5,
}
```

## Benefits

1. **Clear separation** - Patch encoder vs slide encoder features
2. **Independent access** - Can use CONCH1_5 or GIGAPATH features directly
3. **Organized by type** - h5, pt, tile_h5 subdirectories
4. **Scalable** - Works with millions of files
5. **Reusable** - Patch encoder features can be used for other purposes

## Example Use Cases

### Use CONCH1_5 patch features independently
```python
# Load CONCH1_5 aggregated features (without TITAN_SLIDE encoding)
features = load_h5("output/CONCH1_5/h5/SLIDE_ID.features.h5")
```

### Use TITAN_SLIDE features (slide encoding)
```python
# Load TITAN_SLIDE slide-level encoding
features = load_h5("output/TITAN_SLIDE/h5/SLIDE_ID.features.h5")
```

### Access tile-level features
```python
# Load CONCH1_5 tile features
tiles = load_h5("output/CONCH1_5/tile_h5/SLIDE_ID.patch.h5")
```

## Implementation

### Multi-Model Processing
The `_main_batch_multi_model()` function processes all models:

```python
# Patch models - single output directory
_main_batch(cfg_copy)  # OPTIMUS/h5/, pt/, tile_h5/

# Slide models - separate patch and slide directories
_main_batch(cfg_copy, patch_output_dir="CONCH1_5")
# Creates: CONCH1_5/h5/, pt/, tile_h5/ + TITAN_SLIDE/h5/, pt/
```

### Processing Phases
For slide models, three phases:

1. **Phase 1:** Tessellation (all models)
2. **Phase 2:** Extract patch features → `CONCH1_5/tile_h5/`
3. **Phase 2b:** Aggregate patch features → `CONCH1_5/h5/`, `CONCH1_5/pt/`
4. **Phase 3:** Slide-level encoding → `TITAN_SLIDE/h5/`, `TITAN_SLIDE/pt/`

## Migration

### Path Changes
Old structure (before this update):
```
TITAN_SLIDE/
  ├── h5/SLIDE_ID.features.h5
  ├── pt/SLIDE_ID.features.pt
  └── tile_h5/SLIDE_ID.patch.h5  <- CONCH1_5 features
```

New structure:
```
TITAN_SLIDE/
  ├── h5/SLIDE_ID.features.h5
  └── pt/SLIDE_ID.features.pt
CONCH1_5/
  ├── h5/SLIDE_ID.features.h5
  ├── pt/SLIDE_ID.features.pt
  └── tile_h5/SLIDE_ID.patch.h5
```

### Code Updates
Update paths to access patch encoder features:
- Old: `TITAN_SLIDE/tile_h5/SLIDE_ID.patch.h5`
- New: `CONCH1_5/tile_h5/SLIDE_ID.patch.h5`

## Backward Compatibility

b��️ **Breaking Change**: This is a breaking change to the output structure.

Existing downstream tools will need to be updated to use the new paths, especially for accessing patch encoder features for slide-level models.

---

**Recent Commits:**
- `e9a0932` - Fix output filename formatting (added dots)
- `716c4c3` - Organize output files into subdirectories by file type
- `e2cae3e` - Fix documentation: slide models DO save tile_h5 files
- `f5a474f` - Correct patch encoder documentation for slide models
- `38fe944` - Separate patch encoder outputs from slide encoder outputs ✨
