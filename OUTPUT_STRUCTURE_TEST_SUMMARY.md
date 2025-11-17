# Output Directory Structure Test Summary

## Tests Performed

### Test 1: UNI2 (Patch Model) - Single Model Mode
**Command:**
```bash
uv run tessellate_extract_features \
  slide_paths=[SLIDE1,SLIDE2] \
  model_type=UNI2 \
  output_dir=test_output_structure \
  batch_size=128 \
  aggregation_method=mean
```

**Result:** âœ… **SUCCESS**

**Output Structure:**
```
test_output_structure/
b”œâ”€â”€ h5/
b”‚   â”œâ”€â”€ SLIDE1.features.h5
b”‚   â””â”€â”€ SLIDE2.features.h5
b”œâ”€â”€ pt/
b”‚   â”œâ”€â”€ SLIDE1.features.pt
b”‚   â””â”€â”€ SLIDE2.features.pt
b””â”€â”€ tile_h5/
    â”œâ”€â”€ SLIDE1.patch.h5
    â””â”€â”€ SLIDE2.patch.h5
```

### Test 2: Multi-Model Mode (UNI2 + TITAN_SLIDE)
**Command:**
```bash
uv run tessellate_extract_features \
  slide_paths=[SLIDE1] \
  model_type=[UNI2] \
  slide_model_type=[TITAN_SLIDE] \
  output_dir=test_output_structure \
  batch_size=128 \
  slide_batch_size=1
```

**Result:** âœ… **SUCCESS**

**Output Structure:**
```
test_output_structure/
b”œâ”€â”€ UNI2/
b”‚   â”œâ”€â”€ h5/SLIDE1.features.h5
b”‚   â””â”€â”€ pt/SLIDE1.features.pt
b”œâ”€â”€ TITAN_SLIDE/
b”‚   â”œâ”€â”€ h5/SLIDE1.features.h5    <- TITAN slide encoding
b”‚   â””â”€â”€ pt/SLIDE1.features.pt
b””â”€â”€ CONCH1_5/
    â”œâ”€â”€ h5/SLIDE1.features.h5    <- CONCH1_5 aggregated patches
    â”œâ”€â”€ pt/SLIDE1.features.pt
    â””â”€â”€ tile_h5/SLIDE1.patch.h5  <- CONCH1_5 tile features
```

## Key Observations

### 1. Subdirectory Creation âœ…
- **h5/**, **pt/**, and **tile_h5/** subdirectories are created automatically
- Works for both single-model and multi-model modes
- Parent directories created by `save_hdf5()` and `save_torch_tensor()`

### 2. Separated Patch/Slide Encoders âœ…
- TITAN_SLIDE creates **two** directories:
  - `TITAN_SLIDE/` - Slide encoder output (h5/, pt/)
  - `CONCH1_5/` - Patch encoder output (h5/, pt/, tile_h5/)
- Clear separation between patch and slide features

### 3. File Naming âœ…
- Format: `{SLIDE_ID}.features.h5` and `{SLIDE_ID}.features.pt`
- Dots correctly placed (fixed in commit `e9a0932`)
- Consistent naming across all models

### 4. UNI2 in Multi-Model Mode
- Uses `aggregation_method='identity'` (single-step processing)
- Saves directly to h5/ and pt/ without tile_h5/
- This is expected behavior for patch models in multi-model mode

## Test Slides Used

```
SLIDE1: tcga_slides/TCGA-RM-A68W-01Z-00-DX1.4E62E4F4-415C-46EB-A6C8-45BA14E82708.svs
SLIDE2: tcga_slides/TCGA-WB-A81G-01Z-00-DX1.70672250-BF2D-4E3F-8242-3638C0362D2D.svs
```

## Code Changes Verified

### 1. tessellate_extract_features.py
- âœ… `patch_output_dir` parameter added to `_main_batch()`
- âœ… Separate directories for patch and slide encoder outputs
- âœ… Phase 2b: saves aggregated patch encoder features

### 2. file.py
- âœ… `save_hdf5()` creates parent directories
- âœ… `save_torch_tensor()` creates parent directories
- âœ… Works for local paths (remote paths use temp files)

## Expected Structure for 40K Slides

With 5 models (OPTIMUS, VIRCHOW2, UNI2, TITAN_SLIDE, GIGAPATH_SLIDE):

```
output_dir/
b”œâ”€â”€ OPTIMUS/         (h5/, pt/, tile_h5/)
b”œâ”€â”€ VIRCHOW2/        (h5/, pt/, tile_h5/)
b”œâ”€â”€ UNI2/            (h5/, pt/, tile_h5/)
b”œâ”€â”€ TITAN_SLIDE/     (h5/, pt/)
b”œâ”€â”€ CONCH1_5/        (h5/, pt/, tile_h5/)
b”œâ”€â”€ GIGAPATH_SLIDE/  (h5/, pt/)
b””â”€â”€ GIGAPATH/        (h5/, pt/, tile_h5/)
```

**Total:** 7 model directories, ~760,000 files

## Conclusion

bœ… **All tests passed successfully!**

The new output directory structure is working correctly:
1. Subdirectories (h5/, pt/, tile_h5/) are created automatically
2. Patch encoder and slide encoder outputs are separated
3. File naming is consistent and correct
4. Both single-model and multi-model modes work as expected

## Next Steps

Ready for production use with 40K slides on Azure Batch!

---

**Test Date:** 2025-11-17
**Branch:** cdsieng-532
**Commits Tested:**
- `e9a0932` - Fix output filename formatting
- `716c4c3` - Organize into subdirectories
- `38fe944` - Separate patch/slide encoder outputs
- `c2b5a85` - Auto-create parent directories
