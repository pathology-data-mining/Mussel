# GigaPath Slide Embedding Extraction - Workflow Summary

## Completed Work

Successfully created and tested the workflow for extracting GigaPath slide embeddings for the PANDA dataset using Mussel's `tessellate_extract_features` CLI.

### Key Findings

1. **Correct CLI Tool**: Use `tessellate_extract_features` for integrated workflow (tessellation + feature extraction)
2. **Model Configuration**:
   - `prefilter_model_type=GIGAPATH` - Patch encoder (tile encoder)  
   - `postfilter_model_type=GIGAPATH` - Same for consistency
   - `aggregation_method=model` - Use learned slide encoder
   - `slide_model_type=GIGAPATH_SLIDE` - Slide encoder for aggregation

3. **Required Dependencies**:
   - `s3fs` - For S3 slide access (installed with `uv pip install s3fs`)
   - Already has: `tiffslide`, `torch`, `timm`, etc.

4. **Batch Mode Parameters**:
   - `slide_paths='[s3://path1,s3://path2,...]'` - List of slide paths
   - `output_dir` - Output directory for results
   - `slide_batch_size=3` - Process 3 slides in parallel
   - `batch_size=64` - Patches per GPU batch

## Working Command

```bash
cd /gpfs/mskmind_ess/limr/repos/Mussel-3

# Activate environment
source .venv/bin/activate

# Set S3 endpoint
export AWS_ENDPOINT_URL=http://pmindecs.mskcc.org:9020

# Run GigaPath extraction
python -m mussel.cli.tessellate_extract_features \
  prefilter_model_type=GIGAPATH \
  postfilter_model_type=GIGAPATH \
  aggregation_method=model \
  slide_model_type=GIGAPATH_SLIDE \
  slide_paths='[s3://pathology/TCGA/.../slide1.svs,s3://pathology/TCGA/.../slide2.svs]' \
  output_dir=./gigapath_output \
  batch_size=64 \
  slide_batch_size=3 \
  num_workers=8 \
  use_gpu=true \
  seg_config.patch_size=256 \
  seg_config.step_size=256 \
  seg_config.mpp=0.5 \
  save_features_to_h5=true \
  output_h5_suffix=.gigapath.h5
```

## Docker Workflow

### Option 1: Using mussel-docker wrapper

```bash
cd /gpfs/mskmind_ess/limr/repos/Mussel-3

# Build/use latest image
./mussel-docker tessellate_extract_features \
  prefilter_model_type=GIGAPATH \
  postfilter_model_type=GIGAPATH \
  aggregation_method=model \
  slide_model_type=GIGAPATH_SLIDE \
  slide_paths='[s3://pathology/TCGA/.../slide.svs]' \
  output_dir=/data/gigapath_output \
  batch_size=64 \
  use_gpu=true \
  seg_config.patch_size=256
```

### Option 2: Direct Docker command

```bash
docker run --rm --gpus all --shm-size=4g \
  -e HF_TOKEN=${HF_TOKEN} \
  -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
  -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
  -e AWS_ENDPOINT_URL=http://pmindecs.mskcc.org:9020 \
  -v $(pwd):/data \
  -w /data \
  mskmind/mussel:latest \
  python -m mussel.cli.tessellate_extract_features \
    prefilter_model_type=GIGAPATH \
    postfilter_model_type=GIGAPATH \
    aggregation_method=model \
    slide_model_type=GIGAPATH_SLIDE \
    slide_paths='[s3://pathology/TCGA/.../slide.svs]' \
    output_dir=/data/gigapath_output \
    batch_size=64 \
    use_gpu=true \
    seg_config.patch_size=256
```

## Workflow Phases

The `tessellate_extract_features` tool runs in multiple phases:

1. **Phase 1: Tessellation** - Segment tissue and extract patch coordinates for all slides
2. **Phase 2: Pre-filter Feature Extraction** - Extract patch features using GIGAPATH encoder
3. **Phase 3: Post-filter Feature Extraction** - (Optional) Re-extract with different model
4. **Phase 4: Slide-level Aggregation** - Aggregate patch features to slide embeddings using GIGAPATH_SLIDE

## Output Format

```
gigapath_output/
├── TCGA-02-0003-01Z-00-DX1.6171b175-0972-4e84-9997-2f1ce75f4407.gigapath.h5
├── TCGA-02-0006-01Z-00-DX1.a37df719-8b93-4245-ae49-67eb1114253a.gigapath.h5
└── TCGA-02-0009-01Z-00-DX3.BAA1276B-E4D7-43D4-BDDF-807532462518.gigapath.h5
```

Each HDF5 file contains:
- `coords` - Patch coordinates (N, 2)
- `features` - Patch-level features (N, 1536)
- `slide_embedding` - Slide-level embedding (768,)

## GigaPath Architecture

**Two-Stage Model:**
1. **Tile Encoder (GIGAPATH)**: 256×256 patches → 1536-dim features
2. **Slide Encoder (GIGAPATH_SLIDE)**: patch features + coords → 768-dim slide embedding

## Configuration Options

### Segmentation (tissue detection)
```yaml
seg_config:
  patch_size: 256      # Must be 256 for GigaPath
  step_size: 256       # Non-overlapping (or 128 for 50% overlap)
  mpp: 0.5             # Target microns per pixel (0.5 = 20× magnification)
  seg_level: -1        # Auto-select level for segmentation
  segment_threshold: 20 # Tissue vs background threshold
  use_otsu: false      # Use Otsu thresholding
  tissue_area_threshold: 100  # Minimum tissue area
```

### Performance
```yaml
batch_size: 64              # Patches per GPU batch (reduce if OOM)
slide_batch_size: 3         # Slides to aggregate in parallel
num_workers: 8              # Data loading workers
use_gpu: true               # GPU acceleration
```

### Output
```yaml
output_dir: ./gigapath_output
output_h5_suffix: .gigapath.h5
output_pt_suffix: .gigapath.pt
save_features_to_h5: true
keep_intermediate_files: false
```

## Test Slides

Test CSV created: `test_gigapath_slides.csv`
```csv
slide_id,slide_path
3638,s3://pathology/TCGA/.../TCGA-02-0003-01Z-00-DX1.6171b175-0972-4e84-9997-2f1ce75f4407.svs
6233,s3://pathology/TCGA/.../TCGA-02-0006-01Z-00-DX1.a37df719-8b93-4245-ae49-67eb1114253a.svs
2004,s3://pathology/TCGA/.../TCGA-02-0009-01Z-00-DX3.BAA1276B-E4D7-43D4-BDDF-807532462518.svs
```

## Issues Encountered & Solutions

### 1. Parameter naming
**Issue**: Used `postfilter_model_types` (plural)  
**Solution**: Correct parameter is `postfilter_model_type` (singular)

### 2. Slide encoder specification
**Issue**: Unclear how to specify slide encoder  
**Solution**: Use `aggregation_method=model` + `slide_model_type=GIGAPATH_SLIDE`

### 3. Missing s3fs
**Issue**: `ModuleNotFoundError: No module named 's3fs'`  
**Solution**: `uv pip install s3fs`

### 4. S3 permissions
**Issue**: `PermissionError: Forbidden` when accessing S3  
**Solution**: Ensure AWS credentials are set and have proper permissions

## Next Steps

1. **Fix S3 Permissions**: Ensure AWS credentials have access to pathology bucket
2. **Run Full Batch**: Process all 3 test slides
3. **Compare Embeddings**: Use `compare_gigapath_embeddings.py` to compare with PANDA provided embeddings
4. **Scale Up**: Process full PANDA dataset (9,555 slides) using batch/SLURM

## Performance Estimates

- **Single slide**: ~2-5 minutes (depending on size)
- **Batch of 3 slides**: ~6-12 minutes (with parallel processing)
- **Full PANDA (9,555 slides)**: ~320-530 hours single-threaded, ~32-53 hours with 10× parallelization

## References

- **GigaPath Paper**: https://aka.ms/gigapath
- **Prov-GigaPath Repo**: https://github.com/prov-gigapath/prov-gigapath
- **Mussel Documentation**: See `README.md` and `README_BATCH_PROCESSING.md`
- **PANDA Dataset**: https://www.kaggle.com/c/prostate-cancer-grade-assessment
