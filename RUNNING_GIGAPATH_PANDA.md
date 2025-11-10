# Running GigaPath on PANDA Slides - Complete Guide

## Status

The GigaPath workflow is ready to run on PANDA slides. The slides need to be downloaded from Kaggle first.

## Prerequisites

### 1. Download PANDA Dataset

PANDA slides are available from the Kaggle competition:

```bash
# Install Kaggle CLI
pip install kaggle

# Configure Kaggle API credentials
# Place kaggle.json in ~/.kaggle/

# Download PANDA dataset
kaggle competitions download -c prostate-cancer-grade-assessment

# Extract training images
unzip train_images.zip -d /data/panda/
```

**Alternative**: Download manually from https://www.kaggle.com/c/prostate-cancer-grade-assessment/data

### 2. Environment Setup

```bash
cd /gpfs/mskmind_ess/limr/repos/Mussel-3
source .venv/bin/activate

# Ensure s3fs is installed (already done)
uv pip list | grep s3fs
```

## Running GigaPath on 5 PANDA Slides

### Option 1: Local Files

If PANDA slides are stored locally:

```bash
cd /gpfs/mskmind_ess/limr/repos/Mussel-3

# Create slide list (update paths)
cat > panda_5_local.csv << 'EOF'
slide_id,slide_path
0005f7aaab2800f6170c399693a96917,/data/panda/train_images/0005f7aaab2800f6170c399693a96917.tiff
000920ad0b612851f8e01bcc880d9b3d,/data/panda/train_images/000920ad0b612851f8e01bcc880d9b3d.tiff
001d865e65ef5d2579c190a0e0350d8f,/data/panda/train_images/001d865e65ef5d2579c190a0e0350d8f.tiff
00412139e6b04d1e1cee8421f38f6e90,/data/panda/train_images/00412139e6b04d1e1cee8421f38f6e90.tiff
006f4d8d3556dd21f6424202c2d294a9,/data/panda/train_images/006f4d8d3556dd21f6424202c2d294a9.tiff
EOF

# Run GigaPath extraction
source .venv/bin/activate

python -m mussel.cli.tessellate_extract_features \
  prefilter_model_type=GIGAPATH \
  postfilter_model_type=GIGAPATH \
  aggregation_method=model \
  slide_model_type=GIGAPATH_SLIDE \
  slide_paths='[/data/panda/train_images/0005f7aaab2800f6170c399693a96917.tiff,/data/panda/train_images/000920ad0b612851f8e01bcc880d9b3d.tiff,/data/panda/train_images/001d865e65ef5d2579c190a0e0350d8f.tiff,/data/panda/train_images/00412139e6b04d1e1cee8421f38f6e90.tiff,/data/panda/train_images/006f4d8d3556dd21f6424202c2d294a9.tiff]' \
  output_dir=./panda_gigapath_output \
  batch_size=64 \
  slide_batch_size=5 \
  num_workers=8 \
  use_gpu=true \
  seg_config.patch_size=256 \
  seg_config.step_size=256 \
  seg_config.mpp=0.5 \
  save_features_to_h5=true \
  output_h5_suffix=.gigapath.h5 \
  2>&1 | tee panda_gigapath_run.log
```

### Option 2: S3/MinIO Storage

If slides are in S3-compatible storage:

```bash
cd /gpfs/mskmind_ess/limr/repos/Mussel-3

# Set S3 endpoint
export AWS_ENDPOINT_URL=http://pmindecs.mskcc.org:9020

source .venv/bin/activate

python -m mussel.cli.tessellate_extract_features \
  prefilter_model_type=GIGAPATH \
  postfilter_model_type=GIGAPATH \
  aggregation_method=model \
  slide_model_type=GIGAPATH_SLIDE \
  slide_paths='[s3://bucket/panda/0005f7aaab2800f6170c399693a96917.tiff,s3://bucket/panda/000920ad0b612851f8e01bcc880d9b3d.tiff,s3://bucket/panda/001d865e65ef5d2579c190a0e0350d8f.tiff,s3://bucket/panda/00412139e6b04d1e1cee8421f38f6e90.tiff,s3://bucket/panda/006f4d8d3556dd21f6424202c2d294a9.tiff]' \
  output_dir=./panda_gigapath_output \
  batch_size=64 \
  slide_batch_size=5 \
  num_workers=8 \
  use_gpu=true \
  seg_config.patch_size=256 \
  seg_config.step_size=256 \
  seg_config.mpp=0.5 \
  save_features_to_h5=true \
  output_h5_suffix=.gigapath.h5 \
  2>&1 | tee panda_gigapath_run.log
```

### Option 3: Docker

Using Docker for a reproducible environment:

```bash
cd /gpfs/mskmind_ess/limr/repos/Mussel-3

# Ensure PANDA slides are mounted at /data/panda/train_images/
docker run --rm --gpus all --shm-size=4g \
  -e HF_TOKEN=${HF_TOKEN} \
  -v /data/panda:/data/panda:ro \
  -v $(pwd)/panda_gigapath_output:/output \
  -w /output \
  mskmind/mussel:latest \
  python -m mussel.cli.tessellate_extract_features \
    prefilter_model_type=GIGAPATH \
    postfilter_model_type=GIGAPATH \
    aggregation_method=model \
    slide_model_type=GIGAPATH_SLIDE \
    slide_paths='[/data/panda/train_images/0005f7aaab2800f6170c399693a96917.tiff,/data/panda/train_images/000920ad0b612851f8e01bcc880d9b3d.tiff,/data/panda/train_images/001d865e65ef5d2579c190a0e0350d8f.tiff,/data/panda/train_images/00412139e6b04d1e1cee8421f38f6e90.tiff,/data/panda/train_images/006f4d8d3556dd21f6424202c2d294a9.tiff]' \
    output_dir=/output \
    batch_size=64 \
    slide_batch_size=5 \
    use_gpu=true \
    seg_config.patch_size=256 \
    save_features_to_h5=true \
    output_h5_suffix=.gigapath.h5
```

## Expected Output

After processing completes, you'll have 5 HDF5 files:

```
panda_gigapath_output/
├── 0005f7aaab2800f6170c399693a96917.gigapath.h5
├── 000920ad0b612851f8e01bcc880d9b3d.gigapath.h5
├── 001d865e65ef5d2579c190a0e0350d8f.gigapath.h5
├── 00412139e6b04d1e1cee8421f38f6e90.gigapath.h5
└── 006f4d8d3556dd21f6424202c2d294a9.gigapath.h5
```

Each HDF5 file contains:
- `coords`: Patch coordinates (N × 2)
- `features`: Patch-level features from GigaPath tile encoder (N × 1536)
- `slide_embedding`: Slide-level embedding from GigaPath slide encoder (768,)

## Monitoring Progress

Monitor the run with:

```bash
# Watch log file
tail -f panda_gigapath_run.log

# Check output directory
ls -lh panda_gigapath_output/

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## Expected Runtime

For 5 PANDA slides:
- **Tessellation**: ~2-5 minutes (depends on slide size)
- **Feature extraction**: ~10-20 minutes (depends on tissue coverage)
- **Slide aggregation**: ~1-2 minutes
- **Total**: ~15-30 minutes for 5 slides

## Workflow Phases

The tool will run through 4 phases:

1. **Phase 1: Tessellation** - Segment tissue and create patch coordinates
2. **Phase 2: Pre-filter Feature Extraction** - Extract GIGAPATH patch features
3. **Phase 3: Post-filter (skipped)** - No second model specified
4. **Phase 4: Slide-level Aggregation** - Aggregate to slide embeddings using GIGAPATH_SLIDE

## Comparing with Provided Embeddings

Once extraction is complete, compare with prov-gigapath embeddings:

```bash
# Download pre-computed PANDA embeddings (32GB)
wget https://huggingface.co/datasets/prov-gigapath/prov-gigapath-tile-embeddings/resolve/main/GigaPath_PANDA_embeddings.zip
unzip -n GigaPath_PANDA_embeddings.zip

# Run comparison
python compare_gigapath_embeddings.py \
  --mussel-dir ./panda_gigapath_output \
  --provided-dir ./GigaPath_PANDA_embeddings/h5_files \
  --slide-ids 0005f7aaab2800f6170c399693a96917 000920ad0b612851f8e01bcc880d9b3d 001d865e65ef5d2579c190a0e0350d8f 00412139e6b04d1e1cee8421f38f6e90 006f4d8d3556dd21f6424202c2d294a9 \
  --output-csv panda_comparison_results.csv
```

Expected similarity metrics:
- Cosine similarity: ≥ 0.99
- L2 distance: < 0.01
- Mean absolute difference: < 0.001

## Troubleshooting

### GPU Out of Memory
Reduce batch size:
```bash
batch_size=32  # or even 16
```

### Slow Processing
Increase parallelization:
```bash
num_workers=16        # More data loading workers
slide_batch_size=5    # Process all 5 slides together (if GPU memory allows)
```

### Missing Tissue
Adjust segmentation threshold:
```bash
seg_config.segment_threshold=15  # Lower = more permissive
seg_config.use_otsu=true          # Use Otsu thresholding
```

## Files Created

- `panda_5_slides.csv` - List of 5 PANDA slides with metadata
- `GIGAPATH_WORKFLOW_SUMMARY.md` - Complete workflow documentation
- `RUNNING_GIGAPATH_PANDA.md` - This guide

## Next Steps

1. Download PANDA slides from Kaggle
2. Update slide paths in command
3. Run GigaPath extraction (15-30 minutes)
4. Download pre-computed embeddings
5. Run comparison script
6. Analyze results

## Reference Commands

```bash
# Quick test with first slide only
python -m mussel.cli.tessellate_extract_features \
  prefilter_model_type=GIGAPATH \
  postfilter_model_type=GIGAPATH \
  aggregation_method=model \
  slide_model_type=GIGAPATH_SLIDE \
  slide_path=/data/panda/train_images/0005f7aaab2800f6170c399693a96917.tiff \
  output_h5_path=./test_output.h5 \
  output_pt_path=./test_output.pt \
  batch_size=64 \
  use_gpu=true \
  seg_config.patch_size=256 \
  save_features_to_h5=true

# View HDF5 output
python -c "
import h5py
with h5py.File('test_output.h5', 'r') as f:
    print('Keys:', list(f.keys()))
    print('Slide embedding shape:', f['slide_embedding'].shape)
    print('Patch features shape:', f['features'].shape)
    print('Coordinates shape:', f['coords'].shape)
"
```

## Support

For issues or questions:
- Workflow documentation: `GIGAPATH_WORKFLOW_SUMMARY.md`
- Comparison guide: `PANDA_GIGAPATH_COMPARISON.md`
- GigaPath repository: https://github.com/prov-gigapath/prov-gigapath
