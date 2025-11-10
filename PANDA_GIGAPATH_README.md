# GigaPath PANDA Embedding Comparison - Quick Start

This directory contains scripts and documentation for extracting GigaPath slide embeddings for the PANDA dataset using Mussel and comparing them with pre-computed embeddings from the prov-gigapath repository.

## Files Created

1. **`setup_panda_gigapath.py`** - Setup script to download PANDA metadata and create Mussel configurations
2. **`extract_gigapath_panda_embeddings.py`** - Full extraction and comparison pipeline
3. **`compare_gigapath_embeddings.py`** - Standalone script to compare embeddings
4. **`PANDA_GIGAPATH_COMPARISON.md`** - Comprehensive documentation

## Quick Start

### Step 1: Setup

Download PANDA metadata and create configuration files:

```bash
source .venv/bin/activate

python setup_panda_gigapath.py \
    --output-dir ./panda_setup \
    --panda-slides-dir /path/to/panda/train_images \
    --limit 10  # Optional: test with 10 slides
```

### Step 2: Extract Embeddings

Use Mussel to extract GigaPath slide embeddings:

```bash
# Single slide example
python -m mussel.cli.extract_features \
    slide_path=/path/to/slide.tiff \
    prefilter_model_type=GIGAPATH \
    postfilter_model_types=GIGAPATH_SLIDE \
    output_path=./output/slide.h5 \
    aggregation_method=model \
    patch_size=256 \
    use_gpu=true

# Batch processing
mussel-batch \
    --config panda_setup/panda_gigapath_config.yaml \
    --csv-manifest panda_setup/panda_processing_manifest.csv
```

### Step 3: Compare Embeddings

Compare Mussel-extracted embeddings with pre-computed ones:

```bash
python compare_gigapath_embeddings.py \
    --mussel-dir ./mussel_output \
    --provided-dir /path/to/GigaPath_PANDA_embeddings/h5_files \
    --output-csv comparison_results.csv
```

## PANDA Dataset Information

- **Total slides:** 9,555 whole-slide images
- **Task:** Prostate cancer grade assessment (ISUP grades 0-5)
- **Data providers:** Karolinska Institute and Radboud University Medical Center
- **Download:** https://www.kaggle.com/c/prostate-cancer-grade-assessment

## Pre-computed Embeddings

Download from HuggingFace:
- URL: https://huggingface.co/datasets/prov-gigapath/prov-gigapath-tile-embeddings/tree/main
- File: `GigaPath_PANDA_embeddings.zip` (32GB)

```bash
wget https://huggingface.co/datasets/prov-gigapath/prov-gigapath-tile-embeddings/resolve/main/GigaPath_PANDA_embeddings.zip
unzip -n GigaPath_PANDA_embeddings.zip
```

## Expected Results

When comparing embeddings, you should see:
- **Cosine similarity:** ≥ 0.99 (excellent match)
- **L2 distance:** < 0.01 (very similar)
- **Mean absolute difference:** < 0.001 (minimal difference)

## GigaPath Architecture

**Two-stage model:**

1. **Tile Encoder (Patch Level)**
   - Input: 256×256 patches
   - Output: 1536-dimensional features
   - Model: Vision Transformer

2. **Slide Encoder (Slide Level)**
   - Input: Patch features + coordinates
   - Output: 768-dimensional slide embedding
   - Model: Transformer with positional encoding

## Example Workflow

```bash
# 1. Setup
python setup_panda_gigapath.py --output-dir ./panda_test --limit 5

# 2. Download PANDA slides (manual step)
# Go to https://www.kaggle.com/c/prostate-cancer-grade-assessment
# Download train_images.zip and extract

# 3. Extract embeddings for a test slide
python -m mussel.cli.extract_features \
    slide_path=/data/panda/train_images/0005f7aaab2800f6170c399693a96917.tiff \
    prefilter_model_type=GIGAPATH \
    postfilter_model_types=GIGAPATH_SLIDE \
    output_path=./test_output/test_slide.h5 \
    aggregation_method=model \
    patch_size=256 \
    use_gpu=true \
    batch_size=128

# 4. Download pre-computed embeddings (optional)
wget https://huggingface.co/datasets/prov-gigapath/prov-gigapath-tile-embeddings/resolve/main/GigaPath_PANDA_embeddings.zip

# 5. Compare embeddings
python compare_gigapath_embeddings.py \
    --mussel-dir ./test_output \
    --provided-dir ./GigaPath_PANDA_embeddings/h5_files \
    --output-csv comparison_results.csv
```

## Troubleshooting

### Memory Issues
- Reduce `batch_size` (try 64 or 32)
- Reduce `num_workers` (try 4)
- Use CPU: `use_gpu=false`

### Model Download Issues
- Set HuggingFace token: `export HF_TOKEN=<your_token>`
- Check internet connection
- Manually cache models

### Slide Loading Errors
- Verify file paths (.tiff vs .tif)
- Check OpenSlide: `python -c "import openslide"`
- Ensure valid TIFF format

## Documentation

See `PANDA_GIGAPATH_COMPARISON.md` for:
- Detailed instructions
- Configuration options
- Advanced usage examples
- Performance tuning tips
- Troubleshooting guide

## References

- **GigaPath Repository:** https://github.com/prov-gigapath/prov-gigapath
- **Paper:** Xu et al. (2024). "A whole-slide foundation model for digital pathology from real-world data." Nature.
- **HuggingFace Model:** https://huggingface.co/prov-gigapath/prov-gigapath
- **PANDA Dataset:** https://www.kaggle.com/c/prostate-cancer-grade-assessment

## Citation

```bibtex
@article{xu2024gigapath,
  title={A whole-slide foundation model for digital pathology from real-world data},
  author={Xu, Hanwen and Usuyama, Naoto and Bagga, Jaspreet and others},
  journal={Nature},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
