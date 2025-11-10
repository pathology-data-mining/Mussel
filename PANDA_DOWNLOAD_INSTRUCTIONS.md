# Downloading PANDA Slides for GigaPath Extraction

## Issue
The PANDA dataset is not available in the current S3 storage. To run GigaPath extraction on PANDA slides, you need to download them from Kaggle first.

## Option 1: Download Full Dataset from Kaggle

### Prerequisites
1. Create a Kaggle account: https://www.kaggle.com
2. Accept competition rules: https://www.kaggle.com/c/prostate-cancer-grade-assessment
3. Get API credentials:
   - Go to https://www.kaggle.com/settings/account
   - Click "Create New API Token"
   - Save `kaggle.json` to `~/.kaggle/`

### Download Commands
```bash
# Install Kaggle CLI
pip install kaggle

# Set permissions
chmod 600 ~/.kaggle/kaggle.json

# Download PANDA dataset (~400GB)
cd /gpfs/mskmind_ess/limr/repos/Mussel-3
mkdir -p panda_slides
kaggle competitions download -c prostate-cancer-grade-assessment -f train_images.zip -p panda_slides/

# Extract (takes time)
cd panda_slides
unzip train_images.zip

# Verify first 5 slides exist
ls -lh train_images/0005f7aaab2800f6170c399693a96917.tiff
ls -lh train_images/000920ad0b612851f8e01bcc880d9b3d.tiff
ls -lh train_images/001d865e65ef5d2579c190a0e0350d8f.tiff
ls -lh train_images/00412139e6b04d1e1cee8421f38f6e90.tiff
ls -lh train_images/006f4d8d3556dd21f6424202c2d294a9.tiff
```

## Option 2: Download Just 5 Slides (Manual)

Since Kaggle doesn't allow downloading individual files via API, you'll need to:

1. Download full dataset (train_images.zip)
2. Extract only the 5 slides we need:
   - 0005f7aaab2800f6170c399693a96917.tiff
   - 000920ad0b612851f8e01bcc880d9b3d.tiff
   - 001d865e65ef5d2579c190a0e0350d8f.tiff
   - 00412139e6b04d1e1cee8421f38f6e90.tiff
   - 006f4d8d3556dd21f6424202c2d294a9.tiff

```bash
# Extract only specific files
cd panda_slides
unzip train_images.zip train_images/0005f7aaab2800f6170c399693a96917.tiff \
                        train_images/000920ad0b612851f8e01bcc880d9b3d.tiff \
                        train_images/001d865e65ef5d2579c190a0e0350d8f.tiff \
                        train_images/00412139e6b04d1e1cee8421f38f6e90.tiff \
                        train_images/006f4d8d3556dd21f6424202c2d294a9.tiff
```

## Option 3: Upload to MinIO/S3

Once downloaded, upload to MinIO for easier access:

```bash
cd /gpfs/mskmind_ess/limr/repos/Mussel-3
source secrets.env

# Create bucket
aws --endpoint-url=$AWS_ENDPOINT_URL s3 mb s3://panda-dataset

# Upload 5 slides
aws --endpoint-url=$AWS_ENDPOINT_URL s3 cp \
  panda_slides/train_images/ \
  s3://panda-dataset/train_images/ \
  --recursive \
  --exclude "*" \
  --include "0005f7aaab2800f6170c399693a96917.tiff" \
  --include "000920ad0b612851f8e01bcc880d9b3d.tiff" \
  --include "001d865e65ef5d2579c190a0e0350d8f.tiff" \
  --include "00412139e6b04d1e1cee8421f38f6e90.tiff" \
  --include "006f4d8d3556dd21f6424202c2d294a9.tiff"

# Verify
aws --endpoint-url=$AWS_ENDPOINT_URL s3 ls s3://panda-dataset/train_images/
```

## Running GigaPath After Download

### With Local Files
```bash
cd /gpfs/mskmind_ess/limr/repos/Mussel-3
source .venv/bin/activate

python -m mussel.cli.tessellate_extract_features \
  prefilter_model_type=GIGAPATH \
  postfilter_model_type=GIGAPATH \
  aggregation_method=model \
  slide_model_type=GIGAPATH_SLIDE \
  slide_paths='[panda_slides/train_images/0005f7aaab2800f6170c399693a96917.tiff,panda_slides/train_images/000920ad0b612851f8e01bcc880d9b3d.tiff,panda_slides/train_images/001d865e65ef5d2579c190a0e0350d8f.tiff,panda_slides/train_images/00412139e6b04d1e1cee8421f38f6e90.tiff,panda_slides/train_images/006f4d8d3556dd21f6424202c2d294a9.tiff]' \
  output_dir=./panda_gigapath_output \
  batch_size=64 \
  slide_batch_size=5 \
  num_workers=8 \
  use_gpu=true \
  seg_config.patch_size=256 \
  save_features_to_h5=true \
  output_h5_suffix=.gigapath.h5
```

### With S3 (after upload)
```bash
cd /gpfs/mskmind_ess/limr/repos/Mussel-3
source secrets.env
export AWS_ENDPOINT_URL=http://pmindecs.mskcc.org:9020
source .venv/bin/activate

python -m mussel.cli.tessellate_extract_features \
  prefilter_model_type=GIGAPATH \
  postfilter_model_type=GIGAPATH \
  aggregation_method=model \
  slide_model_type=GIGAPATH_SLIDE \
  slide_paths='[s3://panda-dataset/train_images/0005f7aaab2800f6170c399693a96917.tiff,s3://panda-dataset/train_images/000920ad0b612851f8e01bcc880d9b3d.tiff,s3://panda-dataset/train_images/001d865e65ef5d2579c190a0e0350d8f.tiff,s3://panda-dataset/train_images/00412139e6b04d1e1cee8421f38f6e90.tiff,s3://panda-dataset/train_images/006f4d8d3556dd21f6424202c2d294a9.tiff]' \
  output_dir=./panda_gigapath_output \
  batch_size=64 \
  slide_batch_size=5 \
  num_workers=8 \
  use_gpu=true \
  seg_config.patch_size=256 \
  save_features_to_h5=true \
  output_h5_suffix=.gigapath.h5
```

## Expected Results

After ~15-30 minutes, you'll have 5 HDF5 files:
```
panda_gigapath_output/
b”œâ”€â”€ 0005f7aaab2800f6170c399693a96917.gigapath.h5  (768-dim slide embedding)
b”œâ”€â”€ 000920ad0b612851f8e01bcc880d9b3d.gigapath.h5
b”œâ”€â”€ 001d865e65ef5d2579c190a0e0350d8f.gigapath.h5
b”œâ”€â”€ 00412139e6b04d1e1cee8421f38f6e90.gigapath.h5
b””â”€â”€ 006f4d8d3556dd21f6424202c2d294a9.gigapath.h5
```

Each file contains:
- `slide_embedding`: 768-dimensional GigaPath slide encoding
- `features`: Patch-level features (N Ã— 1536)
- `coords`: Patch coordinates (N Ã— 2)

## Alternative: Use Pre-computed Embeddings Only

If downloading slides is not feasible, you can download the pre-computed embeddings for comparison:

```bash
# Download pre-computed GigaPath PANDA embeddings (32GB)
wget https://huggingface.co/datasets/prov-gigapath/prov-gigapath-tile-embeddings/resolve/main/GigaPath_PANDA_embeddings.zip
unzip GigaPath_PANDA_embeddings.zip

# Inspect provided embeddings
python -c "
import h5py
with h5py.File('GigaPath_PANDA_embeddings/h5_files/0005f7aaab2800f6170c399693a96917.h5', 'r') as f:
    print('Keys:', list(f.keys()))
    print('Embedding shape:', f['features'].shape if 'features' in f else 'N/A')
"
```

This lets you verify the embedding format without extracting your own.
