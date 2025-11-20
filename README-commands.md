# Mussel Command Reference

This document provides detailed information about the command-line tools provided by Mussel, including parameters, examples, and use cases.

## Table of Contents

- [Overview](#overview)
- [Getting Help](#getting-help)
- [Commands](#commands)
  - [tessellate](#tessellate)
  - [extract_features](#extract_features)
  - [filter_tessellate](#filter_tessellate)
  - [tessellate_extract_features](#tessellate_extract_features)
  - [aggregate_slide_features](#aggregate_slide_features)
  - [create_class_embeddings](#create_class_embeddings)
  - [annotate](#annotate)
  - [cache_tiles](#cache_tiles)
  - [export_tiles](#export_tiles)
  - [filter_features](#filter_features)
  - [merge_annotation_features](#merge_annotation_features)
  - [linear_probe_benchmark](#linear_probe_benchmark)
  - [save_model](#save_model)

## Overview

Mussel provides a set of CLI tools for tiling whole-slide images, working with tiled
slides, and generating feature embeddings with pathology foundation models.

* `tessellate` - Tile whole-slide images with foreground detection
* `extract_features` - Extract feature embeddings using foundation models
* `filter_tessellate` - Integrated workflow: tessellate, extract CTRANSPATH features, and filter
* `tessellate_extract_features` - Integrated workflow with multi-model support: tessellate, extract patch/slide features with optional filtering
* `aggregate_slide_features` - Aggregate patch-level features to slide-level using various methods
* `create_class_embeddings` - Generate tissue-type embeddings for zero-shot classification
* `annotate` - Annotate tiles with tissue types using zero-shot learning
* `cache_tiles` - Cache tiles in an efficient format for training
* `export_tiles` - Export tiles as individual PNG files
* `filter_features` - Filter features using a trained classifier
* `merge_annotation_features` - Merge tile features with BMP annotations
* `linear_probe_benchmark` - Benchmark linear probe classifiers
* `save_model` - Download and save foundation models locally

## Getting Help

Each command provides detailed help information accessible via the `--help` flag:

```bash
<command> --help
```

For example:
```bash
tessellate --help
extract_features --help
```

This displays all available parameters, their types, and default values.

### Examples

<img src="docs/example-mask.jpg" width="600px" />

*Example of tissue segmentation and tiling output from the tessellate command*

The example commands below use the test data provided in the `tests/testdata` folder.

---

## Commands

### `tessellate`

**Purpose**: Tile a whole-slide image into smaller patches and detect tissue foreground regions.

The `tessellate` command performs tissue segmentation and generates a coordinate file (HDF5 format) that contains the locations of all tiles that contain tissue. This is typically the first step in a processing pipeline.

**Key Parameters:**
- `slide_path`: Path to your whole-slide image (.svs, .tif, etc.)
- `output_h5_path`: Where to save the tile coordinates
- `seg_config.patch_size`: Size of each tile in pixels (default: 256)
- `seg_config.mpp`: Microns per pixel for consistent physical sizing (default: 0.5)
- `seg_config.segment_threshold`: Threshold for tissue detection (0-255). Lower values are more permissive.
- `num_workers`: Number of parallel processing threads

**Example:**
```bash
tessellate \
    slide_path=tests/testdata/948176.svs \
    output_h5_path=948176_coord.h5 \
    seg_config.segment_threshold=0 \
    num_workers=1
```

**Output Files:**
- `948176_coord.h5`: HDF5 file containing tile coordinates and metadata (patch size, pyramid level, etc.)

**Tips:**
- For slides with light staining, decrease `segment_threshold` to capture more tissue
- For slides with artifacts, increase `segment_threshold` to be more selective
- Use `seg_config.mpp` to ensure consistent tile sizes across slides with different native resolutions

---

### `extract_features`

**Purpose**: Extract feature embeddings from slide tiles using a pathology foundation model.

This command processes the tiles identified by `tessellate` and generates feature vectors (embeddings) for each tile using a pre-trained foundation model. These embeddings can be used for downstream classification, clustering, or analysis tasks.

**Key Parameters:**
- `slide_path`: Path to your whole-slide image
- `patch_h5_path`: Tile coordinates from tessellate command
- `model_type`: Which foundation model to use (see table below)
- `output_h5_path`: Where to save features (HDF5 format)
- `output_pt_path`: Where to save features (PyTorch format)
- `batch_size`: Number of tiles to process at once (adjust based on GPU memory)
- `num_workers`: Number of data loading threads

**Supported Models:**

| Model         | model_type  | Framework | Gated? | Reference |
|---------------|-------------|-----------|--------|-----------|
| OpenCLIP (QuiltNet) | CLIP | PyTorch | No | https://github.com/mlfoundations/open_clip |
| ResNet-50     | RESNET50    | PyTorch | No | https://huggingface.co/microsoft/resnet-50 |
| TransPath     | CTRANSPATH  | PyTorch | No | https://github.com/Xiyue-Wang/TransPath |
| Prov-GigaPath | GIGAPATH    | PyTorch | Yes* | https://github.com/prov-gigapath/prov-gigapath |
| Virchow       | VIRCHOW     | PyTorch | Yes* | https://huggingface.co/paige-ai/Virchow |
| Virchow2      | VIRCHOW2    | PyTorch | Yes* | https://huggingface.co/paige-ai/Virchow2 |
| H-Optimus-0   | OPTIMUS     | PyTorch | No | https://huggingface.co/bioptimus/H-optimus-0 |
| Conch v1.5    | CONCH1_5    | PyTorch | No | https://huggingface.co/MahmoodLab/conchv1_5 |
| GooglePath    | GOOGLEPATH  | TensorFlow | Yes* | https://huggingface.co/google/path-foundation | 

*Gated models require HuggingFace authentication. See the [Troubleshooting](#troubleshooting) section in README.md.

**Default Model**: OpenCLIP with QuiltNet-B-16-PMB weights is used by default. This model works well for general histopathology tasks and doesn't require authentication.

**Automatic Patch Size Selection**: When using integrated workflows like `tessellate_extract_features` or `filter_tessellate`, the patch size for tessellation is automatically set based on the model type to match recommended values for optimal performance. The default patch sizes are:

| Model Type | Default Patch Size | Note |
|------------|-------------------|------|
| CONCH1_5, TITAN_SLIDE | 512 | Higher resolution capture |
| RESNET50, GIGAPATH, GIGAPATH_SLIDE, UNI, UNI2 | 256 | Standard resolution |
| CTRANSPATH, VIRCHOW, VIRCHOW2, OPTIMUS, CLIP, GOOGLEPATH | 224 | Optimized for ViT models |

You can override these defaults by explicitly setting `seg_config.patch_size` to a different value. The model will automatically resize patches to its expected input size during inference.

**Example - Using the default CLIP model:**
```bash
extract_features \
    slide_path=tests/testdata/948176.svs \
    patch_h5_path=tests/testdata/948176.patch.h5 \
    output_h5_path=948176_feat.h5 \
    output_pt_path=948176_embed.pt
```

**Example - Using H-Optimus-0 model:**
```bash
extract_features \
    slide_path=tests/testdata/948176.svs \
    patch_h5_path=tests/testdata/948176.patch.h5 \
    model_type=OPTIMUS \
    output_h5_path=948176_feat.h5 \
    output_pt_path=948176_embed.pt
```

**Example - Processing pre-tiled images:**
**Example - Processing pre-tiled images:**
If you have a folder of pre-extracted tile images, you can process them directly:
```bash
extract_features \
    slide_path=None \
    patch_h5_path=None \
    patch_path=<path to folder w/ tiles in image format (.tif, .png, .jpg, etc.)> \
    output_h5_path=<path to output h5 file> \
    output_pt_path=None
```

**Example - Two-step feature extraction:**
The tool automatically uses two-step process when aggregation_method is set to mean, max, or model:
```bash
# Extract features with two-step process (mean aggregation)
extract_features \
    slide_path=tests/testdata/948176.svs \
    patch_h5_path=tests/testdata/948176.patch.h5 \
    output_h5_path=948176_feat.h5 \
    output_pt_path=948176_embed.pt \
    intermediate_h5_path=948176_patch_feat.h5 \
    aggregation_method=mean

# Extract features with model-based slide aggregation (e.g., Prov-GigaPath)
# The patch encoder is automatically inferred from the slide encoder
# The aggregation_method is automatically set to 'model' when slide_model_type is specified
# Two-step mode is automatically inferred from aggregation_method
extract_features \
    slide_path=tests/testdata/948176.svs \
    patch_h5_path=tests/testdata/948176.patch.h5 \
    output_h5_path=948176_feat.h5 \
    output_pt_path=948176_embed.pt \
    intermediate_h5_path=948176_patch_feat.h5 \
    slide_model_type=GIGAPATH_SLIDE
```

The two-step process:
1. **Step 1 (Patch Encoding)**: Extracts features from individual patches and saves to `intermediate_h5_path`
2. **Step 2 (Slide Aggregation)**: Aggregates patch features to slide-level and saves to `output_h5_path`

Available aggregation methods:
- `identity`: No aggregation, keeps all patch features (default, backward compatible)
- `mean`: Mean pooling across patches (creates single slide-level feature vector)
- `max`: Max pooling across patches (creates single slide-level feature vector)
- `model`: Use a slide encoder model for learned aggregation (e.g., Prov-GigaPath slide encoder)

**Simplified model-based aggregation:**
When you specify `slide_model_type`, the `aggregation_method` is automatically set to `model`. You only need to specify:
- `slide_model_type`: The type of slide encoder model (e.g., GIGAPATH_SLIDE for Prov-GigaPath slide encoder)
- `slide_model_path`: Optional path to slide encoder model weights

**Important:** Each slide encoder is tied to a specific patch encoder:
- `GIGAPATH_SLIDE` automatically uses `GIGAPATH` as the patch encoder
- `TITAN_SLIDE` automatically uses `CONCH1_5` as the patch encoder
- The required patch encoder is inferred automatically - no need to specify `model_type`
- If you specify a different `model_type`, it will be overridden with the required patch encoder

**Output Files:**
- `*.h5`: HDF5 file with features array and coordinate information
- `*.pt`: PyTorch tensor file with features (can be loaded with `torch.load()`)
- `*.patch.h5`: Intermediate patch-level features (when using two-step mode)

**Tips:**
- Use `batch_size=32` or lower if you encounter GPU memory errors
- PyTorch models generally require the `torch-gpu` or `torch-cpu` installation
- TensorFlow models (GooglePath) require the `tensorflow-gpu` or `tensorflow-cpu` installation
- For gated models, set `HF_TOKEN` environment variable with your HuggingFace token

---

### `filter_tessellate`

**Purpose**: Integrated workflow that tessellates a whole-slide image, extracts features using a foundation model, and filters tiles using a classifier model in a single command.

This command combines the functionality of `tessellate`, `extract_features`, and `filter_features` into a streamlined workflow. It's particularly useful when you want to:
- Process slides end-to-end with a single command
- Use any supported foundation model for feature extraction
- Filter tiles based on a pre-trained classifier
- Reduce intermediate file management

**Key Parameters:**
- `slide_path`: Path to your whole-slide image
- `output_h5_path`: Path to save the final filtered HDF5 file with tile coordinates
- `output_pt_path`: Path to save the final filtered features in PyTorch format
- `classifier_pkl`: Path to the classifier model in pickle format
- `classifier_threshold`: Threshold for the classifier to filter features (default: 0.75)
- `model_type`: Type of foundation model to use (default: CTRANSPATH). Supports all models from extract_features.
- `model_path`: Path to the model weights file (optional, depends on model_type)
- `seg_config.*`: Segmentation parameters (same as tessellate)
- `batch_size`: Batch size for feature extraction (default: 64)
- `num_workers`: Number of workers for processing (default: 4)
- `use_gpu`: Whether to use GPU for feature extraction (default: True)
- `keep_intermediate_files`: Whether to keep intermediate files (default: False)
- `save_features_to_h5`: Whether to save filtered features to HDF5 (default: False)

**Example - Basic usage with CTRANSPATH:**
```bash
filter_tessellate \
    slide_path=tests/testdata/948176.svs \
    output_h5_path=948176_filtered.h5 \
    output_pt_path=948176_filtered.pt \
    classifier_pkl=my_classifier.pkl \
    classifier_threshold=0.75 \
    model_type=CTRANSPATH \
    model_path=ctranspath_model.pth \
    seg_config.segment_threshold=0 \
    num_workers=8 \
    batch_size=64
```

**Example - Using H-Optimus-0 model:**
```bash
filter_tessellate \
    slide_path=tests/testdata/948176.svs \
    output_h5_path=948176_filtered.h5 \
    output_pt_path=948176_filtered.pt \
    classifier_pkl=my_classifier.pkl \
    model_type=OPTIMUS \
    seg_config.segment_threshold=0
```

**Example - With visualization outputs:**
```bash
filter_tessellate \
    slide_path=tests/testdata/948176.svs \
    output_h5_path=948176_filtered.h5 \
    output_pt_path=948176_filtered.pt \
    classifier_pkl=my_classifier.pkl \
    model_type=CTRANSPATH \
    model_path=ctranspath_model.pth \
    output_mask_path=948176_mask.png \
    output_grid_mask_path=948176_grid.png \
    output_thumbnail_path=948176_thumb.png \
    seg_config.segment_threshold=0
```

**Example - Keeping intermediate files for debugging:**
```bash
filter_tessellate \
    slide_path=tests/testdata/948176.svs \
    output_h5_path=948176_filtered.h5 \
    output_pt_path=948176_filtered.pt \
    classifier_pkl=my_classifier.pkl \
    model_type=CTRANSPATH \
    model_path=ctranspath_model.pth \
    keep_intermediate_files=True \
    save_features_to_h5=True
```

**Workflow:**
1. **Step 1 - Tessellation**: Tiles the whole-slide image and detects tissue regions
2. **Step 2 - Feature Extraction**: Extracts features from detected tiles using the specified foundation model
3. **Step 3 - Filtering**: Applies the classifier and keeps only tiles above the threshold

**Output Files:**
- `output_h5_path`: HDF5 file with filtered tile coordinates
- `output_pt_path`: PyTorch tensor file with filtered features
- `*.tessellate.h5`: Intermediate tessellation coordinates (if keep_intermediate_files=True)
- `*.features.h5`: Intermediate feature file (if keep_intermediate_files=True)
- Optional visualization outputs (mask, grid, thumbnail) if specified

**Tips:**
- Supports all foundation models available in `extract_features` (CTRANSPATH, CLIP, OPTIMUS, VIRCHOW, etc.)
- The classifier should be compatible with the feature dimensions of the selected model
- Some models (like CTRANSPATH) require `model_path`, while others (like CLIP, OPTIMUS) download automatically
- Use `keep_intermediate_files=True` for debugging or if you need the intermediate results
- By default, intermediate files are created in a temporary directory and cleaned up automatically
- Lower `classifier_threshold` to keep more tiles, higher to be more selective
- Adjust `batch_size` based on available GPU memory

---

### `tessellate_extract_features`

**Purpose**: Integrated workflow that tessellates whole-slide images, extracts patch-level and/or slide-level features, with optional filtering in a single command.

This command combines tessellation, feature extraction, and optional filtering into one streamlined workflow. It supports both single-slide and batch processing modes, and can extract features using multiple models simultaneously.

**Key Model Parameters:**

There are two types of models used in this workflow:

1. **Patch-Level Models** (`model_type`):
   - Extracts features from individual tissue patches/tiles
   - Examples: OPTIMUS, VIRCHOW, UNI, CTRANSPATH, RESNET50
   - Single mode: `model_type=OPTIMUS`
   - Batch mode with multiple models: `model_type=[OPTIMUS,VIRCHOW,UNI]`
   - Command line usage: Use unquoted list notation: `model_type=[MODEL1,MODEL2]`

2. **Slide-Level Models** (`slide_model_type`):
   - Aggregates patch features into slide-level representations
   - Examples: GIGAPATH_SLIDE, TITAN_SLIDE
   - Single mode: `slide_model_type=GIGAPATH_SLIDE`
   - Batch mode with multiple models: `slide_model_type=[GIGAPATH_SLIDE,TITAN_SLIDE]`
   - Command line usage: Use unquoted list notation: `slide_model_type=[MODEL1,MODEL2]`
   - **Important**: Each slide encoder automatically uses its required patch encoder:
     - `GIGAPATH_SLIDE` → automatically uses `GIGAPATH` patch encoder
     - `TITAN_SLIDE` → automatically uses `CONCH1_5` patch encoder

**Key Parameters:**
- `slide_path` (single mode): Path to whole-slide image
- `slide_paths` (batch mode): List of paths to whole-slide images (use `slide_paths=[path1.svs,path2.svs]`)
- `output_h5_path` (single mode): Path to save features HDF5
- `output_pt_path` (single mode): Path to save features PyTorch tensor
- `output_dir` (batch mode): Directory to save all outputs
- `model_type`: Patch-level feature extraction model(s) - accepts lists in batch mode
- `slide_model_type`: Slide-level aggregation model(s) - accepts lists in batch mode
- `prefilter_model_type`: Model for initial filtering (if using classifier)
- `classifier_pkl`: Optional classifier for tile filtering
- `seg_config.*`: Segmentation parameters

**Example - Single slide with patch-level model:**
```bash
tessellate_extract_features \
    slide_path=tests/testdata/948176.svs \
    output_h5_path=948176_features.h5 \
    output_pt_path=948176_features.pt \
    model_type=OPTIMUS
```

**Example - Single slide with slide-level model:**
```bash
# GIGAPATH_SLIDE automatically uses GIGAPATH as the patch encoder
tessellate_extract_features \
    slide_path=tests/testdata/948176.svs \
    output_h5_path=948176_slide_features.h5 \
    output_pt_path=948176_slide_features.pt \
    slide_model_type=GIGAPATH_SLIDE
```

**Example - Batch processing with multiple patch-level models:**
```bash
tessellate_extract_features \
    slide_paths=[slide1.svs,slide2.svs,slide3.svs] \
    output_dir=./output \
    model_type=[OPTIMUS,VIRCHOW,UNI]
```

**Example - Batch processing with multiple slide-level models:**
```bash
# Both GIGAPATH_SLIDE and TITAN_SLIDE will automatically use their required patch encoders
tessellate_extract_features \
    slide_paths=[slide1.svs,slide2.svs,slide3.svs] \
    output_dir=./output \
    slide_model_type=[GIGAPATH_SLIDE,TITAN_SLIDE]
```

**Example - With filtering:**
```bash
tessellate_extract_features \
    slide_paths=[slide1.svs,slide2.svs] \
    output_dir=./output \
    model_type=OPTIMUS \
    classifier_pkl=my_classifier.pkl \
    classifier_threshold=0.75 \
    prefilter_model_type=CTRANSPATH
```

**Output Files:**
- Batch mode with multiple models creates subdirectories for each model
- Each model's output is organized in: `{output_dir}/{slide_id}/{model_name}/`
- Includes HDF5 features, PyTorch tensors, and optional visualizations

**Tips:**
- Use batch mode with lists for efficient multi-model processing
- When specifying lists on command line, don't quote the brackets: `model_type=[A,B,C]`
- Slide encoders automatically pair with their required patch encoders
- Batch processing provides 6-8x speedup when using slide-level models

---

### `aggregate_slide_features`

**Purpose**: Aggregate patch-level features to slide-level features using various aggregation methods.

This command takes an HDF5 file containing patch-level feature embeddings (as produced by `extract_features` with two-step mode) and aggregates them to slide-level features. This is useful when you want to:
- Apply different aggregation strategies to the same patch features
- Use slide encoder models (e.g., Prov-GigaPath, TITAN) for learned aggregation
- Separate patch extraction from slide aggregation for more flexible processing

**Key Parameters:**
- `patch_features_h5_path`: Path to HDF5 file containing patch-level features
- `output_h5_path`: Where to save the aggregated slide-level features
- `aggregation_method`: Aggregation method - 'identity', 'mean', 'max', or 'model'
- `slide_model_type`: Type of slide encoder model (when using aggregation_method='model')
- `slide_model_path`: Optional path to slide encoder model weights
- `use_gpu`: Whether to use GPU for model-based aggregation
- `gpu_device_id`: Specific GPU device ID to use

**Supported Slide Encoder Models:**
- `GIGAPATH_SLIDE`: Prov-GigaPath slide encoder (requires GIGAPATH patch features)
- `TITAN_SLIDE`: MahmoodLab/TITAN slide encoder (requires CONCH1_5 patch features)

**Example - Mean pooling aggregation:**
```bash
# First, extract patch-level features
extract_features \
    slide_path=tests/testdata/948176.svs \
    patch_h5_path=tests/testdata/948176.patch.h5 \
    output_h5_path=948176_patch_feat.h5 \
    intermediate_h5_path=948176_patch_feat.h5 \
    aggregation_method=identity

# Then aggregate to slide-level using mean pooling
aggregate_slide_features \
    patch_features_h5_path=948176_patch_feat.h5 \
    output_h5_path=948176_slide_feat_mean.h5 \
    aggregation_method=mean
```

**Example - Using Prov-GigaPath slide encoder:**
```bash
# First, extract patch-level features with GIGAPATH
extract_features \
    slide_path=tests/testdata/948176.svs \
    patch_h5_path=tests/testdata/948176.patch.h5 \
    model_type=GIGAPATH \
    output_h5_path=948176_patch_feat.h5 \
    intermediate_h5_path=948176_patch_feat.h5 \
    aggregation_method=identity

# Then aggregate using GigaPath slide encoder
# aggregation_method is automatically set to 'model' when slide_model_type is specified
aggregate_slide_features \
    patch_features_h5_path=948176_patch_feat.h5 \
    output_h5_path=948176_slide_feat_gigapath.h5 \
    slide_model_type=GIGAPATH_SLIDE
```

**Example - Using TITAN slide encoder:**
```bash
# First, extract patch-level features with CONCH1_5
extract_features \
    slide_path=tests/testdata/948176.svs \
    patch_h5_path=tests/testdata/948176.patch.h5 \
    model_type=CONCH1_5 \
    output_h5_path=948176_patch_feat.h5 \
    intermediate_h5_path=948176_patch_feat.h5 \
    aggregation_method=identity

# Then aggregate using TITAN slide encoder
aggregate_slide_features \
    patch_features_h5_path=948176_patch_feat.h5 \
    output_h5_path=948176_slide_feat_titan.h5 \
    slide_model_type=TITAN_SLIDE
```

**Output Files:**
- `*.h5`: HDF5 file with aggregated slide-level features

**Tips:**
- The patch features file must contain features from the correct patch encoder for the slide encoder you're using
- `GIGAPATH_SLIDE` requires patch features extracted with `GIGAPATH`
- `TITAN_SLIDE` requires patch features extracted with `CONCH1_5`
- Coordinates and patch_size are automatically extracted from the patch features HDF5 file
- For gated models, set `HF_TOKEN` environment variable with your HuggingFace token

---

### `create_class_embeddings`

**Purpose**: Generate text embeddings for tissue types to enable zero-shot classification.

This command creates embeddings for natural language descriptions of tissue types using the CLIP model. These embeddings can then be used to classify tiles without any training data.

**Key Parameters:**
- `classes`: List of tissue type descriptions (natural language)
- `output_pt_path`: Where to save the class embeddings
- `model_type`: Model to use (default: CLIP, recommended for zero-shot)

**Example:**
```bash
create_class_embeddings \
    classes='["carcinoma in situ","invasive carcinoma with lymphocytes","tumor infiltrating lymphocytes","lymphocytes","carcinoma in situ with lymphocytes","tumor-associated stroma with lymphocytes"]' \
    output_pt_path=my_classes.pt
```

**Tips:**
- Use descriptive, natural language for class names (e.g., "invasive carcinoma" rather than "IC")
- You can include multiple descriptors (e.g., "tumor with lymphocytes")
- No training data required - the model understands the semantic meaning of the text
- Works best with the CLIP model (default)

---

### `annotate`

**Purpose**: Classify slide tiles using zero-shot learning with pre-computed embeddings.

This command compares tile feature embeddings with class embeddings to assign tissue type labels to each tile. It uses cosine similarity to find the best matching tissue type for each tile.

**Key Parameters:**
- `features_pt_path`: Tile features from extract_features command
- `class_embedding_pt_path`: Class embeddings from create_class_embeddings command
- `classes`: Same list of classes used in create_class_embeddings
- `output_csv_path`: Where to save the classification results
- `interrogate`: Whether to generate an HTML visualization report
- `slide_path`: Required if interrogate=True
- `patch_path`: Required if interrogate=True

**Example - Basic annotation:**

**Example - Basic annotation:**

```bash
annotate \
    features_pt_path=tests/testdata/948176.features.pt \
    class_embedding_pt_path=tests/testdata/class_embedding.pt \
    classes='["carcinoma in situ","invasive carcinoma","collagenous stroma","adipose","vessel","necrosis", "invasive adenocarcinoma","sarcoma"]' \
    output_csv_path=948176.annotations.csv 
```

**Example - With visualization:**
```bash
annotate \
    features_pt_path=tests/testdata/948176.features.pt \
    class_embedding_pt_path=tests/testdata/class_embedding.pt \
    classes='["carcinoma in situ","invasive carcinoma","stroma","lymphocytes"]' \
    output_csv_path=948176.annotations.csv \
    interrogate=True \
    slide_path=tests/testdata/948176.svs \
    patch_path=tests/testdata/948176.patch.h5 \
    interrogation_report_path=948176_report.html
```

**Output Files:**
- `*.csv`: CSV file with tile coordinates and predicted class labels
- `*.html` (if interrogate=True): Interactive HTML report showing predictions overlaid on slide

<img src="docs/example-interrog.png" width="600px" />

*Example of interrogation report showing tissue type predictions*

**Tips:**
- The classes list must match exactly what was used in create_class_embeddings
- Use interrogate mode to visually verify annotation quality
- Results are based on cosine similarity between tile and class embeddings

---

### `cache_tiles`

**Purpose**: Pre-cache tile images in an efficient format for fast training data loading.

This command loads tiles from a slide and saves them as a PyTorch tensor file. This dramatically speeds up data loading during training, as tiles are pre-extracted and stored in a contiguous format.

**Key Parameters:**
- `slide_path`: Path to whole-slide image
- `patch_h5_path`: Tile coordinates from tessellate
- `output_pt_path`: Where to save cached tiles
- `annotation_csv_path`: Optional annotations to filter tiles
- `limit_to_class`: Optional list of classes to include
- `batch_size`: Number of tiles to process at once
- `num_workers`: Number of data loading threads

**Example:**
**Example:**
```bash
cache_tiles \
    slide_path=tests/testdata/948176.svs \
    patch_h5_path=948176_coord.h5 \
    annotation_csv_path=tests/testdata/948176.annotation.csv \
    'limit_to_class=["carcinoma in situ", "invasive carcinoma with lymphocytes"]' \
    output_pt_path=948176_cache.pt \
    output_indices_json_path=948176_output_indices.json
```

**Output Files:**
- `*.pt`: PyTorch tensor file with cached tile images (N x C x H x W)
- `*_indices.json`: JSON file mapping indices to original coordinates

**Tips:**
- Use `limit_to_class` to cache only specific tissue types for targeted training
- Cached files can be very large - ensure sufficient disk space
- Processing time: ~10 seconds per thousand tiles on typical hardware

---

### `export_tiles`

**Purpose**: Export individual tiles as PNG image files for visualization or external processing.

This command extracts tiles from a whole-slide image based on tessellate coordinates and saves each as a separate PNG file.

**Key Parameters:**
- `slide_path`: Path to whole-slide image
- `patch_h5_path`: Tile coordinates from tessellate
- `output_png_path`: Directory to save PNG files
- `patch_size`: Size of exported tiles in pixels
- `mpp`: Microns per pixel
- `num_workers`: Number of parallel workers

**Example:**
```bash
export_tiles \
    slide_path=tests/testdata/948176.svs \
    patch_h5_path=948176_coord.h5 \
    output_png_path=./exported_tiles/ \
    patch_size=256 \
    mpp=0.5 \
    num_workers=8
```

**Output Files:**
- Individual PNG files named by tile coordinates (e.g., `tile_x1024_y2048.png`)

**Tips:**
- Use this for visual inspection of tiles or creating custom datasets
- Can generate many files - ensure sufficient disk space
- Adjust `num_workers` based on available CPU cores

---

### `filter_features`

**Purpose**: Filter tile features using a trained classifier to select specific tile types.

This command applies a pre-trained classifier to tile features and keeps only tiles that meet a threshold score. Useful for selecting high-quality tiles or specific tissue types.

**Key Parameters:**
- `features_h5_path`: Input features (HDF5 format)
- `features_pt_path`: Input features (PyTorch format, optional)
- `classifier_pkl`: Path to trained classifier (pickle format)
- `classifier_threshold`: Minimum score to keep a tile (0-1)
- `output_h5_path`: Filtered features (HDF5 format)
- `output_pt_path`: Filtered features (PyTorch format)

**Example:**
```bash
filter_features \
    features_h5_path=948176_feat.h5 \
    features_pt_path=948176_embed.pt \
    classifier_pkl=my_classifier.pkl \
    classifier_threshold=0.75 \
    output_h5_path=948176_filtered.h5 \
    output_pt_path=948176_filtered.pt
```

**Output Files:**
- Filtered HDF5 and PyTorch files containing only tiles meeting the threshold

**Tips:**
- Train your classifier using tools like scikit-learn
- Lower threshold = more permissive (more tiles kept)
- Higher threshold = more selective (fewer tiles kept)

---

### `merge_annotation_features`

**Purpose**: Merge tile features with annotations from external annotation tools (BMP format).

This command combines feature embeddings with pixel-level annotations, useful when you have manual annotations from tools like Aperio or QuPath exported as BMP files.

**Key Parameters:**
- `features_h5_path`: Tile features from extract_features
- `annotation_bmp_path`: BMP annotation file
- `output_parquet_path`: Where to save merged data
- `slide_id`: Optional identifier for the slide
- `class_mapping_yaml_path`: Optional YAML file for mapping annotation colors to class labels

**Example:**
```bash
merge_annotation_features \
    features_h5_path=948176_feat.h5 \
    annotation_bmp_path=948176_annotations.bmp \
    output_parquet_path=948176_merged.parquet \
    slide_id=948176 \
    class_mapping_yaml_path=class_map.yaml
```

**Output Files:**
- `*.parquet`: Parquet file with features and annotation labels per tile

**Tips:**
- BMP annotations should be aligned with the slide coordinate space
- Use class_mapping_yaml to map annotation pixel values to meaningful labels
- Output is in GeoDataFrame format for spatial analysis

---

### `linear_probe_benchmark`

**Purpose**: Train and evaluate a linear probe classifier on extracted features.

This command trains a logistic regression classifier on tile features with annotations, useful for benchmarking feature quality and model performance.

**Key Parameters:**
- `features_annotation_parquet_path`: Merged features and annotations
- `annotation_percent_filter_threshold`: Minimum overlap for a tile to be included
- `test_size`: Proportion of data for testing
- `val_size`: Proportion of training data for validation
- `C`: Regularization strength
- `output_csv`: Classification report output
- `output_png`: Confusion matrix visualization

**Example:**
```bash
linear_probe_benchmark \
    features_annotation_parquet_path=948176_merged.parquet \
    annotation_percent_filter_threshold=0.50 \
    test_size=0.2 \
    val_size=0.1 \
    C=1.0 \
    output_csv=classification_report.csv \
    output_png=confusion_matrix.png
```

**Output Files:**
- `*.csv`: Classification metrics (precision, recall, F1-score)
- `*.png`: Confusion matrix visualization

**Tips:**
- Use this to evaluate how well features discriminate between tissue types
- Adjust `C` parameter for regularization (lower = stronger regularization)
- Results indicate feature quality for downstream tasks

---

### `save_model`

**Purpose**: Download and save a foundation model locally for offline use.

This command downloads a foundation model from HuggingFace or other sources and saves it locally, useful for environments with limited internet access.

**Key Parameters:**
- `model_type`: Which model to download (CLIP, OPTIMUS, etc.)
- `model_path`: Optional custom model path/identifier
- `output_path`: Where to save the model

**Example:**
**Example:**
```bash
save_model model_type=OPTIMUS output_path=optimus.pkl
```

**Tips:**
- Useful for HPC environments or air-gapped systems
- For gated models, ensure HF_TOKEN is set before running
- Saved models can be loaded using the `model_path` parameter in extract_features