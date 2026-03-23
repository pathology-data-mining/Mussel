# Mussel commands

This document describes the main command-line tools provided by Mussel, with examples.

## Commands

Mussel provides a set of CLI tools for tiling whole-slide images, working with tiled
slides, and generating feature embeddings with pathology foundation models.

* `tessellate` - tiling and foreground detection of whole-slide images
* `tessellate_extract_features` - combined tiling + feature extraction pipeline; supports batch processing from a directory
* `extract_features` - extract features from whole slide images (WSI) using a foundation model.
* `create_class_embeddings` - generate tissue-type embeddings for classifying tiles
* `annotate` - annotate tiles with tissue-types
* `cache_tiles` - save tile information in an efficient form for training
* `export_tiles` - export tiles as individual .png files using an HDF5 tile-coordinate manifest.
* `filter_features` - filter features using a classifier model
* `merge_annotation_features` - merge tile features with annotations from a BMP file.
* `linear_probe_benchmark` - benchmark a linear probe classifier on features extracted from a slide
* `save_model` - download and save a foundation model locally

Each of these commands is configurable with a number of different parameters.
You can always get a quick list of the parameters and default values for a given tool
by executing `<command> --help`.

### Examples

<img src="docs/example-mask.jpg" width="600px" />

The example commands below use the test data provided in the `tests/testdata` folder. 

### `tessellate`

Tessellate tiles a whole-slide image.  The tile coordinates and other metadata necessary
for downstream steps are written to an HDF5 (.h5) file.

Example command (see defaults with `tessellate --help`):
```bash
tessellate \
    slide_path=tests/testdata/948176.svs \
    output_h5_path=948176_coord.h5 \
    seg_config.segment_threshold=0 \
    num_workers=1
```

**New segmentation and patching options:**

| Parameter | Default | Description |
|---|---|---|
| `seg_config.overlap` | `0` | Patch overlap in absolute pixels. Sets `step_size = patch_size - overlap`. |
| `seg_config.min_tissue_proportion` | `0.0` | Discard patches where the tissue fraction is below this value (0.0–1.0). |
| `seg_config.remove_artifacts` | `false` | Enable artifact removal (requires `artifact_remover_fn` hook). |
| `seg_config.remove_penmarks` | `false` | Enable pen-mark removal (requires `artifact_remover_fn` hook). |
| `seg_config.seg_model` | `"classic"` | Segmentation backend: `"classic"` (HSV/Otsu) or `"hest"` (neural; requires `uv sync --extra neural-seg`). |

Example with 50% overlap and tissue filtering:
```bash
tessellate \
    slide_path=tests/testdata/948176.svs \
    output_h5_path=948176_coord.h5 \
    seg_config.overlap=128 \
    seg_config.min_tissue_proportion=0.5
```

### `extract_features`

Use a pathology foundation model to calculate feature embeddings for a slide tiled using
the `tessellate` commaand described above.  This generates both an HDF5 (.h5) file and
a PyTorch (.pt) file, with embeddings for each tile.

The following models are currently supported,

| Model          | model_type    | Reference |
|----------------|---------------|-----------|
| ResNet-50      | RESNET50      | https://huggingface.co/microsoft/resnet-50 |
| TransPath      | CTRANSPATH    | https://github.com/Xiyue-Wang/TransPath |
| Prov-GigaPath  | GIGAPATH      | https://github.com/prov-gigapath/prov-gigapath |
| Virchow        | VIRCHOW       | https://huggingface.co/paige-ai/Virchow |
| Virchow2       | VIRCHOW2      | https://huggingface.co/paige-ai/Virchow2 |
| H-Optimus-0    | OPTIMUS       | https://huggingface.co/bioptimus/H-optimus-0 |
| H-Optimus-1    | H_OPTIMUS_1   | https://huggingface.co/bioptimus/H-optimus-1 |
| H0-mini        | H0_MINI       | https://huggingface.co/bioptimus/H0-mini |
| Phikon         | PHIKON        | https://huggingface.co/owkin/phikon |
| Phikon-v2      | PHIKON_V2     | https://huggingface.co/owkin/phikon-v2 |
| Midnight-12k   | MIDNIGHT12K   | https://huggingface.co/kaiko-ai/midnight |
| GPFM           | GPFM          | https://huggingface.co/majiabo/GPFM |
| Hibou-L        | HIBOU_L       | https://huggingface.co/histai/hibou-L |
| UNI            | UNI           | https://huggingface.co/MahmoodLab/UNI |
| UNI2           | UNI2          | https://huggingface.co/MahmoodLab/UNI2 |
| OpenCLIP       | CLIP          | https://github.com/mlfoundations/open_clip |
| GooglePath     | GOOGLEPATH    | https://huggingface.co/google/path-foundation |
| Conch v1.5     | CONCH1_5      | https://huggingface.co/MahmoodLab/conchv1_5 |

**Slide encoders** (require patch-level features as input):

| Model          | model_type      | Patch encoder required |
|----------------|-----------------|------------------------|
| Prov-GigaPath  | GIGAPATH_SLIDE  | GIGAPATH |
| TITAN          | TITAN_SLIDE     | CONCH1_5 |
| PRISM          | PRISM_SLIDE     | VIRCHOW |
| FEATHER        | FEATHER_SLIDE   | CONCH1_5 |
| MADELEINE      | MADELEINE_SLIDE | CONCH1_5 |

OpenCLIP is used by default, with the default model being [QuiltNet-B-16-PMB](https://huggingface.co/wisdomik/QuiltNet-B-16-PMB).  Use the `model_type` parameter to specify a different model.
To use H-Optimus-0, for example,

```bash
extract_features \
    slide_path=tests/testdata/948176.svs \
    patch_h5_path=tests/testdata/948176.patch.h5 \
    model_type=OPTIMUS \
    output_h5_path=948176_feat.h5 \
    output_pt_path=948176_embed.pt
```

Most of the supported models download from HuggingFace, except for CTransPath and the
ResNet-50 model.  Three of the HuggingFace models (Prov-Gigapath, GooglePath, and Virchow)
are "gated", and to use these you need to sign an agreement on the HuggingFace site and
have your HuggingFace access token in the HF_TOKEN environment variable.

UNI and UNI2 are also gated models requiring an agreement and `HF_TOKEN`.

Finally, you can generate features from a folder of pre-tiled images, specifying the
folder using `patch_path` parameter.
```bash
extract_features \
    slide_path=None \
    patch_h5_path=None \
    patch_path=<path to folder w/ tiles in image format (.tif, .png, .jpg, etc.)> \
    output_h5_path=<path to output h5 file> \
    output_pt_path=None
```

### `tessellate_extract_features`

`tessellate_extract_features` runs tessellation and feature extraction in a single command.
It also supports **batch processing** of an entire directory of slides:

```bash
# Single slide
tessellate_extract_features \
    slide_path=tests/testdata/948176.svs \
    output_h5_path=948176_feat.h5 \
    output_pt_path=948176_embed.pt \
    model_type=OPTIMUS

# All slides in a directory (flat)
tessellate_extract_features \
    wsi_dir=/data/slides \
    output_h5_path=/data/features/{name}_feat.h5 \
    output_pt_path=/data/features/{name}_embed.pt \
    model_type=VIRCHOW2

# All slides in a directory tree (recursive)
tessellate_extract_features \
    wsi_dir=/data/slides \
    search_nested=true \
    output_h5_path=/data/features/{name}_feat.h5 \
    output_pt_path=/data/features/{name}_embed.pt \
    model_type=VIRCHOW2
```

Supported WSI extensions discovered during directory scan: `.svs`, `.ndpi`, `.tiff`, `.tif`, `.scn`, `.mrxs`, `.vms`, `.vmu`, `.bif`, `.czi`, `.lif`.

### `annotate`

You can generate embeddings for different tissue types, using the QuiltNet OpenClip model, and
use these to annotate a set of tiles for which you have OpenClip embeddings. 

The `tests/testdata/` folder includes some embeddings generated for the following tissue
types,

* "carcinoma in situ"
* "invasive carcinoma with lymphocytes"
* "tumor infiltrating lymphocytes"
* "lymphocytes"
* "carcinoma in situ with lymphocytes"
* "tumor-associated stroma with lymphocytes"

You can apply these to the sample slide with the command

```bash
annotate \
    features_pt_path=tests/testdata/948176.features.pt \
    class_embedding_pt_path=tests/testdata/class_embedding.pt \
    classes='["carcinoma in situ","invasive carcinoma","collagenous stroma","adipose","vessel","necrosis", "invasive adenocarcinoma","sarcoma"]' \
    output_csv_path=948176.annotations.csv 
```

### `create_class_embeddings`

You can also define your own classes with OpenClip! Any natural language works, and no training
is required.  For example,

```bash
create_class_embeddings \
    classes='["carcinoma in situ","invasive carcinoma with lymphocytes","tumor infiltrating lymphocytes","lymphocytes","carcinoma in situ with lymphocytes","tumor-associated stroma with lymphocytes"]' \
    output_pt_path=my_classes.pt

annotate \
    features_pt_path=tests/testdata/948176.features.pt \
    class_embedding_pt_path=my_classes.pt \
    classes='["carcinoma in situ","invasive carcinoma with lymphocytes","tumor infiltrating lymphocytes","lymphocytes","carcinoma in situ with lymphocytes","tumor-associated stroma with lymphocytes"]' \
    output_csv_path=948176.annotations-my-classes.csv
```

<img src="docs/example-interrog.png" width="600px" />

### `cache_tiles`

Use `cache_tiles` to generate a PyTorch (.pt) file for rapid access to tiles during I/O intense
operations such as training. This can be conditioned on tissue types: e.g. cache only the tiles
containing invasive carcinoma by setting `limit_to_class`. The `patch_h5_path` input file is
the output from `tessellate`.

```bash
cache_tiles \
    slide_path=tests/testdata/948176.svs \
    patch_h5_path=948176_coord.h5 \
    annotation_csv_path=tests/testdata/948176.annotation.csv \
    'limit_to_class=["carcinoma in situ", "invasive carcinoma with lymphocytes"]' \
    output_pt_path=948176_cache.pt \
    output_indices_json_path=948176_output_indices.json
```

*This takes about ten seconds for an example slide.*

### `save_model`

You can download and save a foundation model locally with the `save_model` command.

```bash
save_model model_type=OPTIMUS output_path=optimus.pkl
``` 



