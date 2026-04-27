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
* `convert` - convert whole-slide images to pyramidal TIFF format (single file or batch)

Each of these commands is configurable with a number of different parameters.
You can always get a quick list of the parameters and default values for a given tool
by executing `<command> --help`.

### Examples

<img src="docs/example-mask.jpg" width="600px" />

The example commands below use the test data provided in the `tests/testdata` folder. 

### `tessellate`

Tessellate tiles a whole-slide image.  The tile coordinates and other metadata necessary
for downstream steps are written to an HDF5 (.h5) file.

Mussel reads tiles from the slide at the resolution specified by `seg_config.mpp`
(default 0.5 µm/px, roughly 20×). The slide's native MPP is determined automatically
from the file metadata; see [MPP fallback chain](#mpp-resolution) below.

Example command (see defaults with `tessellate --help`):
```bash
tessellate \
    slide_path=tests/testdata/948176.svs \
    output_h5_path=948176_coord.h5 \
    seg_config.segment_threshold=0 \
    num_workers=1
```

#### Supported slide formats

Mussel uses [tiffslide](https://github.com/Bayer-Group/tiffslide)
(backed by [tifffile](https://github.com/cgohlke/tifffile)) to read whole-slide images.

| Format | Extension | Vendor | Tiffslide support |
|---|---|---|---|
| Aperio SVS | `.svs` | Leica/Aperio | ✅ Full |
| Leica SCN | `.scn` | Leica | ✅ Full |
| Generic / OME TIFF | `.tif`, `.tiff` | Various | ✅ Full |
| Hamamatsu NDPI | `.ndpi` | Hamamatsu | ⚠️ Partial — MPP from TIFF tags |
| Ventana BIF | `.bif` | Ventana/Roche | ⚠️ Partial — MPP from TIFF tags |
| MIRAX | `.mrxs` | 3DHistech | ⚠️ Generic TIFF; requires sidecar dir |
| Hamamatsu VMS/VMU | `.vms`, `.vmu` | Hamamatsu | ⚠️ Generic TIFF |
| PerkinElmer QPTIFF | `.qptiff` | PerkinElmer | ⚠️ Generic TIFF; first channel only |
| Zeiss CZI | `.czi` | Zeiss | ⚠️ Generic TIFF; first series only |

**Format limitations:**
- **NDPI / BIF** — tiffslide's vendor parsers are incomplete; MPP is derived from
  `tiff.XResolution` / `tiff.ResolutionUnit` tags (works for most files). Use
  `seg_config.slide_mpp_override` if MPP is incorrect.
- **MRXS** — multi-file format: the `.mrxs` file and its sidecar directory (same name,
  no extension) must be in the same location. Moving the `.mrxs` alone will fail.
- **QPTIFF** — multiplex/multi-channel files are tiled using the first channel only.
- **CZI** — multi-series files (multiple acquisitions) use series 0 only.
- **VMS / VMU** — uncommon on modern scanners; validate before production use.

#### MPP resolution

Mussel determines the slide's native microns-per-pixel (MPP) using the following
fallback chain. The first value found is used:

1. **`seg_config.slide_mpp_override`** — explicit CLI override; bypasses all metadata reading
2. **`tiffslide.mpp-x`** — standard property populated by tiffslide for all supported formats
3. **`aperio.MPP`** / **`openslide.mpp-x`** — legacy vendor properties
4. **`tiff.XResolution` + `tiff.ResolutionUnit`** — raw TIFF resolution tags converted to µm/px (INCH, CENTIMETER, MILLIMETER, MICROMETER supported); tiffslide exposes these for partially-supported formats (NDPI, BIF, MRXS, QPTIFF, CZI) even when it cannot normalize them to `tiffslide.mpp-x`
5. **Magnification estimate** — derived from objective-power metadata as `10.0 / magnification`
6. **Default 0.5 µm/px** — used as last resort with a warning logged

If the slide has missing or corrupt MPP metadata, use the override:
```bash
tessellate slide_path=slide.svs seg_config.slide_mpp_override=0.5 ...
tessellate_extract_features slide_path=slide.svs seg_config.slide_mpp_override=0.25 ...
export_tiles slide_path=slide.svs slide_mpp_override=0.5 ...
```

#### Segmentation presets

Pass `seg_config=<preset>` to select a built-in segmentation profile tuned for a specific
specimen type. Individual parameters can still be overridden after the preset:

```bash
tessellate slide_path=slide.svs output_h5_path=out.h5 seg_config=biopsy
tessellate slide_path=slide.svs output_h5_path=out.h5 seg_config=tcga seg_config.mpp=0.25
```

| Preset | Best for | Key differences from `default` | With `seg_model=neural` |
|---|---|---|---|
| `default` | General use | Baseline values (see table below). | Fully compatible; no warnings. |
| `biopsy` | Needle-core / punch biopsies | Lower area thresholds (`tissue_area_threshold=1`, `hole_area_threshold=1`) to keep small tissue cores; fewer holes (`max_num_holes=2`). | Area thresholds and `max_num_holes` still apply; `segment_threshold`/`median_blur_ksize` are ignored with a warning. |
| `resection` | Surgical resection specimens | Stronger morphological closing (`morphology_ex_kernel=4`) to bridge gaps in large sections; same area thresholds as `default`. | Only `morphology_ex_kernel=4` has effect; `segment_threshold`/`median_blur_ksize` are ignored with a warning. Consider `default seg_config.seg_model=neural seg_config.morphology_ex_kernel=4` to avoid the warning. |
| `tcga` | TCGA whole-slide images | Lower `segment_threshold=8` to capture pale/faded tissue; stronger closing (`morphology_ex_kernel=4`); reduced area thresholds. | `segment_threshold` is ignored with a warning (`median_blur_ksize=7` matches the default so no second warning); `morphology_ex_kernel=4` and area thresholds still apply. |

#### Segmentation and patching options

| Parameter | Default | Description |
|---|---|---|
| `seg_config.mpp` | `0.5` | Target resolution for tile extraction (µm/px). |
| `seg_config.patch_size` | `256` | Tile size in pixels at the target MPP. |
| `seg_config.overlap` | `0` | Patch overlap in absolute pixels. Sets `step_size = patch_size - overlap`. |
| `seg_config.min_tissue_proportion` | `0.0` | Discard patches where the tissue fraction is below this value (0.0–1.0). |
| `seg_config.remove_artifacts` | `false` | Enable artifact removal (requires `artifact_remover_fn` hook). |
| `seg_config.remove_penmarks` | `false` | Enable pen-mark removal (requires `artifact_remover_fn` hook). |
| `seg_config.seg_model` | `"classic"` | Segmentation backend: `"classic"` (HSV + fixed threshold), `"otsu"` (HSV + Otsu automatic threshold), or `"neural"` (deep learning; see below). Note: the old `seg_config.use_otsu=true` flag is deprecated — use `seg_model=otsu` instead. |
| `seg_config.slide_mpp_override` | `null` | Override the slide's native MPP; useful when metadata is missing or wrong. |

Example with 50% overlap and tissue filtering:
```bash
tessellate \
    slide_path=tests/testdata/948176.svs \
    output_h5_path=948176_coord.h5 \
    seg_config.overlap=128 \
    seg_config.min_tissue_proportion=0.5
```

#### Neural tissue segmentation (`seg_model="neural"`)

By default Mussel uses a classic HSV/Otsu threshold pipeline (`seg_model="classic"`).
Setting `seg_model="neural"` switches to a deep-learning segmenter that is more
robust on challenging slides (stain variation, artefacts, pale tissue).

The neural segmenter uses a **DeepLabV3-ResNet50** model (2-class: tissue vs background)
trained on histopathology slides as part of the
[HEST](https://github.com/mahmoodlab/HEST) project at the Mahmood Lab, Harvard Medical
School. The pre-trained checkpoint is hosted on HuggingFace at
[MahmoodLab/hest-tissue-seg](https://huggingface.co/MahmoodLab/hest-tissue-seg) and is
downloaded automatically on first use (no account or token required).

> **Reference:** Chan *et al.*, "A Pathology Foundation Model for Cancer Diagnosis and
> Prognosis Prediction", *Nature* 2025.
> [[paper]](https://doi.org/10.1038/s41586-025-08690-5)
> [[GitHub]](https://github.com/mahmoodlab/HEST)
> [[HuggingFace]](https://huggingface.co/MahmoodLab/hest-tissue-seg)

The neural segmenter operates at 1 µm/px resolution (≈10×); images are auto-resampled
before inference and the mask is rescaled back to the slide's native resolution. A CUDA
GPU is recommended for practical performance but CPU inference is supported.

No extra packages are required — neural segmentation works with any `torch-gpu` or
`torch-cpu` install:

```bash
uv sync --extra torch-gpu   # or torch-cpu
```

To use it:
```bash
tessellate \
    slide_path=tests/testdata/948176.svs \
    output_h5_path=948176_coord.h5 \
    seg_config.seg_model=neural

tessellate_extract_features \
    slide_path=tests/testdata/948176.svs \
    output_h5_path=948176_feat.h5 \
    output_pt_path=948176_embed.pt \
    model_type=UNI2 \
    seg_config.seg_model=neural
```

### `extract_features`

Use a pathology foundation model to calculate feature embeddings for a slide tiled using
the `tessellate` commaand described above.  This generates both an HDF5 (.h5) file and
a PyTorch (.pt) file, with embeddings for each tile.

The following models are currently supported,

| Model          | model_type    | Access | Reference |
|----------------|---------------|--------|-----------|
| ResNet-50      | RESNET50      | public | https://huggingface.co/microsoft/resnet-50 |
| TransPath      | CTRANSPATH    | local ckpt | https://github.com/Xiyue-Wang/TransPath |
| Prov-GigaPath  | GIGAPATH      | 🔒 gated | https://huggingface.co/prov-gigapath/prov-gigapath |
| Virchow        | VIRCHOW       | 🔒 gated | https://huggingface.co/paige-ai/Virchow |
| Virchow2       | VIRCHOW2      | 🔒 gated | https://huggingface.co/paige-ai/Virchow2 |
| H-Optimus-0    | OPTIMUS       | 🔒 gated | https://huggingface.co/bioptimus/H-optimus-0 |
| H-Optimus-1    | H_OPTIMUS_1   | 🔒 gated | https://huggingface.co/bioptimus/H-optimus-1 |
| H0-mini        | H0_MINI       | 🔒 gated | https://huggingface.co/bioptimus/H0-mini |
| Phikon         | PHIKON        | public | https://huggingface.co/owkin/phikon |
| Phikon-v2      | PHIKON_V2     | public | https://huggingface.co/owkin/phikon-v2 |
| Midnight-12k   | MIDNIGHT12K   | public | https://huggingface.co/kaiko-ai/midnight |
| GPFM           | GPFM          | public   | https://huggingface.co/majiabo/GPFM |
| Hibou-L        | HIBOU_L       | 🔒 gated | https://huggingface.co/histai/hibou-L |
| UNI            | UNI           | 🔒 gated | https://huggingface.co/MahmoodLab/UNI |
| UNI2           | UNI2          | 🔒 gated | https://huggingface.co/MahmoodLab/UNI2-h |
| OpenCLIP       | CLIP          | public | https://github.com/mlfoundations/open_clip |
| GooglePath     | GOOGLEPATH    | 🔒 gated | https://huggingface.co/google/path-foundation |
| Conch v1.5     | CONCH1_5      | 🔒 gated | https://huggingface.co/MahmoodLab/TITAN |
| CONCH v1.0     | CONCH_V1      | 🔒 gated | https://huggingface.co/MahmoodLab/CONCH |
| Kaiko ViT-S/8  | KAIKO_VITS8   | public | https://huggingface.co/1aurent/vit_small_patch8_224.kaiko_ai_towards_large_pathology_fms |
| Kaiko ViT-S/16 | KAIKO_VITS16  | public | https://huggingface.co/1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms |
| Kaiko ViT-B/8  | KAIKO_VITB8   | public | https://huggingface.co/1aurent/vit_base_patch8_224.kaiko_ai_towards_large_pathology_fms |
| Kaiko ViT-B/16 | KAIKO_VITB16  | public | https://huggingface.co/1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms |
| Kaiko ViT-L/14 | KAIKO_VITL14  | public | https://huggingface.co/1aurent/vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms |
| Lunit ViT-S/8  | LUNIT_VITS8   | public | https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino |
| Lunit ViT-S/16 | LUNIT_VITS16  | public | https://huggingface.co/1aurent/vit_small_patch16_224.lunit_dino |
| OpenMidnight   | OPENMIDNIGHT  | 🔒 gated | https://huggingface.co/SophontAI/OpenMidnight |
| GenBio-PathFM  | GENBIO_PATHFM | 🔒 gated | https://huggingface.co/genbio-ai/genbio-pathfm |

**Slide encoders** (require patch-level features as input):

| Model          | model_type      | Patch encoder required | Access |
|----------------|-----------------|------------------------|--------|
| Prov-GigaPath  | GIGAPATH_SLIDE  | GIGAPATH | 🔒 gated |
| TITAN          | TITAN_SLIDE     | CONCH1_5 | 🔒 gated |
| PRISM          | PRISM_SLIDE     | VIRCHOW  | 🔒 gated |
| FEATHER        | FEATHER_SLIDE   | CONCH1_5 | 🔒 gated |
| MADELEINE      | MADELEINE_SLIDE | CONCH1_5 | 🔒 gated |
| CHIEF          | CHIEF_SLIDE     | CTRANSPATH | local ckpt |

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

Most models download automatically from HuggingFace. **🔒 Gated models** require you to visit the model page, sign the access agreement, and set your HuggingFace token:

```bash
export HF_TOKEN=hf_...
```

**Gated models** — visit the link in the table above to request access:

- **Mahmood Lab** (MahmoodLab): UNI, UNI2, CONCH_V1, CONCH1_5, TITAN_SLIDE, FEATHER_SLIDE, MADELEINE_SLIDE
- **Paige AI** (paige-ai): VIRCHOW, VIRCHOW2, PRISM_SLIDE
- **Bioptimus** (bioptimus): OPTIMUS, H_OPTIMUS_1, H0_MINI
- **Prov-GigaPath**: GIGAPATH, GIGAPATH_SLIDE
- **Google**: GOOGLEPATH
- **HistAI**: HIBOU_L
- **SophontAI**: OPENMIDNIGHT
- **GenBio AI**: GENBIO_PATHFM

**Public models** (no token needed): RESNET50, CLIP, PHIKON, PHIKON_V2, MIDNIGHT12K, GPFM, KAIKO_VITS8, KAIKO_VITS16, KAIKO_VITB8, KAIKO_VITB16, KAIKO_VITL14, LUNIT_VITS8, LUNIT_VITS16

**Local-checkpoint-only models**: CTRANSPATH and CHIEF_SLIDE require manually downloaded checkpoints (no HuggingFace download). Pass the checkpoint path via `model_path=`.

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

Supported WSI extensions discovered during directory scan: `.svs`, `.ndpi`, `.tiff`, `.tif`, `.scn`, `.mrxs`, `.vms`, `.vmu`, `.bif`, `.qptiff`, `.czi`.
All `seg_config.*` options (including `seg_model=neural` and `slide_mpp_override`) are
also available on this command; see the [`tessellate` section](#tessellate) above.

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

### `convert`

`convert` converts whole-slide images to pyramidal TIFF format. It supports both
single-file and batch (directory) mode.

**Single file:**
```bash
convert \
    input_path=slide.ndpi \
    output_dir=converted/ \
    mpp=0.25
```

**Batch mode** (directory of slides with an MPP CSV):
```bash
convert \
    input_path=/data/slides/ \
    output_dir=/data/converted/ \
    mpp_csv=slides_mpp.csv \
    num_workers=8
```

The CSV must have columns `wsi` (filename with extension) and `mpp` (microns-per-pixel).
Each input file `<stem>.<ext>` produces `output_dir/<stem>.tiff`. Pass
`bigtiff=true` for files larger than ~4 GB.

| Parameter | Default | Description |
|---|---|---|
| `input_path` | required | Path to a single slide file or a directory of slides. |
| `output_dir` | required | Directory for converted TIFF files (created if absent). |
| `mpp` | — | Microns-per-pixel of the source image. Required for single-file mode. |
| `mpp_csv` | — | CSV with `wsi` and `mpp` columns. Required for batch/directory mode. |
| `downscale_by` | `1` | Integer downsample factor (e.g. `2` converts a 40× slide to 20×). |
| `num_workers` | `1` | Parallel workers for batch mode (`0` = all CPUs). |
| `bigtiff` | `false` | Write BigTIFF format (required for files > ~4 GB). |


