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
* `clustering_benchmark` - evaluate feature quality by clustering tile-level embeddings and comparing with annotation labels
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

Batch mode groups multiple slides into one `tessellate` invocation and writes one patch
HDF5 per slide. This is intended for workflow engines where many short per-slide jobs
create scheduler overhead:

```bash
tessellate \
    'slide_paths=[slide_a.svs,slide_b.svs]' \
    'slide_ids=[slide_a,slide_b]' \
    output_dir=tiles \
    seg_config=biopsy
```

Use `output_h5_paths=[a.patch.h5,b.patch.h5]` instead of `output_dir` for explicit
per-slide destinations. Batch mode writes patch H5 outputs only; thumbnail, mask,
grid-mask, and tile PNG outputs are supported only in single-slide mode.

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
tessellate slide_path=slide.svs output_h5_path=out.h5 seg_config=stain
```

| Preset | Best for | Key differences from `default` | With `seg_model=neural` |
|---|---|---|---|
| `default` | General use | Baseline values (see table below). | Fully compatible; no warnings. |
| `biopsy` | Needle-core / punch biopsies | Lower area thresholds (`tissue_area_threshold=1`, `hole_area_threshold=1`) to keep small tissue cores; fewer holes (`max_num_holes=2`). | Area thresholds and `max_num_holes` still apply; `segment_threshold`/`median_blur_ksize` are ignored with a warning. |
| `resection` | Surgical resection specimens | Stronger morphological closing (`morphology_ex_kernel=4`) to bridge gaps in large sections; same area thresholds as `default`. | Only `morphology_ex_kernel=4` has effect; `segment_threshold`/`median_blur_ksize` are ignored with a warning. Consider `default seg_config.seg_model=neural seg_config.morphology_ex_kernel=4` to avoid the warning. |
| `tcga` | TCGA whole-slide images | Lower `segment_threshold=8` to capture pale/faded tissue; stronger closing (`morphology_ex_kernel=4`); reduced area thresholds. | `segment_threshold` is ignored with a warning (`median_blur_ksize=7` matches the default so no second warning); `morphology_ex_kernel=4` and area thresholds still apply. |
| `stain` | Fast H&E/IHC stain classification | Neural validation, 32-tile cap, 75% minimum tissue fraction, no contour pruning, and bounded candidate sampling (up to 256 candidates). | Uses the bounded neural path; it does not build a full-slide mask. |

#### Segmentation and patching options

| Parameter | Default | Description |
|---|---|---|
| `seg_config.mpp` | `0.5` | Target resolution for tile extraction (µm/px). |
| `seg_config.patch_size` | `256` | Tile size in pixels at the target MPP. |
| `seg_config.overlap` | `0` | Patch overlap in absolute pixels. Sets `step_size = patch_size - overlap`. |
| `seg_config.min_tissue_proportion` | `0.0` | Per-tile filter: discard tiles where the fraction of tissue pixels is below this value (0.0–1.0). Applied after tiling; `0.1` discards mostly-background edge tiles. |
| `seg_config.selection_mode` | `full_mask` | `full_mask` segments the complete slide; `bounded_neural` proposes candidates cheaply and neural-validates only a bounded set. |
| `seg_config.max_candidate_tiles` | `null` | Maximum neural candidates in `bounded_neural` mode; the `stain` preset sets this to `256`. |
| `seg_config.tissue_area_threshold` | `100` | Full-mask mode only: minimum size of a tissue **region** (contour), in number of tiles. Bounded neural mode performs no contour filtering. |
| `seg_config.hole_area_threshold` | `16` | Full-mask mode only: minimum size of a hole inside a tissue region, in number of tiles. |
| `seg_config.remove_artifacts` | `false` | Enable artifact removal (requires `artifact_remover_fn` hook). |
| `seg_config.remove_penmarks` | `false` | Enable pen-mark removal (requires `artifact_remover_fn` hook). |
| `seg_config.seg_model` | `"classic"` | Segmentation backend: `"classic"` (HSV + fixed threshold), `"otsu"` (HSV + Otsu automatic threshold), or `"neural"` (deep learning; see below). Note: the old `seg_config.use_otsu=true` flag is deprecated — use `seg_model=otsu` instead. |
| `seg_config.slide_mpp_override` | `null` | Override the slide's native MPP; useful when metadata is missing or wrong. |
| `seg_config.max_tiles` | `null` | Optional cap on output tiles after tissue and per-tile filtering. |
| `seg_config.max_tiles_strategy` | `"random"` | How to select tiles when the cap is reached: seeded `"random"` or `"first"`. |
| `seg_config.max_tiles_seed` | `42` | Seed for the random output-tile selection. |

Example with 50% overlap and tissue filtering:
```bash
tessellate \
    slide_path=tests/testdata/948176.svs \
    output_h5_path=948176_coord.h5 \
    seg_config.overlap=128 \
    seg_config.min_tissue_proportion=0.5
```

For stain classification, use the speed-oriented preset:

```bash
tessellate_extract_features \
    slide_path=slide.svs \
    output_h5_path=slide.features.h5 \
    output_pt_path=slide.features.pt \
    model_type=HOPTIMUS0 \
    seg_config=stain
```

The preset stops after 32 tiles with at least 75% neural tissue, or after 256
candidate checks. If fewer than 32 tiles qualify, it returns the qualifying
tiles without relaxing the cutoff. In bounded mode, mask output represents
the accepted tile footprints rather than a complete slide tissue contour. Contour
ID filters are unsupported; `morphology_ex_kernel` is applied to each candidate
mask before its tissue fraction is calculated.

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

Neural model loading and inference are controlled independently through
`neural_config.*` (available on `tessellate`, `tessellate_extract_features`, and
`filter_tessellate`):

| Parameter | Default | Description |
|---|---|---|
| `neural_config.weights_path` | `null` | Local checkpoint path; otherwise download the HEST checkpoint on first use. |
| `neural_config.device` | `"auto"` | PyTorch device (`"auto"`, `"cpu"`, `"cuda"`, or `"cuda:N"`). |
| `neural_config.batch_size` | `8` | Number of 512×512 inference tiles per forward pass. |
| `neural_config.confidence_thresh` | `0.5` | Tissue probability threshold (0–1). |
| `neural_config.max_inference_tiles` | `null` (effective default `4096`) | Fail fast if one slide would require more model tiles; explicit values override `MUSSEL_NEURAL_SEG_MAX_TILES`; set to `0` to disable. |

For example, cap the final HDF5 at 10,000 tiles while limiting neural inference:

```bash
tessellate slide_path=slide.svs output_h5_path=out.h5 \
    seg_config.seg_model=neural seg_config.max_tiles=10000 \
    neural_config.batch_size=16 neural_config.max_inference_tiles=8192
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

`model_kwargs={...}` forwards extra constructor arguments to patch encoders, and
`slide_model_kwargs={...}` forwards them to slide encoders. `TITAN_SLIDE` applies
its GPU OOM patch by default (`patch_oom=true`), which also pins the validated
TITAN revision. Disable it only when testing upstream behavior:

```bash
aggregate_slide_features \
    patch_features_h5_path=patch_features.h5 \
    output_h5_path=slide_features.h5 \
    slide_model_type=TITAN_SLIDE \
    slide_model_kwargs={patch_oom:false}
```

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

#### Embedding precision

By default embeddings are stored at full float32 precision. Use `embedding_precision`
to reduce the on-disk and in-memory size of the HDF5 and `.pt` outputs:

| Value | Bytes / dim | Notes |
|---|---|---|
| `float32` (default) | 4 | Full model precision |
| `float16` | 2 | IEEE half-precision; cuts storage in half with slight loss of precision |
| `bfloat16` | 2 | Brain-float: same exponent range as float32, less mantissa precision than float16; widely used in ML training |

```bash
extract_features \
    slide_path=tests/testdata/948176.svs \
    patch_h5_path=tests/testdata/948176.patch.h5 \
    output_h5_path=948176_feat.h5 \
    output_pt_path=948176_embed.pt \
    embedding_precision=float16
```

The `embedding_precision` parameter is also supported by `tessellate_extract_features`:

```bash
tessellate_extract_features \
    slide_path=slide.svs \
    output_h5_path=out_feat.h5 \
    output_pt_path=out_embed.pt \
    model_type=VIRCHOW2 \
    embedding_precision=bfloat16
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

### `clustering_benchmark`

`clustering_benchmark` evaluates the quality of tile-level feature embeddings by clustering
them and comparing the resulting assignments to annotation labels.  It is designed to work
with the GeoParquet file produced by `merge_annotation_features`.

Outputs three files:

| Output | Default filename | Contents |
|---|---|---|
| Metrics CSV | `clustering_metrics.csv` | Per-algorithm NMI, ARI, purity at tile and slide level |
| Summary JSON | `clustering_results.json` | All scalar metrics as a nested dict |
| UMAP PNG | `umap.png` | UMAP scatter-plot grid (cluster coloring vs annotation coloring for each algorithm) |

#### Example

```bash
clustering_benchmark \
    features_annotation_parquet_path=features_with_annotations.parquet \
    output_metrics_csv=metrics.csv \
    output_summary_json=results.json \
    output_umap_png=umap.png \
    'algorithms=["kmeans","hierarchical","dbscan"]' \
    n_clusters=3 \
    multiclass=true
```

#### Parameters

| Parameter | Default | Description |
|---|---|---|
| `features_annotation_parquet_path` | required | GeoParquet from `merge_annotation_features` (must contain `slide_id`, `annotation`, `overlap_area`, `tile_area`, and `feature_*` columns). |
| `output_metrics_csv` | `clustering_metrics.csv` | Path for per-algorithm metrics table. |
| `output_summary_json` | `clustering_results.json` | Path for nested-dict metrics JSON. |
| `output_umap_png` | `umap.png` | Path for UMAP scatter-plot grid. |
| `annotation_percent_filter_threshold` | `0.50` | Minimum overlap fraction to include a tile. |
| `positive_annotation_label` | `2` | Annotation value treated as the positive class in binary mode. |
| `multiclass` | `false` | Use all non-zero annotation values as class labels (background tiles excluded). |
| `algorithms` | `["kmeans","hierarchical"]` | Clustering algorithms to run. Supported: `"kmeans"`, `"hierarchical"`, `"dbscan"`. |
| `n_clusters` | `2` | Number of clusters for kmeans and hierarchical. |
| `dbscan_eps` | `0.5` | DBSCAN neighbourhood radius. |
| `dbscan_min_samples` | `5` | DBSCAN minimum samples per core point. |
| `umap_n_neighbors` | `15` | UMAP `n_neighbors`. |
| `umap_min_dist` | `0.1` | UMAP `min_dist`. |
| `umap_n_components` | `2` | UMAP output dimensionality: `2` for 2-D plots, `3` for 3-D scatter plots. |
| `umap_subsample` | `10000` | Max tiles used for UMAP projection (random subsample for speed). Use `0` to disable. |
| `random_state` | `42` | Random seed. |

UMAP requires the optional `umap-learn` package:

```bash
uv sync --extra umap
# or
pip install umap-learn
```

### `save_model`

You can download and save a foundation model locally with the `save_model` command.

```bash
save_model model_type=OPTIMUS output_path=optimus.pkl
``` 

### `convert`

`convert` converts whole-slide images and microscopy files to pyramidal GeoTIFF.
It supports both single-file and batch (directory) mode.

#### Supported input formats

| Category | Extensions | Dependency |
|---|---|---|
| WSI scanners | `.svs`, `.ndpi`, `.scn`, `.mrxs`, `.vsi`, `.bif`, `.qptiff` | `aicsimageio[bioformats]` + Java |
| Leica / Zeiss | `.lif`, `.zvi` | `aicsimageio[bioformats]` + Java |
| Zeiss CZI | `.czi` | `pylibCZIrw` |
| TIFF variants | `.tif`, `.tiff`, `.btf`, `.ome.tiff`, `.ome.tif`, `.ome.btf` | `aicsimageio[bioformats]` + Java |
| HDF5 | `.h5`, `.hdf`, `.hdf5`, `.he5` | `aicsimageio[bioformats]` + Java |
| DICOM | `.dicom`, `.dcm` | `aicsimageio[bioformats]` + Java |
| Other scientific | `.ims`, `.ome.xml`, `.pcoraw`, `.jp2`, `.nrrd`, `.fg7` | `aicsimageio[bioformats]` + Java |
| Flat images | `.png`, `.jpg`, `.jpeg` | Pillow (built-in) |

All formats use **pyvips** as the fast streaming path when available.
Bio-Formats formats fall back to **aicsimageio** (requires Java and
`pip install "aicsimageio[bioformats]"`).
CZI files require `pip install pylibCZIrw`.
pyvips is required for writing the output pyramidal TIFF (`pip install pyvips`).

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
