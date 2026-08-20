# Mussel

This is a fork of Faisal Mahmood's [CLAM repository](https://github.com/mahmoodlab/CLAM)
 (GPL v3 license), with a handful of modifications:
- Added additional foundation models for generating embeddings
- Added zero-shot tissue-type annotation of tiles
- Added caching of images for inference right on the tiles (rather than on embeddings)
- Added microns per pixel (mpp) as parameter for tiling, supported regardless of native slide resolution
- Made usable for job submission (one script run, one slide)
- Removed modeling
- Updated the tiling algorithm

## Installation

### System requirements

Supported systems:
* Mac OS (x86 and ARM) (cpu only)
* Linux (x86) (cpu and gpu)

### Supported slide formats

Mussel reads whole-slide images via [tiffslide](https://github.com/Bayer-Group/tiffslide)
(backed by [tifffile](https://github.com/cgohlke/tifffile)).
The following formats are supported:

| Extension | Format | Scanner / Vendor | Tiffslide support |
|-----------|--------|-----------------|-------------------|
| `.svs` | Aperio SVS | Leica (Aperio) | ✅ Full |
| `.scn` | Leica SCN | Leica | ✅ Full |
| `.tif` / `.tiff` | TIFF, BigTIFF, OME-TIFF | Generic / various | ✅ Full |
| `.ndpi` | Hamamatsu NDPI | Hamamatsu | ⚠️ Partial — see notes |
| `.bif` | Ventana BIF | Roche (Ventana) | ⚠️ Partial — see notes |
| `.mrxs` | MIRAX | 3DHISTECH | ⚠️ Generic TIFF — see notes |
| `.vms` / `.vmu` | Hamamatsu VMS / VMU | Hamamatsu | ⚠️ Generic TIFF — see notes |
| `.qptiff` | PerkinElmer / Akoya QPTIFF | PerkinElmer / Akoya | ⚠️ Generic TIFF — see notes |
| `.czi` | Carl Zeiss CZI | Zeiss | ⚠️ Generic TIFF — see notes |

**Format support notes:**

- **SVS, SCN, TIFF/BigTIFF/OME-TIFF** — fully supported; tiffslide parses vendor metadata
  and reliably populates `tiffslide.mpp-x`.
- **NDPI** — tiffslide's Hamamatsu parser is marked "only partially implemented"; MPP
  is read from standard TIFF resolution tags (`tiff.XResolution` / `tiff.ResolutionUnit`).
  Most Hamamatsu scanners embed resolution in TIFF tags, so this works in practice.
  If MPP is wrong or missing, use `seg_config.slide_mpp_override`.
- **BIF** — tiffslide has no special Ventana parser; falls back to generic TIFF tag reading.
  Use `seg_config.slide_mpp_override` if MPP is not found automatically.
- **MRXS** — tiffslide uses generic TIFF parsing. MRXS is a multi-file format: the `.mrxs`
  file must be accompanied by its sidecar directory (same name, no extension) in the same
  location; moving only the `.mrxs` file will cause a read error.
- **VMS / VMU** — older Hamamatsu pyramid formats; treated as generic TIFF. These formats
  are uncommon on modern scanners; test before relying on them in production.
- **QPTIFF** — PerkinElmer/Akoya format; treated as generic TIFF. Multiplex (multi-channel)
  QPTIFF files are supported for tiling but feature extraction uses the first channel only.
- **CZI** — Zeiss format; tifffile provides CZI support. Multi-series CZI files (multiple
  acquisitions in one file) are supported but only the first series (index 0) is used.

**MPP (microns per pixel) retrieval** — Mussel reads MPP from slide metadata
using the following fallback chain:
1. `slide_mpp_override` CLI parameter — if provided, used directly; all metadata reading is skipped
2. `tiffslide.mpp-x` — standard property populated by tiffslide for all supported formats
3. `aperio.MPP` / `openslide.mpp-x` — legacy vendor property names
4. `tiff.XResolution` + `tiff.ResolutionUnit` — raw TIFF resolution tags converted to µm/px;
   tiffslide exposes these for partially-supported formats (NDPI, BIF, MRXS, QPTIFF, CZI)
   even when it cannot normalize them to `tiffslide.mpp-x`
5. Magnification-based estimate: scans `aperio.AppMag`, `openslide.objective-power`,
   and `tiffslide.objective-power`; computes MPP as `10.0 / magnification`
6. Configurable default (0.5 µm/px, typical for 20× TCGA slides) with a warning log

When slides lack MPP metadata and the default 0.5 µm/px doesn't match the actual
scanner resolution, pass the known value explicitly:
```bash
tessellate slide_path=slide.svs seg_config.slide_mpp_override=1.0 ...
tessellate_extract_features slide_path=slide.svs seg_config.slide_mpp_override=0.25 ...
export_tiles slide_path=slide.svs slide_mpp_override=0.5 ...
```

### Pre-requisites
- [uv](https://docs.astral.sh/uv/)
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

### Create virtual environment and install packages

Model inference may require either PyTorch or TensorFlow, depending on which 
foundation models you wish to use.  Because it can be challenging to satisfy the dependencies
for both of those at the same time, you need to choose whether to install the module for
PyTorch or for TensorFlow.

In addition, you can choose to install Mussel with or without GPU support.  GPUs are
necessary to run model inference for feature extraction or for generating class embeddings,
but other operations can just run on cpus.  (Technically, model inference can just run on
cpus, as well, but it's very slow.)

#### PyTorch

Install PyTorch support first, then models are downloaded automatically on first use:

```bash
uv sync --extra torch-gpu   # GPU (CUDA) — recommended
uv sync --extra torch-cpu   # CPU only (Mac or CPU-only Linux)
```

PyTorch is required for the following patch encoders:

| Model | `model_type` | Access | HuggingFace |
|---|---|---|---|
| ResNet-50 | `RESNET50` | public | built-in (torchvision) |
| TransPath | `CTRANSPATH` | public | [Xiyue-Wang/TransPath](https://github.com/Xiyue-Wang/TransPath) |
| OpenCLIP | `CLIP` | public | [wisdomik/QuiltNet-B-16-PMB](https://huggingface.co/wisdomik/QuiltNet-B-16-PMB) |
| Phikon | `PHIKON` | public | [owkin/phikon](https://huggingface.co/owkin/phikon) |
| Phikon-v2 | `PHIKON_V2` | public | [owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2) |
| Midnight-12k | `MIDNIGHT12K` | public | [kaiko-ai/midnight](https://huggingface.co/kaiko-ai/midnight) |
| Prov-GigaPath | `GIGAPATH` | 🔒 gated | [prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) |
| Virchow | `VIRCHOW` | 🔒 gated | [paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow) |
| Virchow2 | `VIRCHOW2` | 🔒 gated | [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) |
| H-Optimus-0 | `OPTIMUS` | 🔒 gated | [bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) |
| H-Optimus-1 | `H_OPTIMUS_1` | 🔒 gated | [bioptimus/H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) |
| H0-mini | `H0_MINI` | 🔒 gated | [bioptimus/H0-mini](https://huggingface.co/bioptimus/H0-mini) |
| UNI | `UNI` | 🔒 gated | [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) |
| UNI2 | `UNI2` | 🔒 gated | [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) |
| CONCH v1.5 | `CONCH1_5` | 🔒 gated | [MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN) |
| GPFM | `GPFM` | public | [majiabo/GPFM](https://huggingface.co/majiabo/GPFM) |
| Hibou-L | `HIBOU_L` | 🔒 gated | [histai/hibou-L](https://huggingface.co/histai/hibou-L) |
| CONCH v1.0 | `CONCH_V1` | 🔒 gated | [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH) |
| Kaiko ViT-S/8 | `KAIKO_VITS8` | public | [1aurent/vit_small_patch8_224.kaiko_ai_towards_large_pathology_fms](https://huggingface.co/1aurent/vit_small_patch8_224.kaiko_ai_towards_large_pathology_fms) |
| Kaiko ViT-S/16 | `KAIKO_VITS16` | public | [1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms](https://huggingface.co/1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms) |
| Kaiko ViT-B/8 | `KAIKO_VITB8` | public | [1aurent/vit_base_patch8_224.kaiko_ai_towards_large_pathology_fms](https://huggingface.co/1aurent/vit_base_patch8_224.kaiko_ai_towards_large_pathology_fms) |
| Kaiko ViT-B/16 | `KAIKO_VITB16` | public | [1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms](https://huggingface.co/1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms) |
| Kaiko ViT-L/14 | `KAIKO_VITL14` | public | [1aurent/vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms](https://huggingface.co/1aurent/vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms) |
| Lunit DINO ViT-S/8 | `LUNIT_VITS8` | public | [1aurent/vit_small_patch8_224.lunit_dino](https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino) |
| Lunit DINO ViT-S/16 | `LUNIT_VITS16` | public | [1aurent/vit_small_patch16_224.lunit_dino](https://huggingface.co/1aurent/vit_small_patch16_224.lunit_dino) |
| OpenMidnight | `OPENMIDNIGHT` | 🔒 gated | [SophontAI/OpenMidnight](https://huggingface.co/SophontAI/OpenMidnight) |
| GenBio-PathFM | `GENBIO_PATHFM` | 🔒 gated | [genbio-ai/genbio-pathfm](https://huggingface.co/genbio-ai/genbio-pathfm) |

And the following slide encoders (aggregate patch features into a single slide embedding):

| Model | `model_type` | Patch encoder | Access | HuggingFace |
|---|---|---|---|---|
| Prov-GigaPath | `GIGAPATH_SLIDE` | `GIGAPATH` | 🔒 gated | [prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) |
| TITAN | `TITAN_SLIDE` | `CONCH1_5` | 🔒 gated | [MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN) |
| PRISM | `PRISM_SLIDE` | `VIRCHOW` | 🔒 gated | [paige-ai/Prism](https://huggingface.co/paige-ai/Prism) |
| FEATHER | `FEATHER_SLIDE` | `CONCH1_5` | 🔒 gated | [MahmoodLab/abmil.base.conch_v15.pc108-24k](https://huggingface.co/MahmoodLab/abmil.base.conch_v15.pc108-24k) |
| MADELEINE | `MADELEINE_SLIDE` | `CLIP` | 🔒 gated | [MahmoodLab/madeleine](https://huggingface.co/MahmoodLab/madeleine) |
| CHIEF | `CHIEF_SLIDE` | `CTRANSPATH` | ⬇ access req. | [hms-dbmi/CHIEF](https://github.com/hms-dbmi/CHIEF) |

**🔒 Gated models** require signing an access agreement on the HuggingFace model page and setting your token:
```bash
export HF_TOKEN=hf_...
```

**⬇ Models requiring access request** are downloaded automatically once access is granted:

| Model | Request access | Notes |
|---|---|---|
| CHIEF (`CHIEF_SLIDE`) | [Google Drive folder](https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV) | Request via [hms-dbmi/CHIEF](https://github.com/hms-dbmi/CHIEF); `gdown` downloads automatically on first use |

TransPath (`CTRANSPATH`) and CHIEF (`CHIEF_SLIDE`) are downloaded automatically via `gdown` on first use (cached in the HuggingFace hub cache directory).

GenBio-PathFM (`GENBIO_PATHFM`) downloads its model architecture code from GitHub on first use and caches it at `~/.cache/mussel/genbio_pathfm/`. The model weights are downloaded from HuggingFace (requires a token with access to `genbio-ai/genbio-pathfm`).

OpenMidnight (`OPENMIDNIGHT`) uses the DINOv2 ViT-G/14 architecture from the `facebookresearch/dinov2` torch.hub repository. On first use, Mussel downloads the repository code and caches it at `~/.cache/torch/hub/facebookresearch_dinov2_main/`. The model weights are downloaded from HuggingFace (requires a token with access to `SophontAI/OpenMidnight`).

#### TensorFlow

TensorFlow is required for GooglePath only:

| Model | `model_type` | Access | HuggingFace |
|---|---|---|---|
| GooglePath | `GOOGLEPATH` | 🔒 gated | [google/path-foundation](https://huggingface.co/google/path-foundation) |

```bash
uv sync --extra tensorflow-gpu   # GPU (CUDA)
uv sync --extra tensorflow-cpu   # CPU only (e.g. Mac)
```

#### Neural segmentation (`seg_model="neural"`)

Mussel includes built-in neural tissue segmentation using a **DeepLabV3-ResNet50**
model (2-class: tissue vs background) trained on histopathology slides as part of the
[HEST](https://github.com/mahmoodlab/HEST) project at the Mahmood Lab, Harvard Medical
School.

Pre-trained weights are hosted at
[MahmoodLab/hest-tissue-seg](https://huggingface.co/MahmoodLab/hest-tissue-seg) on
HuggingFace and are downloaded automatically on first use (no account or token
required). The model operates at 1 µm/px; Mussel handles resampling automatically.

> **Reference:** Chan *et al.*, "A Pathology Foundation Model for Cancer Diagnosis and
> Prognosis Prediction", *Nature* 2025.
> [[paper]](https://doi.org/10.1038/s41586-025-08690-5)
> [[GitHub]](https://github.com/mahmoodlab/HEST)
> [[HuggingFace model card]](https://huggingface.co/MahmoodLab/hest-tissue-seg)

No extra packages are required — it works with any `torch-gpu` or `torch-cpu` install:

```bash
uv sync --extra torch-gpu
```

Then pass `seg_config.seg_model=neural` to `tessellate` or
`tessellate_extract_features`. A CUDA GPU is recommended for practical
performance but CPU inference is supported.

Neural runtime controls are available under `neural_config.*` (weights path,
device, batch size, confidence threshold, and `max_inference_tiles`) for
`tessellate`, `tessellate_extract_features`, and `filter_tessellate`. An explicit
neural device takes precedence over the workflow's `use_gpu` setting; `auto`
uses that setting. Use
`seg_config.max_tiles` with `seg_config.max_tiles_strategy` and
`seg_config.max_tiles_seed` to cap and reproducibly sample the final output tiles.

For stain classification, use `seg_config=stain`. This speed-oriented preset
uses bounded neural validation, keeps up to 32 tiles with at least 75% tissue,
checks at most 256 candidates, and honors `neural_config.max_inference_tiles` as
an additional inference budget. It avoids full-slide neural inference; when
fewer than 32 candidates qualify or the inference budget is reached, it returns
the qualifying tiles without relaxing the tissue cutoff.

## Development Notes

* Any commands executed using `uv run <command...>` are automatically executed in the project environment.
* You can also explicitly activate the virtual environment created by `uv` by executing
```bash
source .venv/bin/activate
```
* To install Mussel into an existing environment, activate that environment and use `uv pip` or `conda` to install
  one of `Mussel[torch-gpu]`, `Mussel[tensorflow-gpu]`, `Mussel[torch-cpu]`, or `Mussel[tensorflow-cpu]`
  into that environment.  (Here, `Mussel` would be replaced with the path to the Mussel
  repo you've checked out.)

(The example commands in README-commands.md all expect you to have a activated python environment, so that `uv run` isn't necessary.)

### Modifying package requirements

* Use `uv sync --extra <extra-deps>` to install this project and its dependencies into the project's virtual environment,
  where <extra-deps> is one of `torch-gpu`, `tensorflow-gpu`, `torch-cpu`, or `tensorflow-cpu`
* Execute `uv sync --extra <extra-deps>` after making any changes to the requirements.

```bash
uv sync --extra torch-gpu
```

### Cloud/Remote slide processing
Mussel can process slides stored on the cloud or remote object stores via the `tiffslide` and `fsspec` packages. In order to properly configure mussel for this use case ensure that you: 
* Install additional packages via `uv sync --extra remote`
* Have a valid cloud profile set up on your machine (e.g. you have an access key and secret key for your profile stored in your `~/.aws/credentials`)
* Have a valid configuration for `fsspec` defined in your configuration in `~/.config/fsspec/` directory (e.g. you have a `~/.config/fsspec/s3.json` file with the profile set to the profile defined in `~/.aws/credentials` and all required `client_kwargs` are specified)


### Run unit tests

Make sure that the dev dependencies are installed. (They should be installed by default.)

```bash
uv run pytest tests
```

### Create conda environment

To install this module into an existing Python environment, activate that environment
and install mussel and its extra dependencies with the command, (for example)
```bash
uv pip install .[torch-gpu]
```

## Command-line interface

Mussel provides a set of CLI tools for tiling whole-slide images, working with tiled
slides, and generating feature embeddings with pathology foundation models.
The tools currently available from Mussel are,

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

These are described, with examples, in the accompanying document, [README-commands.md](README-commands.md)


## Nextflow pipeline

For running Mussel at scale on a compute cluster, see
**[mussel-nf](https://github.com/pathology-data-mining/mussel-nf)** — a Nextflow
pipeline that wraps the Mussel CLI tools and handles job scheduling, parallelism,
and output management across large slide cohorts.

## Development

### Docker/Apptainer Containers with Flash Attention

Mussel supports building Docker containers with **flash-attn 2.0** for accelerated attention in the CONCH1.5 patch encoder and the Prov-GigaPath slide encoder. Flash attention provides ~30-50% speedup on patch encoding (~20% overall TITAN pipeline improvement).

**Building the flash-attn container:**

```bash
# Build Docker image with flash-attn backend
make docker-build BACKEND=fastattn
# Or manually:
docker build --build-arg BACKEND=fastattn -t mussel:fastattn .

# Convert to Apptainer SIF for HPC deployment
make sif
# Or manually:
apptainer build --force mussel-fastattn.sif docker-daemon://mussel:fastattn
```

**Key details:**
- The `[fastattn]` extra in `pyproject.toml` installs:
  - torch 2.11.0+cu126 (CUDA 12.6)
  - flash-attn 2.6.3 (custom manylinux_2_28 wheels for Rocky 8 compatibility)
  - xformers 0.0.35
- Flash attention is required for **CONCH1.5** (patch encoding) and **Prov-GigaPath** (slide encoding); both require CUDA compute capability ≥ 8.0 (A100, H100, etc.)
- For V100 (compute 7.0) or CPU, the code automatically falls back to PyTorch SDPA
- TITAN slide encoder uses `SDPBackend.EFFICIENT_ATTENTION` (Phase 1 optimization); flash-attn accelerates CONCH1.5 patch encoding and GigaPath slide encoding

**Using with mussel-nf:**

Copy the SIF to your mussel-nf repo and use the `apptainer_fastattn` profile:

```bash
cp mussel-fastattn.sif /path/to/mussel-nf/
cd /path/to/mussel-nf
nextflow run main.nf -profile cluster,slurm,apptainer_fastattn ...
```

The profile automatically uses the flash-attn container for `FEATURIZE` tasks.

## License
This code is made available under the GPLv3 License and is available for non-commercial academic purposes.
Forked from CLAM, © [Mahmood Lab](http://www.mahmoodlab.org).

## Reference

Please cite the original CLAM [paper](https://www.nature.com/articles/s41551-020-00682-w):

Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w
```
@article{lu2021data,
  title={Data-efficient and weakly supervised computational pathology on whole-slide images},
  author={Lu, Ming Y and Williamson, Drew FK and Chen, Tiffany Y and Chen, Richard J and Barbieri, Matteo and Mahmood, Faisal},
  journal={Nature Biomedical Engineering},
  volume={5},
  number={6},
  pages={555--570},
  year={2021},
  publisher={Nature Publishing Group}
}
```
