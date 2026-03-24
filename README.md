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

PyTorch is required for the following patch encoders:

| Model | `model_type` | Access | HuggingFace |
|---|---|---|---|
| ResNet-50 | `RESNET50` | public | built-in (torchvision) |
| TransPath | `CTRANSPATH` | local ckpt | [Xiyue-Wang/TransPath](https://github.com/Xiyue-Wang/TransPath) |
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
| GPFM | `GPFM` | 🔒 gated | [majiabo/GPFM](https://huggingface.co/majiabo/GPFM) |
| Hibou-L | `HIBOU_L` | 🔒 gated | [histai/hibou-L](https://huggingface.co/histai/hibou-L) |

And the following slide encoders (aggregate patch features into a single slide embedding):

| Model | `model_type` | Patch encoder | Access | HuggingFace |
|---|---|---|---|---|
| Prov-GigaPath | `GIGAPATH_SLIDE` | `GIGAPATH` | 🔒 gated | [prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) |
| TITAN | `TITAN_SLIDE` | `CONCH1_5` | 🔒 gated | [MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN) |
| PRISM | `PRISM_SLIDE` | `VIRCHOW` | 🔒 gated | [paige-ai/Prism](https://huggingface.co/paige-ai/Prism) |
| FEATHER | `FEATHER_SLIDE` | `CONCH1_5` | 🔒 gated | [MahmoodLab/abmil.base.conch_v15.pc108-24k](https://huggingface.co/MahmoodLab/abmil.base.conch_v15.pc108-24k) |
| MADELEINE | `MADELEINE_SLIDE` | `CONCH1_5` | 🔒 gated | [MahmoodLab/madeleine](https://huggingface.co/MahmoodLab/madeleine) |
| CHIEF | `CHIEF_SLIDE` | `CTRANSPATH` | local ckpt | — |

**🔒 Gated models** require signing an access agreement on the HuggingFace model page and setting your token:
```bash
export HF_TOKEN=hf_...
```


##### GPU (CUDA)
If you need to run a PyTorch model on GPUs, you can create the Mussel dev environment with
the command
```bash
uv sync --extra torch-gpu
```

##### CPU
If you just want CPU support for a PyTorch model, you can create your Mussel environment with 
```bash
uv sync --extra torch-cpu
```
Mussel doesn't currently support Apple Metal GPUs, so this is what you'd use to install on a modern MacBook.

#### TensorFlow
TensorFlow is required to run the Google Path Foundation model,

| Model | `model_type` | Access | HuggingFace |
|---|---|---|---|
| GooglePath | `GOOGLEPATH` | 🔒 gated | [google/path-foundation](https://huggingface.co/google/path-foundation) |

##### GPU (CUDA)
To run the GooglePath with GPUs, create your dev environment with
```bash
uv sync --extra tensorflow-gpu
```

##### CPU
If you just want CPU support for working with GooglePath, create your Mussel environment with 
```bash
uv sync --extra tensorflow-cpu
```
Again, this is what you'd install on a MacBook running on Apple Silicon.

#### Neural segmentation (`seg_model="neural"`)

Mussel includes built-in neural tissue segmentation using a DeepLabV3 model
(pre-trained weights from `MahmoodLab/hest-tissue-seg` on HuggingFace,
downloaded automatically on first use). No extra packages are required — it
works with any `torch-gpu` or `torch-cpu` install:

```bash
uv sync --extra torch-gpu
```

Then pass `seg_config.seg_model=neural` to `tessellate` or
`tessellate_extract_features`. A CUDA GPU is recommended for practical
performance but CPU inference is supported.

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

Make sure that the dev dependencies are installed. (They should be installed by default).   (Note that the tests in
this repo expect you to have installed the `torch-gpu` version of the project, and only
the default model, `CLIP`, is used for feature extraction.)

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

These are described, with examples, in the accompanying document, [README-commands.md](README-commands.md)


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
