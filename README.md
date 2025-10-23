# Mussel

[![CI](https://github.com/pathology-data-mining/Mussel/actions/workflows/ci.yml/badge.svg)](https://github.com/pathology-data-mining/Mussel/actions/workflows/ci.yml)
[![Docker](https://github.com/pathology-data-mining/Mussel/actions/workflows/docker.yml/badge.svg)](https://github.com/pathology-data-mining/Mussel/actions/workflows/docker.yml)

**Mussel** is a comprehensive toolkit for computational pathology on whole-slide images (WSI). It provides efficient tools for tiling, feature extraction using pathology foundation models, and zero-shot tissue classification.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
  - [Native Installation](#native-installation)
  - [Docker Installation](#docker-installation)
- [Quick Start](#quick-start)
- [Command-Line Interface](#command-line-interface)
- [Development Notes](#development-notes)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Reference](#reference)

## Overview

This is a fork of Faisal Mahmood's [CLAM repository](https://github.com/mahmoodlab/CLAM) (GPL v3 license), enhanced with modern pathology foundation models and streamlined for high-throughput processing.

## Key Features

- **Multiple Foundation Models**: Support for ResNet-50, TransPath, Prov-GigaPath, Virchow, Virchow2, H-Optimus-0, OpenCLIP (QuiltNet), GooglePath, and Conch v1.5
- **Zero-Shot Classification**: Annotate tissue tiles using natural language descriptions without training
- **Flexible Tiling**: Microns per pixel (mpp) specification for tiling, independent of native slide resolution
- **Efficient Processing**: Optimized for batch processing and job submission systems
- **Caching Support**: Fast tile access for I/O-intensive operations like training
- **Multi-GPU Support**: Scale feature extraction across multiple GPUs

## Installation

You can install Mussel either natively on your system or use Docker for a containerized environment.

### Docker Installation

**Recommended for:** Easy setup, reproducible environments, deployment on servers/HPC clusters.

#### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed and running
- [NVIDIA Docker runtime](https://github.com/NVIDIA/nvidia-docker) (for GPU support)

#### Quick Start with Docker

1. Clone the repository:
```bash
git clone https://github.com/pathology-data-mining/Mussel.git
cd Mussel
```

2. Build the Docker image:
```bash
./mussel-docker build
# Or using Make: make build
```

3. Run Mussel commands:
```bash
./mussel-docker tessellate slide_path=slide.svs output_h5_path=tiles.h5
./mussel-docker extract_features slide_path=slide.svs patch_h5_path=tiles.h5 output_h5_path=features.h5
```

For detailed Docker usage instructions, see [README-docker.md](README-docker.md).

### Native Installation

### System Requirements

**Supported Operating Systems:**
* Linux (x86_64) - CPU and GPU support
* macOS (x86 and ARM/Apple Silicon) - CPU only

**Hardware Requirements:**
* GPU (recommended): NVIDIA GPU with CUDA support for fast feature extraction
* CPU: Modern multi-core processor (minimum 4 cores recommended)
* RAM: At least 16GB recommended for processing large slides
* Storage: Varies by dataset size; whole-slide images can be several GB each

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

PyTorch is required for the following models:

* [ResNet-50](https://huggingface.co/microsoft/resnet-50)
* [TransPath](https://github.com/Xiyue-Wang/TransPath)
* [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath)
* [Virchow](https://huggingface.co/paige-ai/Virchow)
* [H-Optimus-0](https://huggingface.co/bioptimus/H-optimus-0)
* [OpenCLIP](https://github.com/mlfoundations/open_clip)


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

#### Docker

Pre-built Docker images are available on Docker Hub with different backend configurations:

```bash
# Pull the latest torch-gpu image (default)
# Note: Replace YOUR_DOCKERHUB_USERNAME with the actual DockerHub organization/username
docker pull YOUR_DOCKERHUB_USERNAME/mussel:latest-torch-gpu

# Pull specific version
docker pull YOUR_DOCKERHUB_USERNAME/mussel:v1.1.1-torch-gpu

# Other available backends
docker pull YOUR_DOCKERHUB_USERNAME/mussel:latest-torch-cpu
docker pull YOUR_DOCKERHUB_USERNAME/mussel:latest-tensorflow-gpu
docker pull YOUR_DOCKERHUB_USERNAME/mussel:latest-tensorflow-cpu
```

See [.github/GITHUB_ACTIONS.md](.github/GITHUB_ACTIONS.md) for more details on Docker image tags and building custom images.

## Quick Start

Here's a simple workflow to process a whole-slide image and extract features:

### 1. Install Mussel with PyTorch GPU support
```bash
uv sync --extra torch-gpu
```

### 2. Activate the virtual environment
After installation, activate the virtual environment:
```bash
source .venv/bin/activate
```

Alternatively, you can prefix commands with `uv run` without activating:
```bash
uv run tessellate --help
```

### 3. Tile a whole-slide image
```bash
tessellate \
    slide_path=path/to/your/slide.svs \
    output_h5_path=slide_tiles.h5 \
    seg_config.segment_threshold=0 \
    num_workers=4
```

### 4. Extract features using a foundation model
```bash
extract_features \
    slide_path=path/to/your/slide.svs \
    patch_h5_path=slide_tiles.h5 \
    model_type=CLIP \
    output_h5_path=slide_features.h5 \
    output_pt_path=slide_features.pt
```

### 5. Annotate tiles with tissue types (zero-shot)
```bash
# Create embeddings for your tissue types
create_class_embeddings \
    classes='["tumor","stroma","necrosis","lymphocytes"]' \
    output_pt_path=class_embeddings.pt

# Annotate tiles
annotate \
    features_pt_path=slide_features.pt \
    class_embedding_pt_path=class_embeddings.pt \
    classes='["tumor","stroma","necrosis","lymphocytes"]' \
    output_csv_path=annotations.csv
```

For more detailed examples and command options, see [README-commands.md](README-commands.md).

#### TensorFlow
TensorFlow is required to run the Google Path Foundation model,

* [Google Path Foundation](https://huggingface.co/google/path-foundation)

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

## Contributing

We welcome contributions from the community! Whether you want to:

- Report a bug
- Suggest a new feature
- Add support for a new foundation model
- Improve documentation
- Submit a bug fix

Please see our [CONTRIBUTING.md](CONTRIBUTING.md) guide for detailed information on how to contribute.

For information about GitHub Actions CI/CD pipelines, see [.github/GITHUB_ACTIONS.md](.github/GITHUB_ACTIONS.md).

Quick contribution steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run tests and linting
5. Submit a pull request

For questions or discussions, please open an issue or start a discussion on GitHub.

### Modifying Package Requirements

* Use `uv sync --extra <extra-deps>` to install this project and its dependencies into the project's virtual environment,
  where <extra-deps> is one of `torch-gpu`, `tensorflow-gpu`, `torch-cpu`, or `tensorflow-cpu`
* Execute `uv sync --extra <extra-deps>` after making any changes to the requirements.

```bash
uv sync --extra torch-gpu
```

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

## Troubleshooting

### Installation Issues

**Issue**: `uv: command not found`
- **Solution**: Install uv using the command provided in the Pre-requisites section. Make sure to restart your terminal after installation.

**Issue**: CUDA/GPU not detected
- **Solution**: 
  - Verify your NVIDIA drivers are installed: `nvidia-smi`
  - Ensure you installed the `torch-gpu` extra: `uv sync --extra torch-gpu`
  - Check that CUDA is available in Python:
    ```python
    import torch
    print(torch.cuda.is_available())
    ```

**Issue**: Conflicting dependencies between PyTorch and TensorFlow
- **Solution**: Choose either PyTorch OR TensorFlow installation, not both. They cannot be installed in the same environment.

### Runtime Issues

**Issue**: Out of memory errors during feature extraction
- **Solution**: 
  - Reduce batch size: Add `batch_size=32` (or lower) to your command
  - Reduce number of workers: Add `num_workers=4` (or lower)
  - For very large slides, process on a machine with more RAM or use a smaller tile size

**Issue**: HuggingFace gated model access denied (Prov-GigaPath, GooglePath, Virchow)
- **Solution**: 
  1. Visit the model page on HuggingFace
  2. Sign the access agreement
  3. Generate an access token from your HuggingFace account settings
  4. Set the token: `export HF_TOKEN=your_token_here`

**Issue**: Slide file format not supported
- **Solution**: Mussel uses tiffslide for reading slides. Supported formats include .svs, .tif, .tiff, and other formats supported by the OpenSlide library.

**Issue**: Command not found after installation
- **Solution**: Make sure your virtual environment is activated:
  ```bash
  source .venv/bin/activate
  ```
  Or use `uv run` before commands: `uv run tessellate --help`



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
