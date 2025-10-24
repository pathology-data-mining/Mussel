# Docker Usage Guide

This guide explains how to use Mussel with Docker for easy deployment and reproducible environments.

## Overview

The Docker setup provides:
- Pre-configured environment with all dependencies
- Support for both GPU and CPU execution
- Easy switching between PyTorch and TensorFlow backends
- Seamless CLI access through a wrapper script

## Quick Start

### 1. Build the Docker Image

**Option 1: Using the wrapper script (Recommended)**

Build with GPU support (default):
```bash
./mussel-docker build
```

Build with CPU-only support:
```bash
MUSSEL_BACKEND=torch-cpu ./mussel-docker build
```

Build with TensorFlow GPU support:
```bash
MUSSEL_BACKEND=tensorflow-gpu ./mussel-docker build
```

**Option 2: Using Make**

```bash
# Build with PyTorch GPU (default)
make build

# Build with PyTorch CPU
make build-cpu

# Build with TensorFlow GPU
make build-tf

# Build with TensorFlow CPU
make build-tf-cpu

# See all available commands
make help
```

### 2. Run Mussel Commands

Once the image is built, you can run any Mussel command using the `mussel-docker` wrapper:

```bash
# Tile a whole-slide image
./mussel-docker tessellate slide_path=slide.svs output_h5_path=tiles.h5

# Extract features
./mussel-docker extract_features \
    slide_path=slide.svs \
    patch_h5_path=tiles.h5 \
    model_type=CLIP \
    output_h5_path=features.h5

# Get help for any command
./mussel-docker tessellate --help
```

### 3. Interactive Shell

Start an interactive shell inside the container:
```bash
./mussel-docker shell
```

This is useful for:
- Running multiple commands
- Debugging
- Exploring the data
- Manual processing steps

## Environment Variables

Configure the Docker wrapper using environment variables:

### `MUSSEL_DOCKER_IMAGE`
Docker image name to use
- **Default:** `mussel:latest`
- **Example:** `MUSSEL_DOCKER_IMAGE=mussel:v1.0 ./mussel-docker tessellate --help`

### `MUSSEL_BACKEND`
Backend to use for building the image
- **Options:** `torch-gpu`, `torch-cpu`, `tensorflow-gpu`, `tensorflow-cpu`
- **Default:** `torch-gpu`
- **Example:** `MUSSEL_BACKEND=torch-cpu ./mussel-docker build`

### `MUSSEL_USE_GPU`
Enable GPU support when running commands
- **Options:** `true`, `false`
- **Default:** `true`
- **Example:** `MUSSEL_USE_GPU=false ./mussel-docker extract_features ...`

### `MUSSEL_WORK_DIR`
Working directory to mount into the container
- **Default:** Current directory (`$(pwd)`)
- **Example:** `MUSSEL_WORK_DIR=/path/to/data ./mussel-docker tessellate ...`

## Working with Files

The wrapper script automatically mounts your current working directory into the container at `/data`. This means:

- All file paths in commands should be relative to your current directory
- Output files will be written to your current directory with your user's ownership
- You can access any files in subdirectories

**Important:** The Docker container runs with your user ID and group ID, ensuring that all files created by the container have the correct ownership on the host system. This prevents permission issues when accessing output files.

### Example Workflow

```bash
# Create a project directory
mkdir my-project
cd my-project

# Put your slide file here
cp /path/to/slide.svs .

# Run commands (they will access files in the current directory)
../mussel-docker tessellate slide_path=slide.svs output_h5_path=tiles.h5
../mussel-docker extract_features \
    slide_path=slide.svs \
    patch_h5_path=tiles.h5 \
    output_h5_path=features.h5

# All output files are in the current directory
ls -lh
```

## Advanced Usage

### Custom Docker Image Name

Build with a custom image name:
```bash
MUSSEL_DOCKER_IMAGE=my-mussel:custom MUSSEL_BACKEND=torch-gpu ./mussel-docker build
```

Use the custom image:
```bash
MUSSEL_DOCKER_IMAGE=my-mussel:custom ./mussel-docker tessellate --help
```

### Mount Multiple Directories

If you need to access files from different directories, use the interactive shell:

```bash
docker run --rm -it \
    --user $(id -u):$(id -g) \
    -v /path/to/slides:/slides \
    -v /path/to/output:/output \
    --gpus all \
    mussel:latest \
    /bin/bash
```

Then inside the container:
```bash
uv run tessellate slide_path=/slides/slide.svs output_h5_path=/output/tiles.h5
```

**Note:** The `--user $(id -u):$(id -g)` flag ensures files are created with your user's ownership.

### GPU Configuration

Check if GPUs are available:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 nvidia-smi
```

Run with specific GPUs:
```bash
docker run --rm --gpus '"device=0,1"' \
    --user $(id -u):$(id -g) \
    -v $(pwd):/data -w /data \
    mussel:latest \
    uv run extract_features ...
```

### Building for Different Backends

Build separate images for different backends:

```bash
# Build PyTorch GPU image
MUSSEL_BACKEND=torch-gpu MUSSEL_DOCKER_IMAGE=mussel:torch-gpu ./mussel-docker build

# Build TensorFlow CPU image
MUSSEL_BACKEND=tensorflow-cpu MUSSEL_DOCKER_IMAGE=mussel:tf-cpu ./mussel-docker build

# Use specific image
MUSSEL_DOCKER_IMAGE=mussel:torch-gpu ./mussel-docker extract_features ...
```

## Troubleshooting

### Docker Not Found
```
Error: Docker is not installed or not in PATH
```
**Solution:** Install Docker from https://docs.docker.com/get-docker/

### Docker Daemon Not Running
```
Error: Docker daemon is not running
```
**Solution:** Start Docker Desktop or the Docker service

### GPU Not Available
```
Warning: GPU support requested but not available, running on CPU
```
**Solution:** 
1. Install NVIDIA Docker runtime: https://github.com/NVIDIA/nvidia-docker
2. Verify GPU access: `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`
3. If GPUs aren't needed, use CPU backend: `MUSSEL_USE_GPU=false ./mussel-docker ...`

### Image Not Found
```
Docker image 'mussel:latest' not found
```
**Solution:** Build the image first: `./mussel-docker build`

### Permission Denied
```
permission denied while trying to connect to the Docker daemon socket
```
**Solution:** 
- Add your user to the docker group: `sudo usermod -aG docker $USER`
- Log out and back in
- Or run with sudo: `sudo ./mussel-docker ...`

### File Not Found Errors

If commands can't find your files:
1. Make sure you're running the command from the directory containing your files
2. Use relative paths, not absolute paths
3. Or set `MUSSEL_WORK_DIR` to the directory containing your files

## Examples

### Complete Workflow Example

```bash
# Setup
mkdir ~/mussel-workspace
cd ~/mussel-workspace
cp /path/to/slides/*.svs .

# Build image (first time only)
/path/to/Mussel/mussel-docker build

# Process slides
for slide in *.svs; do
    base=$(basename "$slide" .svs)
    
    # Tessellate
    /path/to/Mussel/mussel-docker tessellate \
        slide_path="$slide" \
        output_h5_path="${base}_tiles.h5"
    
    # Extract features
    /path/to/Mussel/mussel-docker extract_features \
        slide_path="$slide" \
        patch_h5_path="${base}_tiles.h5" \
        model_type=CLIP \
        output_h5_path="${base}_features.h5"
done

# Annotate with tissue types
/path/to/Mussel/mussel-docker create_class_embeddings \
    classes='["tumor","stroma","necrosis"]' \
    output_pt_path=class_embeddings.pt

for features in *_features.pt; do
    base=$(basename "$features" _features.pt)
    /path/to/Mussel/mussel-docker annotate \
        features_pt_path="$features" \
        class_embedding_pt_path=class_embeddings.pt \
        classes='["tumor","stroma","necrosis"]' \
        output_csv_path="${base}_annotations.csv"
done
```

### CPU-Only Processing

For machines without GPUs:

```bash
# Build CPU-only image
MUSSEL_BACKEND=torch-cpu MUSSEL_DOCKER_IMAGE=mussel:cpu ./mussel-docker build

# Run without GPU
MUSSEL_DOCKER_IMAGE=mussel:cpu MUSSEL_USE_GPU=false ./mussel-docker tessellate \
    slide_path=slide.svs \
    output_h5_path=tiles.h5
```

## Best Practices

1. **Build once, run many times:** Build the Docker image once, then use it for multiple runs
2. **Use relative paths:** Keep your data in a project directory and use relative paths
3. **Choose the right backend:** Build separate images for PyTorch vs TensorFlow if you need both
4. **Monitor resources:** Use `docker stats` to monitor resource usage
5. **Clean up:** Remove old images with `docker image prune` to save space

## Integration with HPC/Cluster Systems

The Docker wrapper can be integrated with job schedulers like SLURM:

```bash
#!/bin/bash
#SBATCH --job-name=mussel
#SBATCH --gpus=1
#SBATCH --time=4:00:00

module load singularity  # or docker

# Convert Docker image to Singularity if needed
# singularity pull mussel.sif docker://mussel:latest

# Run with Singularity
singularity exec --nv mussel.sif \
    uv run extract_features \
    slide_path=/data/slide.svs \
    patch_h5_path=/data/tiles.h5 \
    output_h5_path=/data/features.h5
```

For more information, see the main [README.md](README.md) and [README-commands.md](README-commands.md).
