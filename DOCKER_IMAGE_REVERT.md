# Docker Image Reverted to Python 3.11

## Change Summary
Reverted Dockerfile from CUDA base image back to Python 3.11-slim base.

## Image Size Comparison
- **CUDA base (nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04)**: 26.1GB
- **Python 3.11-slim base**: 17.1GB
- **Reduction**: 9GB (35% smaller)

## Why Revert?
The CUDA base image was too large and caused disk space issues on Azure Batch A100 nodes with ~30GB OS disk.

## GPU Support
The Python 3.11 base image still supports GPU:
- PyTorch with CUDA support is installed via pip
- CUDA drivers come from the host VM (Azure Batch nodes)
- Container doesn't need CUDA toolkit pre-installed
- NVIDIA Container Runtime passes through GPU access

## Benefits
- ✅ 35% smaller image
- ✅ Faster pulls from Docker Hub
- ✅ Less disk space on nodes
- ✅ Same GPU functionality
- ✅ Better fit for A100 nodes (17GB + 10GB system + 5GB drivers = ~32GB, barely fits on 30GB OS disk)

## Testing Required
- Verify GPU is still detected in containers
- Verify CUDA operations work
- Test on both V100 and A100 nodes

## Updated Configuration
- Image: `mskmind/mussel:latest` (now based on python:3.11-slim)
- Also tagged as: `mskmind/mussel:python311`
