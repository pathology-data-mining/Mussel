# Azure Batch GPU Image Fix

## Problem Discovery

The `microsoft-dsvm/ubuntu-hpc/2204` image does NOT have NVIDIA drivers installed!

```
nvidia-smi: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

## Solution

Use Azure Batch's GPU-enabled container image instead:

```python
publisher = 'microsoft-azure-batch'
offer = 'ubuntu-server-container'  
sku = '20-04-lts'  # Has GPU drivers pre-installed
```

This image has:
- ✅ NVIDIA drivers
- ✅ nvidia-container-toolkit
- ✅ Docker with GPU support

## Alternative: Install Drivers in Start Task

If we want to use ubuntu-hpc for Generation 2 support, we'd need:

```bash
start_task = """
  # Install NVIDIA drivers
  apt-get update
  apt-get install -y nvidia-driver-535 nvidia-container-toolkit
  systemctl restart docker
  docker pull mskmind/mussel:latest
  usermod -aG docker _azbatch
"""
```

But this adds 5-10 minutes to node startup time.

## Recommendation

Use `microsoft-azure-batch/ubuntu-server-container/20-04-lts` which:
- Has drivers pre-installed
- Starts faster  
- Is designed for Azure Batch GPU workloads
- Supports V100 (Gen 1) 

For A100 (Gen 2), we need to install drivers OR find a Gen 2 image with GPU support.
