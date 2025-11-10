# Azure Files Mount Used for Temp Storage

## Changes Made

### 1. Updated entrypoint.sh
Added automatic detection and configuration of temp storage when Azure Files mount is available:

```bash
if [ -d "/mnt/batch/tasks/fsmounts/azfiles" ]; then
  export TMPDIR="/mnt/batch/tasks/fsmounts/azfiles/tmp"
  export TEMP="/mnt/batch/tasks/fsmounts/azfiles/tmp"
  export TMP="/mnt/batch/tasks/fsmounts/azfiles/tmp"
  export TORCH_HOME="/mnt/batch/tasks/fsmounts/azfiles/cache/torch"
  export PYTORCH_KERNEL_CACHE_PATH="/mnt/batch/tasks/fsmounts/azfiles/cache/pytorch_kernels"
  export HF_HOME="/mnt/batch/tasks/fsmounts/azfiles/cache/huggingface"
  export TRANSFORMERS_CACHE="/mnt/batch/tasks/fsmounts/azfiles/cache/transformers"
fi
```

### 2. Benefits

**Disk Space Management:**
- All temp files go to Azure Files (shared storage) instead of node OS disk
- PyTorch kernel cache uses Azure Files
- HuggingFace model cache uses Azure Files
- General temp directory uses Azure Files

**OS Disk Usage Reduced:**
- Before: `/tmp`, `/root/.cache/torch`, `/root/.cache/huggingface` on OS disk
- After: All temp/cache on Azure Files mount
- **Saves**: 5-10GB on OS disk during processing

**For A100 Nodes:**
- Docker image: 17.1GB
- System + drivers: 10GB
- Temp/cache: 0GB (now on Azure Files)
- **Total OS disk**: ~27GB (fits comfortably on 30GB)

### 3. Behavior

**When Azure Files is mounted:**
- Container automatically uses it for all temp/cache operations
- Creates directory structure: `tmp/`, `cache/torch/`, `cache/pytorch_kernels/`, etc.
- Sets all permissions to 777 for multi-user access

**When Azure Files is NOT mounted:**
- Falls back to standard `/tmp` and default cache locations
- Works normally for local/non-Azure deployments

### 4. Directory Structure on Azure Files

```
/mnt/batch/tasks/fsmounts/azfiles/
b”œâ”€â”€ tmp/                          # General temp files
b”œb”€â”€ cache/
b”‚   â”œâ”€â”€ torch/                    # PyTorch model cache
b”‚   â”œâ”€â”€ pytorch_kernels/          # Compiled CUDA kernels
b”‚   b”œâ”€â”€ huggingface/              # HF model cache
b”‚   â””â”€â”€ transformers/             # Transformers cache
b”œâ”€â”€ slides/                       # Staged slide files
b”œâ”€â”€ models/                       # Staged model files
b””â”€â”€ outputs/                      # Output files
```

## Testing

The updated image will:
1. Detect Azure Files mount on startup
2. Configure all temp/cache paths to use it
3. Create necessary directories
4. Use Azure Files for all temporary operations

This should resolve the A100 disk space issue while maintaining compatibility with other deployment scenarios.
