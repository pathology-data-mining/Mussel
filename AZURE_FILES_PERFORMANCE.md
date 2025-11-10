# Azure Files Performance Analysis

## Short Answer: Probably NOT Fast Enough for Heavy Temp Usage

### Azure Files Performance Characteristics

**Azure Files (SMB/NFS)**:
- IOPS: 1,000-100,000 (tier dependent)
- Throughput: 60-300 MB/s per share
- Latency: 10-20ms (network mount)
- Type: Network file system

**Local SSD/NVMe**:
- IOPS: 500,000+ 
- Throughput: 2,000+ MB/s
- Latency: <1ms
- Type: Local disk

**Performance Ratio**: Local disk is **100-1000x faster** for IOPS

## Use Cases Where Azure Files is SLOW

1. **Many small file operations** âŒ
   - Creating/deleting thousands of temp files
   - Each operation has network latency
   - PyTorch kernel compilation (many .o, .so files)

2. **Random I/O** âŒ
   - Database files
   - Swap space
   - Build artifacts

3. **High-frequency reads/writes** âŒ
   - Application logs written continuously
   - Streaming data processing

## Use Cases Where Azure Files is ACCEPTABLE

1. **Large sequential reads/writes** âœ…
   - Slide image files (GB sized)
   - Model weight files
   - Feature output files

2. **Infrequent access** âœ…
   - Model downloads (once per task)
   - Configuration files
   - Final outputs

## Our Use Case Analysis

### What We're Doing Wrong âŒ

**PyTorch cache on Azure Files**:
- Kernel compilation creates many small files
- High IOPS requirement
- Will be SLOW (10-50x slower than local)

**TMPDIR on Azure Files**:
- General temp files for various operations
- Could have many small file operations
- Network latency adds up

### What We Should Do âœ…

**Use local disk for:**
- PyTorch compilation cache
- General temp directory
- Any high-IOPS operations

**Use Azure Files for:**
- Large input files (slides)
- Large output files (features)
- Model weights (read once)

## Recommended Configuration

```bash
# Use LOCAL disk for performance-critical temp/cache
export TMPDIR="/tmp"
export TORCH_HOME="/tmp/torch_cache"
export PYTORCH_KERNEL_CACHE_PATH="/tmp/pytorch_kernels"

# Use Azure Files for data sharing
export HF_HOME="/mnt/batch/tasks/fsmounts/azfiles/cache/huggingface"
export TRANSFORMERS_CACHE="/mnt/batch/tasks/fsmounts/azfiles/cache/transformers"

# Input/output on Azure Files is fine (large files)
# Slides: /mnt/batch/tasks/fsmounts/azfiles/slides/
# Outputs: /mnt/batch/tasks/fsmounts/azfiles/outputs/
```

## The A100 Disk Problem - Better Solution

Instead of moving temp to Azure Files, **use the VM's temporary disk**:

### Azure VM Temporary Disk
- **Location**: `/mnt` or `/mnt/resource` on Linux
- **Size**: 64GB for Standard_NC24ads_A100_v4
- **Performance**: Local SSD (fast!)
- **Type**: Ephemeral (lost on VM restart)
- **Perfect for**: Temp files, cache

### Updated Strategy

```bash
if [ -d "/mnt/resource" ]; then
  # Use fast local temp disk
  export TMPDIR="/mnt/resource/tmp/${UNIQUE_ID}"
  export TORCH_HOME="/mnt/resource/cache/torch"
  export PYTORCH_KERNEL_CACHE_PATH="/mnt/resource/cache/pytorch_kernels"
elif [ -d "/mnt/batch/tasks/fsmounts/azfiles" ]; then
  # Fallback to Azure Files if temp disk not available
  export TMPDIR="/mnt/batch/tasks/fsmounts/azfiles/tmp/${UNIQUE_ID}"
  # ... etc
else
  # Default to /tmp
  export TMPDIR="/tmp"
fi

# Always use Azure Files for shareable data
if [ -d "/mnt/batch/tasks/fsmounts/azfiles" ]; then
  export HF_HOME="/mnt/batch/tasks/fsmounts/azfiles/cache/huggingface"
fi
```

## Performance Impact Estimate

**PyTorch kernel compilation**:
- Local disk: ~5-10 seconds
- Azure Files: ~30-60 seconds (3-6x slower)

**Task overhead per task**: +20-50 seconds on Azure Files

For 1000 tasks: **+5-14 hours total wasted time**

## Recommendation

bŒ **DON'T** use Azure Files for temp/cache
bœ… **DO** use VM temporary disk (`/mnt/resource`) for temp/cache
bœ… **DO** use Azure Files for data sharing (slides, models, outputs)

This gives us:
- Fast temp operations (local SSD)
- Enough space (64GB temp disk)
- OS disk stays clean (only 17GB Docker image + 10GB system = 27GB)
- Data sharing where needed (Azure Files for inputs/outputs)
