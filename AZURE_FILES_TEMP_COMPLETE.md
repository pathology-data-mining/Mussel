# Azure Files Temp Storage - Task-Isolated Implementation âœ…

## Summary
Configured Docker container to use Azure Files mount with **task-specific isolation** to prevent cache collisions between concurrent tasks.

## Key Design Decision: Task Isolation

### Problem
Multiple nodes/tasks writing to the same cache directories can cause:
- Race conditions during PyTorch kernel compilation
- Cache corruption
- File locking issues
- Unpredictable failures

### Solution
Each task gets its own isolated temp and cache directories:

```bash
UNIQUE_ID="${AZ_BATCH_TASK_ID:-$(hostname)}"

# Task-specific (isolated)
TMPDIR="/mnt/batch/tasks/fsmounts/azfiles/tmp/${UNIQUE_ID}"
TORCH_HOME="/mnt/batch/tasks/fsmounts/azfiles/cache/torch/${UNIQUE_ID}"
PYTORCH_KERNEL_CACHE_PATH="/mnt/batch/tasks/fsmounts/azfiles/cache/pytorch_kernels/${UNIQUE_ID}"

# Shared (safe for concurrent reads)
HF_HOME="/mnt/batch/tasks/fsmounts/azfiles/cache/huggingface"
TRANSFORMERS_CACHE="/mnt/batch/tasks/fsmounts/azfiles/cache/transformers"
```

## Cache Strategy

### Task-Specific (No Sharing)
**PyTorch kernel cache** and **temp directories** are isolated per task because:
- CUDA kernel compilation creates temp files
- PyTorch writes to cache during model initialization
- Concurrent writes cause corruption

### Shared (Safe)
**HuggingFace/Transformers cache** is shared because:
- Models are downloaded once and read by all tasks
- Download process has built-in locking
- Only reads after initial download

## Directory Structure

```
/mnt/batch/tasks/fsmounts/azfiles/
b”œâ”€â”€ tmp/
b”‚   â”œâ”€â”€ task-12345/              # Task 1's temp files
b”‚   â”œâ”€â”€ task-67890/              # Task 2's temp files
b”‚   â””â”€â”€ hostname-node1/          # Fallback to hostname
b”œâ”€â”€ cache/
b”‚   â”œâ”€â”€ torch/
b”‚   â”‚   â”œâ”€â”€ task-12345/          # Task 1's PyTorch cache
b”‚   â”‚   â””â”€â”€ task-67890/          # Task 2's PyTorch cache
b”‚   â”œâ”€â”€ pytorch_kernels/
b”‚   â”‚   â”œâ”€â”€ task-12345/          # Task 1's compiled kernels
b”‚   â”‚   â””â”€â”€ task-67890/          # Task 2's compiled kernels
b”‚   â”œâ”€â”€ huggingface/             # SHARED - all tasks
b”‚   â””â”€â”€ transformers/            # SHARED - all tasks
b”œâ”€â”€ slides/                      # Staged inputs
b”œâ”€â”€ models/                      # Staged models
b””â”€â”€ outputs/                     # Task outputs
```

## Disk Space Impact

### OS Disk (30GB on A100 nodes)
- Docker image: 17.1GB
- System + drivers: 10GB
- Task-specific temp: 0GB (on Azure Files)
- **Total: ~27GB** âœ… Fits comfortably

### Azure Files (Per Task)
- Temp files: ~2-5GB per task
- PyTorch cache: ~500MB-1GB per task (initial compilation)
- Shared HF cache: ~2-5GB total (downloaded once)

### Cleanup
Task-specific directories can be cleaned up after task completion, or Azure Batch can auto-cleanup based on retention policies.

## Benefits

1. âœ… **No cache collisions** - Each task isolated
2. âœ… **Solves A100 disk issue** - All temp on Azure Files
3. âœ… **Efficient HF sharing** - Models downloaded once
4. âœ… **Backward compatible** - Falls back to local if no Azure Files
5. âœ… **Automatic** - Uses AZ_BATCH_TASK_ID when available

## Trade-offs

**Pros:**
- Completely safe for concurrent execution
- No race conditions
- Predictable behavior

**Cons:**
- PyTorch kernels recompiled per task (but cached for task duration)
- Slightly more disk space on Azure Files (still cheap/unlimited)
- Each task's cache starts fresh

**Note:** The PyTorch kernel recompilation overhead is minimal (seconds) compared to task runtime (minutes to hours).

## Testing

Verify isolation in container:
```bash
echo $AZ_BATCH_TASK_ID
echo $TMPDIR
ls -la /mnt/batch/tasks/fsmounts/azfiles/tmp/
ls -la /mnt/batch/tasks/fsmounts/azfiles/cache/torch/
```

## Alternative Considered

**Shared cache with locking** - Rejected because:
- Complex to implement correctly
- Still has race condition risks
- PyTorch kernel compilation not fully atomic
- Not worth the complexity for cache reuse
