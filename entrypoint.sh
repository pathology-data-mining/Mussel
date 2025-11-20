#!/bin/bash
set -e

# Entrypoint script for Mussel container
# Manages user permissions and runs commands with proper UID/GID

# Get the user ID and group ID from environment or use defaults
USER_ID="${MUSSEL_UID:-1000}"
GROUP_ID="${MUSSEL_GID:-1000}"

# Configure temp storage intelligently
# Priority: 1. VM temp disk (fast local SSD), 2. Azure Files (network), 3. /tmp (OS disk)
UNIQUE_ID="${AZ_BATCH_TASK_ID:-$(hostname)}"

if [ -d "/mnt/resource" ]; then
  # Use VM temporary disk (64GB local SSD on A100) - FAST!
  export TMPDIR="/mnt/resource/tmp/${UNIQUE_ID}"
  export TEMP="/mnt/resource/tmp/${UNIQUE_ID}"
  export TMP="/mnt/resource/tmp/${UNIQUE_ID}"
  export TORCH_HOME="/mnt/resource/cache/torch"
  export PYTORCH_KERNEL_CACHE_PATH="/mnt/resource/cache/pytorch_kernels"
  
  mkdir -p "$TMPDIR" "$TORCH_HOME" "$PYTORCH_KERNEL_CACHE_PATH"
  chmod -R 777 /mnt/resource/tmp /mnt/resource/cache 2>/dev/null || true
  
elif [ -d "/mnt/batch/tasks/fsmounts/azfiles" ]; then
  # Fallback to Azure Files if temp disk not available (slower but works)
  export TMPDIR="/mnt/batch/tasks/fsmounts/azfiles/tmp/${UNIQUE_ID}"
  export TEMP="/mnt/batch/tasks/fsmounts/azfiles/tmp/${UNIQUE_ID}"
  export TMP="/mnt/batch/tasks/fsmounts/azfiles/tmp/${UNIQUE_ID}"
  export TORCH_HOME="/mnt/batch/tasks/fsmounts/azfiles/cache/torch/${UNIQUE_ID}"
  export PYTORCH_KERNEL_CACHE_PATH="/mnt/batch/tasks/fsmounts/azfiles/cache/pytorch_kernels/${UNIQUE_ID}"
  
  mkdir -p "$TMPDIR" "$TORCH_HOME" "$PYTORCH_KERNEL_CACHE_PATH"
  chmod -R 777 /mnt/batch/tasks/fsmounts/azfiles/tmp /mnt/batch/tasks/fsmounts/azfiles/cache 2>/dev/null || true
else
  # Final fallback to /tmp if nothing else is available
  export TMPDIR="/tmp/${UNIQUE_ID}"
  export TEMP="/tmp/${UNIQUE_ID}"
  export TMP="/tmp/${UNIQUE_ID}"
  export TORCH_HOME="/tmp/cache/torch"
  export PYTORCH_KERNEL_CACHE_PATH="/tmp/cache/pytorch_kernels"
  
  mkdir -p "$TMPDIR" "$TORCH_HOME" "$PYTORCH_KERNEL_CACHE_PATH"
  chmod -R 777 /tmp/${UNIQUE_ID} /tmp/cache 2>/dev/null || true
fi

# Use persistent model cache on batch node (shared across all tasks)
# This directory is created by the start task and persists across all tasks in the pool
if [ -d "/mnt/batch_models" ]; then
  export HF_HOME="/mnt/batch_models/huggingface"
  export TRANSFORMERS_CACHE="/mnt/batch_models/transformers"
  export HF_HUB_CACHE="/mnt/batch_models/huggingface/hub"
  mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"
  chmod -R 777 /mnt/batch_models 2>/dev/null || true
fi

# Create a user with the specified UID/GID if running as root and custom UID is requested
if [ "$(id -u)" = "0" ] && [ "$USER_ID" != "0" ]; then
  # Create group if it doesn't exist
  if ! getent group mussel >/dev/null 2>&1; then
    groupadd -g "$GROUP_ID" mussel
  fi

  # Create user if it doesn't exist
  if ! id -u mussel >/dev/null 2>&1; then
    useradd -u "$USER_ID" -g "$GROUP_ID" -m -s /bin/bash mussel
  fi

  # Ensure writable cache directories
  mkdir -p /.cache /tmp
  chown -R mussel:mussel /.cache /tmp || true
  chmod -R 777 /.cache /tmp

  # Ensure output directory exists and is writable
  mkdir -p /tmp/output
  chmod 777 /tmp/output || true
  
  # Ensure all temp and cache dirs are writable by mussel user
  if [ -n "$TMPDIR" ]; then
    chown -R mussel:mussel "$TMPDIR" 2>/dev/null || chmod -R 777 "$TMPDIR" 2>/dev/null || true
  fi
  if [ -n "$TORCH_HOME" ]; then
    chown -R mussel:mussel "$TORCH_HOME" 2>/dev/null || chmod -R 777 "$TORCH_HOME" 2>/dev/null || true
  fi
  if [ -n "$HF_HOME" ]; then
    chown -R mussel:mussel "$HF_HOME" 2>/dev/null || chmod -R 777 "$HF_HOME" 2>/dev/null || true
  fi

  # Switch to the created user and execute the command
  exec gosu mussel "$@"
else
  # Not root or no custom UID requested - run directly
  exec "$@"
fi
