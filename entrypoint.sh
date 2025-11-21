#!/bin/bash
set -e

# Entrypoint script for Mussel container
# Manages user permissions and runs commands with proper UID/GID

# Get the user ID and group ID from environment or use defaults
USER_ID="${MUSSEL_UID:-1000}"
GROUP_ID="${MUSSEL_GID:-1000}"

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

  # Ensure output directory exists and is writable (Azure Batch expects this)
  if [ -n "$AZ_BATCH_TASK_WORKING_DIR" ]; then
    mkdir -p "$AZ_BATCH_TASK_WORKING_DIR/output"
    chmod -R 777 "$AZ_BATCH_TASK_WORKING_DIR/output" || true
  fi
  mkdir -p /tmp/output
  chmod 777 /tmp/output || true
  
  # Ensure cache directories are writable by mussel user if they're set
  if [ -n "$TMPDIR" ] && [ -d "$TMPDIR" ]; then
    chown -R mussel:mussel "$TMPDIR" 2>/dev/null || chmod -R 777 "$TMPDIR" 2>/dev/null || true
  fi
  if [ -n "$TORCH_HOME" ] && [ -d "$TORCH_HOME" ]; then
    chown -R mussel:mussel "$TORCH_HOME" 2>/dev/null || chmod -R 777 "$TORCH_HOME" 2>/dev/null || true
  fi
  if [ -n "$HF_HOME" ] && [ -d "$HF_HOME" ]; then
    chown -R mussel:mussel "$HF_HOME" 2>/dev/null || chmod -R 777 "$HF_HOME" 2>/dev/null || true
  fi

  # Switch to the created user and execute the command
  exec gosu mussel "$@"
else
  # Not root or no custom UID requested - run directly
  exec "$@"
fi
