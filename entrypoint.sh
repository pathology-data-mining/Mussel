#!/bin/bash
set -e

# Entrypoint script for Mussel container
# Manages user permissions and runs commands with proper UID/GID

# Get the user ID and group ID from environment or use defaults
USER_ID="${MUSSEL_UID:-1000}"
GROUP_ID="${MUSSEL_GID:-1000}"

# Run user-switching logic whenever the container starts as root with a non-root UID
if [ "$(id -u)" = "0" ] && [ "$USER_ID" != "0" ]; then
  # Validate that USER_ID and GROUP_ID are numeric
  if ! echo "$USER_ID" | grep -qE '^[0-9]+$' || ! echo "$GROUP_ID" | grep -qE '^[0-9]+$'; then
    echo "ERROR: MUSSEL_UID and MUSSEL_GID must be numeric (got UID='$USER_ID' GID='$GROUP_ID')" >&2
    exit 1
  fi

  # Create group if neither the name nor the GID is already in use
  if ! getent group mussel >/dev/null 2>&1 && ! getent group "$GROUP_ID" >/dev/null 2>&1; then
    groupadd -g "$GROUP_ID" mussel
  fi

  # Create user if neither the name nor the UID is already in use
  if ! id -u mussel >/dev/null 2>&1 && ! getent passwd "$USER_ID" >/dev/null 2>&1; then
    useradd -u "$USER_ID" -g "$GROUP_ID" -m -s /bin/bash mussel
  fi

  # Ensure Mussel-specific cache/output dirs are writable; leave /tmp sticky bit intact
  mkdir -p /.cache/mussel /tmp/mussel /tmp/output
  chown -R mussel:mussel /.cache/mussel /tmp/mussel /tmp/output || true
  chmod 777 /.cache/mussel /tmp/mussel /tmp/output || true
  chmod 1777 /tmp || true

  # Ensure Mussel-specific subdirs are writable for optional cache env vars
  if [ -n "$TMPDIR" ]; then
    mkdir -p "$TMPDIR"
    chown mussel:mussel "$TMPDIR" 2>/dev/null || chmod 777 "$TMPDIR" 2>/dev/null || true
  fi
  if [ -n "$TORCH_HOME" ]; then
    mkdir -p "$TORCH_HOME"
    chown -R mussel:mussel "$TORCH_HOME" 2>/dev/null || chmod -R 777 "$TORCH_HOME" 2>/dev/null || true
  fi
  if [ -n "$HF_HOME" ]; then
    mkdir -p "$HF_HOME"
    chown -R mussel:mussel "$HF_HOME" 2>/dev/null || chmod -R 777 "$HF_HOME" 2>/dev/null || true
  fi

  # Switch to the mussel user and execute the command
  exec gosu mussel "$@"
else
  # Not root or UID is 0 — run directly
  exec "$@"
fi
