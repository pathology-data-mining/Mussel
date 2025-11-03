#!/bin/bash
set -e

# Docker entrypoint script for Mussel
# Handles user/group creation and permission management for output files

# Get the user and group IDs from environment or use defaults
USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}

# Create a group with the specified GID if it doesn't exist
if ! getent group mussel-user > /dev/null 2>&1; then
    groupadd -g "$GROUP_ID" mussel-user
fi

# Create a user with the specified UID if it doesn't exist
if ! getent passwd mussel-user > /dev/null 2>&1; then
    useradd -u "$USER_ID" -g "$GROUP_ID" -m -s /bin/bash mussel-user
fi

# Ensure the user owns the virtual environment and cache directories
chown -R mussel-user:mussel-user /app/.venv || true
chown -R mussel-user:mussel-user /.cache || true

# Ensure /data directory is writable
if [ -d /data ]; then
    # Don't change ownership of /data itself (it's mounted from host)
    # But ensure it's writable
    chmod a+w /data || true
fi

# Execute the command as the mussel-user
exec gosu mussel-user "$@"
