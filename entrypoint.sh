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
    if ! getent group mussel > /dev/null 2>&1; then
        groupadd -g "$GROUP_ID" mussel
    fi
    
    # Create user if it doesn't exist
    if ! id -u mussel > /dev/null 2>&1; then
        useradd -u "$USER_ID" -g "$GROUP_ID" -m -s /bin/bash mussel
    fi
    
    # Ensure writable cache directories
    mkdir -p /.cache /tmp
    chown -R mussel:mussel /.cache /tmp || true
    chmod -R 777 /.cache /tmp
    
    # Switch to the created user and execute the command
    exec gosu mussel "$@"
else
    # Not root or no custom UID requested - run directly
    exec "$@"
fi
