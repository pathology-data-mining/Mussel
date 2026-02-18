"""
Model caching utilities with file locking to prevent concurrent download clashes.

This module provides safe model downloading with file-based locking to ensure that
when multiple tasks try to download the same model simultaneously, only one actually
downloads it while the others wait.

Cross-platform support:
- Unix/Linux/macOS: Uses fcntl for robust file locking
- Windows: Uses msvcrt for file locking
- Fallback: No-op if neither available (with warning)
"""

import logging
import os
import platform
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Platform-specific file locking imports
_SYSTEM = platform.system()
_HAS_FCNTL = False
_HAS_MSVCRT = False

if _SYSTEM in ("Linux", "Darwin", "FreeBSD", "OpenBSD"):
    # Unix-like systems
    try:
        import fcntl
        _HAS_FCNTL = True
    except ImportError:
        logger.warning("fcntl not available on this system, file locking disabled")
elif _SYSTEM == "Windows":
    # Windows systems
    try:
        import msvcrt
        _HAS_MSVCRT = True
    except ImportError:
        logger.warning("msvcrt not available on Windows, file locking disabled")
else:
    logger.warning(
        f"Unknown platform '{_SYSTEM}', file locking disabled. "
        "Concurrent model downloads may conflict."
    )


def _acquire_lock(file_handle, blocking=True):
    """
    Acquire an exclusive lock on a file (cross-platform).
    
    Args:
        file_handle: Open file object
        blocking: If True, wait for lock. If False, raise if unavailable.
    
    Raises:
        BlockingIOError: If non-blocking and lock unavailable
    """
    if _HAS_FCNTL:
        # Unix-like: use fcntl
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        fcntl.flock(file_handle.fileno(), flags)
    elif _HAS_MSVCRT:
        # Windows: use msvcrt
        mode = msvcrt.LK_NBLCK if not blocking else msvcrt.LK_LOCK
        try:
            msvcrt.locking(file_handle.fileno(), mode, 1)
        except OSError as e:
            # msvcrt raises OSError for lock conflicts, convert to BlockingIOError
            if not blocking and e.errno in (13, 33):  # Permission denied, lock violation
                raise BlockingIOError("Lock is held by another process") from e
            raise
    else:
        # No locking available - log warning
        if not hasattr(_acquire_lock, "_warned"):
            logger.warning(
                "File locking not available on this platform. "
                "Concurrent model downloads may conflict."
            )
            _acquire_lock._warned = True


def _release_lock(file_handle):
    """
    Release an exclusive lock on a file (cross-platform).
    
    Args:
        file_handle: Open file object
    """
    if _HAS_FCNTL:
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
    elif _HAS_MSVCRT:
        msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
    # If no locking available, this is a no-op


@contextmanager
def model_download_lock(
    model_name: str, cache_dir: Optional[str] = None, timeout: int = 600
):
    """
    Context manager that provides file-based locking for model downloads.

    This ensures that when multiple processes try to download the same model,
    only one actually performs the download while others wait.
    
    Cross-platform support:
    - Unix/Linux/macOS: Uses fcntl for robust file locking
    - Windows: Uses msvcrt for file locking
    - Other platforms: No locking (warning logged)

    Args:
        model_name: Name/identifier of the model being downloaded
        cache_dir: Directory to store lock files (defaults to HF_HOME or /tmp)
        timeout: Maximum time to wait for lock in seconds (default: 600)

    Yields:
        bool: True if this process acquired the lock (should download),
              False if another process already completed the download

    Example:
        >>> with model_download_lock("prov-gigapath/prov-gigapath") as should_download:
        ...     if should_download:
        ...         model = timm.create_model("hf-hub:prov-gigapath/prov-gigapath")
    """
    # Determine cache directory
    if cache_dir is None:
        cache_dir = (
            os.environ.get("HF_HOME") 
            or os.environ.get("TRANSFORMERS_CACHE") 
            or os.environ.get("TMPDIR")
            or "/tmp"
        )

    # Create locks directory
    locks_dir = Path(cache_dir) / ".locks"
    locks_dir.mkdir(parents=True, exist_ok=True)

    # Create a safe filename for the lock
    safe_name = model_name.replace("/", "_").replace(":", "_")
    lock_file = locks_dir / f"{safe_name}.lock"
    done_file = locks_dir / f"{safe_name}.done"

    # Check if download is already complete
    if done_file.exists():
        logger.info(f"Model {model_name} already downloaded (found {done_file})")
        yield False
        return

    # Open lock file
    lock_fd = open(lock_file, "w")

    try:
        logger.info(f"Acquiring lock for model {model_name}...")
        start_time = time.time()

        # Try to acquire exclusive lock with timeout
        while True:
            try:
                _acquire_lock(lock_fd, blocking=False)
                # Lock acquired!
                logger.info(f"Lock acquired for {model_name}")

                # Check again if another process completed while we were waiting
                if done_file.exists():
                    logger.info(f"Model {model_name} was downloaded by another process")
                    yield False
                else:
                    # We should download
                    logger.info(f"Downloading {model_name}...")
                    yield True

                    # Mark as complete
                    done_file.touch()
                    logger.info(f"Download complete for {model_name}")

                break

            except BlockingIOError:
                # Lock is held by another process
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    logger.warning(
                        f"Timeout waiting for lock on {model_name} after {elapsed:.1f}s"
                    )
                    # Proceed anyway - better to duplicate download than fail
                    yield True
                    break

                # Wait and retry
                if elapsed % 30 < 1:  # Log every 30 seconds
                    logger.info(
                        f"Waiting for {model_name} download by another process... ({elapsed:.0f}s)"
                    )
                time.sleep(1)

    finally:
        # Release lock and close file descriptor
        try:
            _release_lock(lock_fd)
        except Exception as exc:
            # Best-effort cleanup: failing to release the lock should not break callers,
            # but we log it for debugging purposes.
            logger.debug(
                "Failed to release file lock for model %s: %s", model_name, exc
            )
        finally:
            # Always close file descriptor, even if unlock fails
            try:
                lock_fd.close()
            except Exception as exc:
                logger.debug(
                    "Failed to close lock file for model %s: %s", model_name, exc
                )


def with_model_cache_lock(download_func):
    """
    Decorator that adds file locking to model download functions.

    The decorated function should accept a `model_name` parameter and
    return the loaded model.

    Example:
        >>> @with_model_cache_lock
        ... def load_timm_model(model_name, **kwargs):
        ...     return timm.create_model(model_name, **kwargs)
    """

    def wrapper(model_name, *args, **kwargs):
        with model_download_lock(model_name) as should_download:
            # Always call the function - if model is cached, it will load instantly
            # The lock just prevents concurrent downloads
            return download_func(model_name, *args, **kwargs)

    wrapper.__name__ = download_func.__name__
    wrapper.__doc__ = download_func.__doc__
    return wrapper
