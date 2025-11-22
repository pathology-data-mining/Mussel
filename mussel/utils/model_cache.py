"""
Model caching utilities with file locking to prevent concurrent download clashes.

This module provides safe model downloading with file-based locking to ensure that
when multiple tasks try to download the same model simultaneously, only one actually
downloads it while the others wait.
"""

import fcntl
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@contextmanager
def model_download_lock(
    model_name: str, cache_dir: Optional[str] = None, timeout: int = 600
):
    """
    Context manager that provides file-based locking for model downloads.

    This ensures that when multiple processes try to download the same model,
    only one actually performs the download while others wait.

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
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
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
        # Release lock
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_fd.close()


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
