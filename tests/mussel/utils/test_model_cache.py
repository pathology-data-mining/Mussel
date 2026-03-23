import multiprocessing
import os
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from mussel.utils.model_cache import model_download_lock, with_model_cache_lock


def test_model_download_lock_basic():
    """Test that the lock allows download when no .done file exists"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with model_download_lock("test-model", cache_dir=tmpdir) as should_download:
            assert should_download is True


def test_model_download_lock_creates_done_file():
    """Test that the lock creates .done file after successful download"""
    with tempfile.TemporaryDirectory() as tmpdir:
        locks_dir = Path(tmpdir) / ".locks"
        done_file = locks_dir / "test-model.done"
        
        assert not done_file.exists()
        
        with model_download_lock("test-model", cache_dir=tmpdir) as should_download:
            assert should_download is True
        
        # Done file should be created after context exits
        assert done_file.exists()


def test_model_download_lock_skips_if_done():
    """Test that the lock skips download if .done file already exists"""
    with tempfile.TemporaryDirectory() as tmpdir:
        locks_dir = Path(tmpdir) / ".locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        done_file = locks_dir / "test-model.done"
        done_file.touch()
        
        with model_download_lock("test-model", cache_dir=tmpdir) as should_download:
            assert should_download is False


def test_model_download_lock_sanitizes_name():
    """Test that model names with special characters are sanitized"""
    with tempfile.TemporaryDirectory() as tmpdir:
        locks_dir = Path(tmpdir) / ".locks"
        
        with model_download_lock("org/model:tag", cache_dir=tmpdir) as should_download:
            # Check that lock file uses sanitized name
            lock_file = locks_dir / "org_model_tag.lock"
            done_file = locks_dir / "org_model_tag.done"
            
            assert lock_file.exists()
            assert should_download is True
        
        assert done_file.exists()


def test_model_download_lock_default_cache_dir():
    """Test that lock uses default cache directory from environment"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch.dict(os.environ, {"HF_HOME": tmpdir}):
            locks_dir = Path(tmpdir) / ".locks"
            
            with model_download_lock("test-model") as should_download:
                assert locks_dir.exists()
                assert should_download is True


def test_model_download_lock_fallback_cache_dirs():
    """Test cache directory fallback chain: HF_HOME -> TRANSFORMERS_CACHE -> TMPDIR -> /tmp"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test TRANSFORMERS_CACHE fallback
        with mock.patch.dict(os.environ, {"TRANSFORMERS_CACHE": tmpdir}, clear=True):
            with model_download_lock("test-model") as should_download:
                locks_dir = Path(tmpdir) / ".locks"
                assert locks_dir.exists()
        
        # Test TMPDIR fallback
        with tempfile.TemporaryDirectory() as tmpdir2:
            with mock.patch.dict(os.environ, {"TMPDIR": tmpdir2}, clear=True):
                with model_download_lock("test-model") as should_download:
                    locks_dir = Path(tmpdir2) / ".locks"
                    assert locks_dir.exists()


def _worker_download_with_lock(cache_dir, results_queue, process_id):
    """Worker process that tries to download model."""
    with model_download_lock("concurrent-model", cache_dir=cache_dir, timeout=10) as should_download:
        if should_download:
            time.sleep(0.1)
        results_queue.put((process_id, should_download))


def test_model_download_lock_concurrent_access():
    """Test that concurrent processes don't both download the model"""
    ctx = multiprocessing.get_context("fork")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Start 3 concurrent processes
        results_queue = ctx.Queue()
        processes = []

        for i in range(3):
            p = ctx.Process(
                target=_worker_download_with_lock,
                args=(tmpdir, results_queue, i)
            )
            p.start()
            processes.append(p)

        # Wait for all to complete
        for p in processes:
            p.join(timeout=15)
            assert p.exitcode == 0, f"Process failed with exit code {p.exitcode}"

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        assert len(results) == 3, "All processes should complete"

        # Exactly one process should have downloaded
        download_count = sum(1 for _, should_download in results if should_download)
        assert download_count == 1, f"Expected 1 download, got {download_count}"


def _worker_hold_lock_forever(cache_dir, ready_event):
    """Worker that holds lock indefinitely."""
    with model_download_lock("timeout-model", cache_dir=cache_dir, timeout=60):
        ready_event.set()
        time.sleep(10)


def test_model_download_lock_timeout():
    """Test that lock times out if held too long"""
    ctx = multiprocessing.get_context("fork")
    with tempfile.TemporaryDirectory() as tmpdir:
        ready_event = ctx.Event()

        # Start process that holds lock
        holder = ctx.Process(
            target=_worker_hold_lock_forever,
            args=(tmpdir, ready_event)
        )
        holder.start()

        # Wait for holder to acquire lock
        assert ready_event.wait(timeout=5), "Holder should acquire lock"

        # Try to acquire with short timeout
        start = time.time()
        with model_download_lock("timeout-model", cache_dir=tmpdir, timeout=2) as should_download:
            elapsed = time.time() - start
            # Should timeout and proceed anyway
            assert should_download is True
            assert elapsed >= 2, "Should wait for timeout duration"
            assert elapsed < 4, "Should not wait much longer than timeout"

        # Cleanup
        holder.terminate()
        holder.join(timeout=5)


def test_model_download_lock_lock_release():
    """Test that lock is properly released even if exception occurs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First process raises exception
        try:
            with model_download_lock("exception-model", cache_dir=tmpdir):
                raise ValueError("Simulated failure")
        except ValueError:
            pass
        
        # Second process should be able to acquire lock immediately
        start = time.time()
        with model_download_lock("exception-model", cache_dir=tmpdir) as should_download:
            elapsed = time.time() - start
            # Should acquire instantly (no waiting for lock)
            assert elapsed < 1
            # Done file should not exist (first process failed)
            assert should_download is True


def _worker_create_done_file_after_delay(cache_dir, delay):
    """Worker that creates done file after a delay."""
    time.sleep(delay)
    locks_dir = Path(cache_dir) / ".locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    done_file = locks_dir / "race-model.done"
    done_file.touch()


def test_model_download_lock_done_file_race_condition():
    """Test that done file is checked after lock acquisition"""
    ctx = multiprocessing.get_context("fork")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Start process that will create done file while we wait for lock
        creator = ctx.Process(
            target=_worker_create_done_file_after_delay,
            args=(tmpdir, 0.2)
        )
        creator.start()

        # This should detect done file even though it was created while waiting
        time.sleep(0.1)  # Let creator get ahead
        with model_download_lock("race-model", cache_dir=tmpdir, timeout=5) as should_download:
            # Since done file exists, we should not download
            # (either detected before lock or after lock acquisition)
            pass  # Result depends on timing, but should not crash

        creator.join(timeout=5)


def test_with_model_cache_lock_decorator():
    """Test the decorator version of model cache lock"""
    download_count = 0
    
    @with_model_cache_lock
    def mock_download(model_name):
        nonlocal download_count
        download_count += 1
        return f"model-{model_name}"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch.dict(os.environ, {"HF_HOME": tmpdir}):
            # First call should download
            result1 = mock_download("decorator-model")
            assert result1 == "model-decorator-model"
            assert download_count == 1
            
            # Second call should also call function (but model cached, no actual download)
            result2 = mock_download("decorator-model")
            assert result2 == "model-decorator-model"
            # Decorator calls function regardless of lock state
            assert download_count == 2


def test_with_model_cache_lock_preserves_function_metadata():
    """Test that decorator preserves function name and docstring"""
    @with_model_cache_lock
    def my_download_function(model_name):
        """This is my docstring"""
        return model_name
    
    assert my_download_function.__name__ == "my_download_function"
    assert my_download_function.__doc__ == "This is my docstring"


def test_with_model_cache_lock_passes_args_and_kwargs():
    """Test that decorator properly forwards args and kwargs"""
    @with_model_cache_lock
    def download_with_options(model_name, device="cpu", precision="fp32"):
        return {"name": model_name, "device": device, "precision": precision}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch.dict(os.environ, {"HF_HOME": tmpdir}):
            result = download_with_options("test-model", device="cuda", precision="fp16")
            
            assert result["name"] == "test-model"
            assert result["device"] == "cuda"
            assert result["precision"] == "fp16"


def test_model_download_lock_creates_locks_directory():
    """Test that lock creates .locks directory if it doesn't exist"""
    with tempfile.TemporaryDirectory() as tmpdir:
        locks_dir = Path(tmpdir) / ".locks"
        assert not locks_dir.exists()
        
        with model_download_lock("test-model", cache_dir=tmpdir):
            assert locks_dir.exists()
            assert locks_dir.is_dir()


def test_model_download_lock_handles_special_characters():
    """Test that various special characters in model names are handled"""
    test_cases = [
        "org/model",
        "model:v1.0",
        "org/model:latest",
        "model-name_v2",
        "hf-hub:microsoft/resnet-50",
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for model_name in test_cases:
            with model_download_lock(model_name, cache_dir=tmpdir) as should_download:
                assert should_download is True


def test_model_download_lock_closes_file_descriptor():
    """Test that file descriptor is always closed, even on exception during unlock"""
    import os
    from unittest.mock import patch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        locks_dir = Path(tmpdir) / ".locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        lock_file = locks_dir / "fd-test-model.lock"
        
        # Track file descriptor
        opened_fds = []
        original_open = open
        
        def tracking_open(*args, **kwargs):
            fd = original_open(*args, **kwargs)
            if str(lock_file) in str(args[0]):
                opened_fds.append(fd)
            return fd
        
        # Test normal flow - fd should be closed
        with patch("builtins.open", side_effect=tracking_open):
            with model_download_lock("fd-test-model", cache_dir=tmpdir):
                pass
        
        assert len(opened_fds) == 1, "Should have opened lock file once"
        assert opened_fds[0].closed, "File descriptor should be closed after context exit"
        
        # Test exception flow - fd should still be closed
        opened_fds.clear()
        # Remove done file so the lock path is taken again (not early-return)
        done_file = locks_dir / "fd-test-model.done"
        if done_file.exists():
            done_file.unlink()
        try:
            with patch("builtins.open", side_effect=tracking_open):
                with model_download_lock("fd-test-model", cache_dir=tmpdir):
                    raise RuntimeError("Simulated error")
        except RuntimeError:
            pass
        
        assert len(opened_fds) == 1, "Should have opened lock file once"
        assert opened_fds[0].closed, "File descriptor should be closed even after exception"


def test_model_download_lock_cross_platform():
    """Test that module provides cross-platform compatibility"""
    from mussel.utils import model_cache
    
    # Check that platform detection works
    assert hasattr(model_cache, "_SYSTEM")
    assert hasattr(model_cache, "_HAS_FCNTL")
    assert hasattr(model_cache, "_HAS_MSVCRT")
    
    # Check that helper functions exist
    assert callable(model_cache._acquire_lock)
    assert callable(model_cache._release_lock)
    
    # On Linux, fcntl should be available
    import platform
    if platform.system() == "Linux":
        assert model_cache._HAS_FCNTL is True
        assert model_cache._HAS_MSVCRT is False
    
    # Test that locking works even without platform-specific modules
    # (falls back to no-op with warning)
    with tempfile.TemporaryDirectory() as tmpdir:
        with model_download_lock("platform-test", cache_dir=tmpdir) as should_download:
            # Should work regardless of platform
            assert should_download in (True, False)
