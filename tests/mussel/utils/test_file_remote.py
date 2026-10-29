"""Remote-path handling in mussel.utils.file: save_hdf5 append, model cache
downloads, resolve_remote_paths and azfiles:// detection.

Remote storage is an in-memory fsspec filesystem (memory://): nothing touches
real S3/Azure.
"""
import os
from types import SimpleNamespace

import fsspec
import h5py
import numpy as np
import pytest

import mussel.utils.file as file_mod
from mussel.utils.file import (check_unique_slide_ids, download_model_path,
                               is_remote_path, resolve_remote_paths, save_hdf5)


@pytest.fixture
def memfs(monkeypatch):
    """Treat memory:// as a remote path served by a fresh in-memory filesystem."""
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    fs.pseudo_dirs.append("")
    real_is_remote = file_mod._is_remote_path
    monkeypatch.setattr(
        file_mod, "_is_remote_path",
        lambda p: isinstance(p, str) and (p.startswith("memory://") or real_is_remote(p)),
    )
    monkeypatch.setattr(file_mod, "_get_fsspec_filesystem", lambda path, ssl_verify=True: fs)
    yield fs
    fs.store.clear()


def _read_remote_h5(fs, path, tmp_path):
    local = tmp_path / "downloaded.h5"
    fs.get(path, str(local))
    with h5py.File(local, "r") as f:
        return {k: f[k][:] for k in f}, {k: dict(f[k].attrs) for k in f}


# --- save_hdf5 to a remote path ---------------------------------------------


def test_save_hdf5_remote_append_keeps_earlier_batches(memfs, tmp_path):
    attr_src = tmp_path / "patch.h5"
    with h5py.File(attr_src, "w") as f:
        f.create_dataset("coords", data=np.zeros((1, 2)))
        f["coords"].attrs["patch_size"] = 256
    out = "memory://bucket/slide.features.h5"

    save_hdf5(out, {"features": np.ones((3, 4), np.float32), "coords": np.zeros((3, 2), np.int64)},
              attr_h5_path=str(attr_src), mode="w")
    save_hdf5(out, {"features": np.full((2, 4), 2, np.float32), "coords": np.ones((2, 2), np.int64)},
              attr_h5_path=str(attr_src), mode="a")

    data, attrs = _read_remote_h5(memfs, out, tmp_path)
    assert data["features"].shape == (5, 4)
    np.testing.assert_array_equal(data["features"][:3], 1)
    np.testing.assert_array_equal(data["features"][3:], 2)
    assert data["coords"].shape == (5, 2)
    assert attrs["coords"]["patch_size"] == 256  # copied on the first write


def test_save_hdf5_remote_append_creates_missing_file(memfs, tmp_path):
    out = "memory://bucket/new.h5"
    save_hdf5(out, {"features": np.ones((2, 3), np.float32)}, mode="a")
    save_hdf5(out, {"features": np.ones((1, 3), np.float32)}, mode="a")
    data, _ = _read_remote_h5(memfs, out, tmp_path)
    assert data["features"].shape == (3, 3)


def test_save_hdf5_remote_write_mode_replaces(memfs, tmp_path):
    out = "memory://bucket/w.h5"
    save_hdf5(out, {"features": np.ones((4, 3), np.float32)}, mode="w")
    save_hdf5(out, {"features": np.ones((1, 3), np.float32)}, mode="w")
    data, _ = _read_remote_h5(memfs, out, tmp_path)
    assert data["features"].shape == (1, 3)


def test_save_hdf5_zero_width_features(tmp_path):
    """A slide with no tiles is written as (0, 0) features (zero-width dims can't be chunked as-is)."""
    out = tmp_path / "empty.h5"
    save_hdf5(str(out), {"features": np.empty((0, 0), np.float32), "coords": np.empty((0, 2), np.int64)},
              mode="w")
    with h5py.File(out, "r") as f:
        assert f["features"].shape == (0, 0)
        assert f["coords"].shape == (0, 2)


# --- download_model_path ------------------------------------------------------


def test_download_model_path_failure_leaves_no_cache_entry(memfs, tmp_path, monkeypatch):
    memfs.pipe("memory://models/enc/weights.pt", b"complete weights")
    real_get = memfs.get
    calls = {"n": 0}

    def flaky_get(src, dst, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            with open(dst, "wb") as f:
                f.write(b"partial")
            raise IOError("connection reset")
        return real_get(src, dst, **kwargs)

    monkeypatch.setattr(memfs, "get", flaky_get)
    cache = tmp_path / "cache"
    with pytest.raises(IOError):
        download_model_path("memory://models/enc/weights.pt", cache_dir=str(cache))
    models_dir = cache / "remote_models"
    assert os.listdir(models_dir) == []  # neither a partial "cache hit" nor a temp file

    local = download_model_path("memory://models/enc/weights.pt", cache_dir=str(cache))
    with open(local, "rb") as f:
        assert f.read() == b"complete weights"
    assert os.listdir(models_dir) == [os.path.basename(local)]


def test_download_model_path_directory_lands_atomically(memfs, tmp_path):
    memfs.pipe("memory://models/dir/config.json", b"{}")
    memfs.pipe("memory://models/dir/model.bin", b"bin")
    cache = tmp_path / "cache"
    local = download_model_path("memory://models/dir/", cache_dir=str(cache))
    assert sorted(os.listdir(local)) == ["config.json", "model.bin"]
    assert os.listdir(cache / "remote_models") == [os.path.basename(local)]
    # Cache hit: no second download.
    memfs.rm("memory://models/dir/", recursive=True)
    assert download_model_path("memory://models/dir/", cache_dir=str(cache)) == local


# --- azfiles:// ---------------------------------------------------------------


def test_azfiles_is_remote():
    assert is_remote_path("azfiles://share/models/model.pth")
    assert not is_remote_path("/local/azfiles/model.pth")


# --- resolve_remote_paths -----------------------------------------------------


def test_resolve_remote_paths_skips_outputs(monkeypatch):
    downloaded = []

    def fake_download(path, ssl_verify=True):
        downloaded.append(path)
        return "/cache/" + path.split("://", 1)[1].replace("/", "_")

    monkeypatch.setattr(file_mod, "download_model_path", fake_download)
    cfg = SimpleNamespace(
        model_path="s3://bucket/model.pt",
        classifier_pkl="azfiles://share/clf.pkl",
        patch_h5_path="s3://bucket/in.patch.h5",
        output_h5_path="s3://bucket/out.features.h5",
        output_pt_path="s3://bucket/out.features.pt",
        output_dir="s3://bucket/out/",
        intermediate_h5_path="s3://bucket/out.patch_features.h5",
        timings_path="s3://bucket/timings.jsonl",
        model_save_path="s3://bucket/saved_model.pt",
    )

    @resolve_remote_paths()
    def run(c):
        return c

    run(cfg)
    assert sorted(downloaded) == ["azfiles://share/clf.pkl", "s3://bucket/in.patch.h5", "s3://bucket/model.pt"]
    assert cfg.model_path.startswith("/cache/")
    assert cfg.output_h5_path == "s3://bucket/out.features.h5"
    assert cfg.output_pt_path == "s3://bucket/out.features.pt"
    assert cfg.output_dir == "s3://bucket/out/"
    assert cfg.intermediate_h5_path == "s3://bucket/out.patch_features.h5"
    assert cfg.timings_path == "s3://bucket/timings.jsonl"
    assert cfg.model_save_path == "s3://bucket/saved_model.pt"


def test_resolve_remote_paths_explicit_output_attr_still_resolved(monkeypatch):
    monkeypatch.setattr(file_mod, "download_model_path", lambda p, ssl_verify=True: "/cache/x")
    cfg = SimpleNamespace(output_dir="s3://bucket/out/")

    @resolve_remote_paths("output_dir", auto_detect=False)
    def run(c):
        return c

    assert run(cfg).output_dir == "/cache/x"


# --- slide IDs ----------------------------------------------------------------


def test_check_unique_slide_ids():
    assert check_unique_slide_ids(("a", "b")) == ["a", "b"]
    with pytest.raises(ValueError, match=r"duplicated: \['S1'\]"):
        check_unique_slide_ids(["S1", "S2", "S1"])
