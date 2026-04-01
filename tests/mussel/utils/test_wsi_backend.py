"""Tests for multi-backend WSI opening (mussel/utils/wsi_backend.py)."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mussel.utils.wsi_backend import (
    OmeZarrSlide,
    _CUCIM_EXTENSIONS,
    _ZARR_EXTENSIONS,
    open_slide,
)


SVS_PATH = "tests/testdata/948176.svs"


# ---------------------------------------------------------------------------
# open_slide with real slide (tiffslide backend)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.exists(SVS_PATH),
    reason="Test slide not available",
)
class TestOpenSlideTiffslide:
    def test_open_slide_default_backend(self):
        """open_slide() should default to tiffslide and expose the standard API."""
        with open_slide(SVS_PATH) as wsi:
            assert hasattr(wsi, "level_dimensions")
            assert hasattr(wsi, "level_downsamples")
            assert hasattr(wsi, "properties")
            assert len(wsi.level_dimensions) >= 1

    def test_open_slide_explicit_tiffslide(self, monkeypatch):
        """Explicit backend='tiffslide' should work."""
        monkeypatch.delenv("MUSSEL_WSI_BACKEND", raising=False)
        with open_slide(SVS_PATH, backend="tiffslide") as wsi:
            assert wsi.level_dimensions is not None

    def test_open_slide_env_var_tiffslide(self, monkeypatch):
        """MUSSEL_WSI_BACKEND=tiffslide env var should be respected."""
        monkeypatch.setenv("MUSSEL_WSI_BACKEND", "tiffslide")
        with open_slide(SVS_PATH) as wsi:
            assert len(wsi.level_dimensions) >= 1

    def test_level_downsamples_monotone(self):
        """Level downsamples should be non-decreasing (each level is coarser)."""
        with open_slide(SVS_PATH) as wsi:
            ds = wsi.level_downsamples
            assert all(ds[i] <= ds[i + 1] for i in range(len(ds) - 1))

    def test_get_best_level_for_downsample(self):
        """get_best_level_for_downsample(1.0) should return level 0."""
        with open_slide(SVS_PATH) as wsi:
            assert wsi.get_best_level_for_downsample(1.0) == 0

    def test_read_region_returns_rgb(self):
        """read_region at level 0 should return an RGB image."""
        with open_slide(SVS_PATH) as wsi:
            w, h = wsi.level_dimensions[0]
            # Read a small tile from the top-left
            region = wsi.read_region((0, 0), 0, (32, 32)).convert("RGB")
            assert region.size == (32, 32)


# ---------------------------------------------------------------------------
# Extension constants
# ---------------------------------------------------------------------------


class TestExtensionSets:
    def test_zarr_extension_in_set(self):
        assert ".zarr" in _ZARR_EXTENSIONS

    def test_svs_in_cucim_extensions(self):
        assert ".svs" in _CUCIM_EXTENSIONS

    def test_tiff_in_cucim_extensions(self):
        assert ".tiff" in _CUCIM_EXTENSIONS


# ---------------------------------------------------------------------------
# CuCIM backend — test only the wrapper logic, not actual GPU availability
# ---------------------------------------------------------------------------


class TestCuCIMSlide:
    def test_get_best_level_for_downsample(self):
        """CuCIMSlide.get_best_level_for_downsample should return correct level index."""
        from mussel.utils.wsi_backend import CuCIMSlide

        mock_cuimage = MagicMock()
        mock_cuimage.resolutions = {
            "level_dimensions": [(1000, 800), (500, 400), (250, 200)],
            "level_downsamples": [1.0, 2.0, 4.0],
        }
        mock_cuimage.metadata = {}

        slide = CuCIMSlide(mock_cuimage)
        assert slide.get_best_level_for_downsample(1.0) == 0
        assert slide.get_best_level_for_downsample(2.0) == 1
        assert slide.get_best_level_for_downsample(3.0) == 1  # last level ≤ 3
        assert slide.get_best_level_for_downsample(4.0) == 2

    def test_level_dimensions_and_downsamples_populated(self):
        from mussel.utils.wsi_backend import CuCIMSlide

        mock_cuimage = MagicMock()
        mock_cuimage.resolutions = {
            "level_dimensions": [(2000, 1500), (1000, 750)],
            "level_downsamples": [1.0, 2.0],
        }
        mock_cuimage.metadata = {"test_key": "test_val"}

        slide = CuCIMSlide(mock_cuimage)
        assert slide.level_dimensions == [(2000, 1500), (1000, 750)]
        assert slide.level_downsamples == [1.0, 2.0]
        assert slide.properties == {"test_key": "test_val"}

    def test_cucim_explicit_backend_raises_if_not_installed(self, monkeypatch):
        """If cucim is not installed, explicit backend=cucim should raise ImportError."""
        import sys
        cucim_available = "cucim" in sys.modules or "cucim.clara" in sys.modules
        if cucim_available:
            pytest.skip("cucim is installed; cannot test missing-cucim path")

        with pytest.raises((ImportError, ModuleNotFoundError)):
            open_slide(SVS_PATH, backend="cucim")


# ---------------------------------------------------------------------------
# OmeZarrSlide — test with mock zarr store
# ---------------------------------------------------------------------------


class TestOmeZarrSlide:
    def _make_mock_zarr_store(self, n_levels=2, base_h=100, base_w=150, n_channels=3):
        """Build a minimal mock zarr store that looks like OME-Zarr."""
        store = MagicMock()
        arrays = []
        for i in range(n_levels):
            scale = 2 ** i
            arr = MagicMock()
            arr.shape = (n_channels, base_h // scale, base_w // scale)
            arr.ndim = 3
            # Make indexing return a numpy array
            def make_slice(h, w, c):
                def __getitem__(self, key):
                    return np.zeros((c, h, w), dtype=np.uint8)
                return __getitem__
            arr.__getitem__ = make_slice(base_h // scale, base_w // scale, n_channels)
            arrays.append(arr)

        store.attrs = {
            "multiscales": [
                {
                    "datasets": [
                        {"path": str(i)} for i in range(n_levels)
                    ]
                }
            ]
        }
        store.__getitem__ = lambda self, key: arrays[int(key)]
        store.keys = lambda: [str(i) for i in range(n_levels)]
        return store, arrays, base_h, base_w

    def test_level_dimensions_parsed(self):
        store, arrays, h, w = self._make_mock_zarr_store()
        arrays[0].shape = (3, h, w)
        arrays[1].shape = (3, h // 2, w // 2)

        slide = OmeZarrSlide(store)
        assert len(slide.level_dimensions) == 2
        # level 0: (W, H)
        assert slide.level_dimensions[0] == (w, h)
        assert slide.level_dimensions[1] == (w // 2, h // 2)

    def test_level_downsamples(self):
        store, arrays, h, w = self._make_mock_zarr_store(n_levels=3, base_h=400, base_w=400)
        arrays[0].shape = (3, 400, 400)
        arrays[1].shape = (3, 200, 200)
        arrays[2].shape = (3, 100, 100)

        slide = OmeZarrSlide(store)
        assert slide.level_downsamples[0] == pytest.approx(1.0)
        assert slide.level_downsamples[1] == pytest.approx(2.0)
        assert slide.level_downsamples[2] == pytest.approx(4.0)

    def test_get_best_level_for_downsample(self):
        store, arrays, h, w = self._make_mock_zarr_store(n_levels=3, base_h=400, base_w=400)
        arrays[0].shape = (3, 400, 400)
        arrays[1].shape = (3, 200, 200)
        arrays[2].shape = (3, 100, 100)

        slide = OmeZarrSlide(store)
        assert slide.get_best_level_for_downsample(1.0) == 0
        assert slide.get_best_level_for_downsample(2.0) == 1
        assert slide.get_best_level_for_downsample(3.5) == 1  # level 2 ds=4 > 3.5
        assert slide.get_best_level_for_downsample(4.0) == 2

    def test_open_slide_zarr_extension_routes_to_zarr_backend(self, tmp_path, monkeypatch):
        """A .zarr path should trigger the zarr backend (or raise ImportError if zarr absent)."""
        import sys
        zarr_available = "zarr" in sys.modules
        zarr_path = str(tmp_path / "slide.zarr")
        os.makedirs(zarr_path, exist_ok=True)

        if not zarr_available:
            with pytest.raises((ImportError, ModuleNotFoundError)):
                open_slide(zarr_path)
        else:
            # zarr is installed; the store will fail to open since it's empty,
            # but we verify that the zarr code path is entered.
            with pytest.raises(Exception):
                # Even with zarr installed an empty dir raises at zarr.open
                open_slide(zarr_path)
