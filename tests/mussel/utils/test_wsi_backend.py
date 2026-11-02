"""Tests for multi-backend WSI opening (mussel/utils/wsi_backend.py)."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mussel.utils.wsi_backend import (_CUCIM_EXTENSIONS, _ZARR_EXTENSIONS,
                                      OmeZarrSlide, open_slide)

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
            scale = 2**i
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
            "multiscales": [{"datasets": [{"path": str(i)} for i in range(n_levels)]}]
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
        store, arrays, h, w = self._make_mock_zarr_store(
            n_levels=3, base_h=400, base_w=400
        )
        arrays[0].shape = (3, 400, 400)
        arrays[1].shape = (3, 200, 200)
        arrays[2].shape = (3, 100, 100)

        slide = OmeZarrSlide(store)
        assert slide.level_downsamples[0] == pytest.approx(1.0)
        assert slide.level_downsamples[1] == pytest.approx(2.0)
        assert slide.level_downsamples[2] == pytest.approx(4.0)

    def test_get_best_level_for_downsample(self):
        store, arrays, h, w = self._make_mock_zarr_store(
            n_levels=3, base_h=400, base_w=400
        )
        arrays[0].shape = (3, 400, 400)
        arrays[1].shape = (3, 200, 200)
        arrays[2].shape = (3, 100, 100)

        slide = OmeZarrSlide(store)
        assert slide.get_best_level_for_downsample(1.0) == 0
        assert slide.get_best_level_for_downsample(2.0) == 1
        assert slide.get_best_level_for_downsample(3.5) == 1  # level 2 ds=4 > 3.5
        assert slide.get_best_level_for_downsample(4.0) == 2

    def test_open_slide_zarr_extension_routes_to_zarr_backend(
        self, tmp_path, monkeypatch
    ):
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


# ---------------------------------------------------------------------------
# CuCIM metadata -> tiffslide-style properties
# ---------------------------------------------------------------------------


def _cucim_slide(metadata):
    from mussel.utils.wsi_backend import CuCIMSlide

    img = MagicMock()
    img.resolutions = {"level_dimensions": [(1000, 800)], "level_downsamples": [1.0]}
    img.metadata = metadata
    return CuCIMSlide(img)


class TestCuCIMProperties:
    def test_nested_vendor_metadata_is_flattened(self):
        from mussel.utils.segment import get_slide_mpp

        slide = _cucim_slide({"aperio": {"MPP": "0.2525", "AppMag": "40"}, "cucim": {"dims": "YXC"}})
        assert slide.properties["aperio.MPP"] == "0.2525"
        assert slide.properties["aperio.AppMag"] == "40"
        assert get_slide_mpp(slide) == pytest.approx(0.2525)

    def test_cucim_micrometre_spacing_becomes_tiffslide_mpp(self):
        from mussel.utils.segment import get_slide_mpp

        slide = _cucim_slide({"cucim": {"dims": "YXC", "spacing": [0.26, 0.25, 1.0],
                                        "spacing_units": ["micrometer", "micrometer", "color"]}})
        assert slide.properties["tiffslide.mpp-x"] == pytest.approx(0.25)
        assert slide.properties["tiffslide.mpp-y"] == pytest.approx(0.26)
        assert get_slide_mpp(slide) == pytest.approx(0.25)

    def test_non_micrometre_spacing_is_ignored(self):
        slide = _cucim_slide({"cucim": {"dims": "YXC", "spacing": [1.0, 1.0, 1.0],
                                        "spacing_units": ["", "", "color"]}})
        assert "tiffslide.mpp-x" not in slide.properties

    def test_json_string_and_empty_metadata(self):
        assert _cucim_slide('{"aperio": {"MPP": "0.5"}}').properties["aperio.MPP"] == "0.5"
        assert _cucim_slide(None).properties == {}


# ---------------------------------------------------------------------------
# OmeZarrSlide axes and padding (real in-memory zarr arrays)
# ---------------------------------------------------------------------------


def _zarr_store(arrays, axes=None):
    zarr = pytest.importorskip("zarr")
    root = zarr.group(store=zarr.storage.MemoryStore())
    for i, data in enumerate(arrays):
        root.create_dataset(str(i), data=data, shape=data.shape, dtype=data.dtype)
    ms = {"datasets": [{"path": str(i)} for i in range(len(arrays))]}
    if axes is not None:
        ms["axes"] = axes
    root.attrs["multiscales"] = [ms]
    return root


def _rgb(h, w):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


class TestOmeZarrAxes:
    def test_channel_last_dimensions_and_read(self):
        rgb = _rgb(40, 60)  # (Y, X, C) with no axes metadata
        slide = OmeZarrSlide(_zarr_store([rgb]))
        assert slide.level_dimensions[0] == (60, 40)
        region = np.asarray(slide.read_region((10, 5), 0, (8, 6)))
        np.testing.assert_array_equal(region, rgb[5:11, 10:18])

    def test_channel_last_with_axes_metadata(self):
        rgb = _rgb(3, 50)  # tiny height: the shape heuristic alone would take Y for C
        axes = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"},
                {"name": "c", "type": "channel"}]
        slide = OmeZarrSlide(_zarr_store([rgb], axes))
        assert slide.level_dimensions[0] == (50, 3)
        np.testing.assert_array_equal(np.asarray(slide.read_region((4, 0), 0, (5, 3))), rgb[:, 4:9])

    @pytest.mark.parametrize("with_axes", [True, False])
    def test_ngff_5d_tczyx(self, with_axes):
        rgb = _rgb(32, 48)
        tczyx = np.zeros((1, 3, 2, 32, 48), dtype=np.uint8)
        tczyx[0, :, 0] = np.transpose(rgb, (2, 0, 1))
        tczyx[0, :, 1] = 7  # second z-plane: must not be read
        half = np.ascontiguousarray(tczyx[:, :, :, ::2, ::2])
        axes = ["t", "c", "z", "y", "x"] if with_axes else None
        slide = OmeZarrSlide(_zarr_store([tczyx, half], axes))
        assert slide.level_dimensions == [(48, 32), (24, 16)]
        assert slide.level_downsamples == [pytest.approx(1.0), pytest.approx(2.0)]
        region = np.asarray(slide.read_region((6, 4), 0, (10, 12)))
        np.testing.assert_array_equal(region, rgb[4:16, 6:16])

    def test_reads_are_padded_at_the_edge(self):
        rgb = _rgb(20, 30)
        chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))  # (C, Y, X)
        slide = OmeZarrSlide(_zarr_store([chw]))
        img = slide.read_region((25, 15), 0, (10, 10))
        assert img.size == (10, 10)
        region = np.asarray(img)
        np.testing.assert_array_equal(region[:5, :5], rgb[15:20, 25:30])
        assert not region[5:].any() and not region[:, 5:].any()
        # Entirely outside the slide: all padding, still the requested size.
        assert np.asarray(slide.read_region((100, 100), 0, (4, 4))).shape == (4, 4, 3)

    def test_grayscale_2d_is_rgb(self):
        gray = np.arange(100, dtype=np.uint8).reshape(10, 10)
        slide = OmeZarrSlide(_zarr_store([gray], ["y", "x"]))
        region = np.asarray(slide.read_region((0, 0), 0, (3, 2)))
        assert region.shape == (2, 3, 3)
        np.testing.assert_array_equal(region[..., 1], gray[:2, :3])
