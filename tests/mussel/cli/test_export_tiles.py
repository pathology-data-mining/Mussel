"""Tests for the mussel export_tiles CLI and tile_export utility module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
from PIL import Image

import mussel.cli.export_tiles as export_tiles_module
from mussel.cli.export_tiles import ExportTilesConfig
from mussel.utils.tile_export import export_tile, export_tiles

# ---------------------------------------------------------------------------
# Patch arrays for is_white_patch / is_black_patch filter logic:
#   _TISSUE: saturation≈128 (>>5) and mean_rgb≈150 (>>40)  → passes both filters
#   _WHITE:  all-255 → HSV saturation=0 (<5)                → is_white=True, skipped
#   _BLACK:  all-10  → mean_rgb=10 (<40)                    → is_black=True, skipped
# ---------------------------------------------------------------------------
_TISSUE = np.full((32, 32, 3), [200, 100, 150], dtype=np.uint8)
_WHITE = np.full((32, 32, 3), 255, dtype=np.uint8)
_BLACK = np.full((32, 32, 3), 10, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_patch_h5(path: str, coords) -> str:
    """Create a minimal HDF5 with a 'coords' dataset."""
    if len(coords) == 0:
        data = np.empty((0, 2), dtype=np.int64)
    else:
        data = np.array(coords, dtype=np.int64)
    with h5py.File(path, "w") as f:
        f.create_dataset("coords", data=data)
    return path


def _mock_wsi(patch_array: np.ndarray) -> MagicMock:
    """Return a mock TiffSlide whose read_region always yields patch_array."""
    pil_img = Image.fromarray(patch_array)
    mock_wsi = MagicMock()
    mock_wsi.read_region.return_value.convert.return_value = pil_img
    return mock_wsi


# ---------------------------------------------------------------------------
# TestExportTile — unit tests for the per-tile function
# ---------------------------------------------------------------------------


class TestExportTile:
    def test_tissue_patch_saved(self, tmp_path):
        """A tissue-coloured patch should be written to disk."""
        out = tmp_path / "tiles"
        out.mkdir()
        export_tile(np.array([128, 256]), _mock_wsi(_TISSUE), patch_size=32, output_path=str(out))
        assert (out / "128_256.png").exists()

    def test_white_patch_not_saved(self, tmp_path):
        """An all-white patch should be silently discarded."""
        out = tmp_path / "tiles"
        out.mkdir()
        export_tile(np.array([0, 0]), _mock_wsi(_WHITE), patch_size=32, output_path=str(out))
        assert list(out.iterdir()) == []

    def test_black_patch_not_saved(self, tmp_path):
        """An all-black patch should be silently discarded."""
        out = tmp_path / "tiles"
        out.mkdir()
        export_tile(np.array([0, 0]), _mock_wsi(_BLACK), patch_size=32, output_path=str(out))
        assert list(out.iterdir()) == []

    def test_filename_uses_xy_coords(self, tmp_path):
        """Output filename must be '{x}_{y}.png' where x/y come from tile_coords."""
        out = tmp_path / "tiles"
        out.mkdir()
        export_tile(np.array([1234, 5678]), _mock_wsi(_TISSUE), patch_size=32, output_path=str(out))
        assert (out / "1234_5678.png").exists()
        assert not (out / "5678_1234.png").exists()

    def test_saved_image_is_valid_png(self, tmp_path):
        """The written file should be a readable image with the requested dimensions."""
        out = tmp_path / "tiles"
        out.mkdir()
        export_tile(np.array([10, 20]), _mock_wsi(_TISSUE), patch_size=32, output_path=str(out))
        img = Image.open(str(out / "10_20.png"))
        assert img.size == (32, 32)


# ---------------------------------------------------------------------------
# TestExportTiles — mocked TiffSlide + get_slide_mpp
# ---------------------------------------------------------------------------

_SMALL_COORDS = [[10, 20], [30, 40], [50, 60]]


class TestExportTiles:
    def _setup(self, tmp_path, coords=None):
        """Return (h5_path, slide_path, out_dir_str) ready for export_tiles."""
        if coords is None:
            coords = _SMALL_COORDS
        h5_path = str(tmp_path / "tiles.patch.h5")
        _make_patch_h5(h5_path, coords)
        out_dir = tmp_path / "pngs"
        out_dir.mkdir()
        return h5_path, str(tmp_path / "fake.svs"), str(out_dir)

    def test_processes_all_tissue_coords(self, tmp_path):
        """All coords with tissue patches should produce one PNG each."""
        h5, slide, out = self._setup(tmp_path)
        with (
            patch("mussel.utils.tile_export.tiffslide.TiffSlide", return_value=_mock_wsi(_TISSUE)),
            patch("mussel.utils.tile_export.get_slide_mpp", return_value=0.5),
        ):
            export_tiles(h5, slide, out, mpp=0.5, num_workers=1)
        assert len(list(Path(out).glob("*.png"))) == 3

    def test_white_tiles_filtered(self, tmp_path):
        """Tiles whose read_region returns an all-white image should be skipped."""
        h5, slide, out = self._setup(tmp_path)
        # First coord → white, remaining two → tissue
        side_effects = []
        for arr in [_WHITE, _TISSUE, _TISSUE]:
            m = MagicMock()
            m.convert.return_value = Image.fromarray(arr)
            side_effects.append(m)
        mock_wsi = MagicMock()
        mock_wsi.read_region.side_effect = side_effects
        with (
            patch("mussel.utils.tile_export.tiffslide.TiffSlide", return_value=mock_wsi),
            patch("mussel.utils.tile_export.get_slide_mpp", return_value=0.5),
        ):
            export_tiles(h5, slide, out, mpp=0.5, num_workers=1)
        assert len(list(Path(out).glob("*.png"))) == 2

    def test_empty_coords_no_error_no_pngs(self, tmp_path):
        """An empty H5 manifest should complete without error and produce no PNGs."""
        h5, slide, out = self._setup(tmp_path, coords=[])
        with (
            patch("mussel.utils.tile_export.tiffslide.TiffSlide", return_value=_mock_wsi(_TISSUE)),
            patch("mussel.utils.tile_export.get_slide_mpp", return_value=0.5),
            patch("mussel.utils.tile_export.export_tile") as mock_et,
        ):
            export_tiles(h5, slide, out, mpp=0.5, num_workers=1)
        mock_et.assert_not_called()
        assert list(Path(out).glob("*.png")) == []

    def test_slide_mpp_override_forwarded(self, tmp_path):
        """slide_mpp_override should be passed through to get_slide_mpp."""
        h5, slide, out = self._setup(tmp_path)
        mock_wsi = _mock_wsi(_TISSUE)
        with (
            patch("mussel.utils.tile_export.tiffslide.TiffSlide", return_value=mock_wsi),
            patch("mussel.utils.tile_export.get_slide_mpp", return_value=0.5) as mock_mpp,
        ):
            export_tiles(h5, slide, out, mpp=0.5, num_workers=1, slide_mpp_override=0.25)
        mock_mpp.assert_called_once_with(mock_wsi, slide, slide_mpp_override=0.25)

    def test_native_patch_size_computed_from_mpp_ratio(self, tmp_path):
        """mpp=0.5 with slide_mpp=0.25 → native_patch_size = 256 * (0.5/0.25) = 512."""
        h5, slide, out = self._setup(tmp_path)
        with (
            patch("mussel.utils.tile_export.tiffslide.TiffSlide", return_value=_mock_wsi(_TISSUE)),
            patch("mussel.utils.tile_export.get_slide_mpp", return_value=0.25),
            patch("mussel.utils.tile_export.export_tile") as mock_et,
        ):
            export_tiles(h5, slide, out, patch_size=256, mpp=0.5, num_workers=1)
        for call in mock_et.call_args_list:
            assert call.kwargs["patch_size"] == 512

    def test_mpp_below_slide_mpp_raises(self, tmp_path):
        """Requesting mpp finer than the slide's native resolution raises AssertionError."""
        h5, slide, out = self._setup(tmp_path)
        with (
            patch("mussel.utils.tile_export.tiffslide.TiffSlide", return_value=_mock_wsi(_TISSUE)),
            patch("mussel.utils.tile_export.get_slide_mpp", return_value=0.5),
            pytest.raises(AssertionError),
        ):
            export_tiles(h5, slide, out, patch_size=256, mpp=0.25, num_workers=1)


# ---------------------------------------------------------------------------
# TestExportTilesRealSlide — end-to-end with real testdata
# ---------------------------------------------------------------------------

# Pre-verified: all three coordinates contain tissue (not white/black-filtered).
_REAL_TISSUE_COORDS = [[74708, 18123], [42956, 16939], [44460, 16875]]


class TestExportTilesRealSlide:
    def test_produces_pngs_for_tissue_coords(self, tmp_path):
        """Export from the real 948176.svs with 3 known-tissue coords → 3 PNGs."""
        h5_path = str(tmp_path / "real.patch.h5")
        _make_patch_h5(h5_path, _REAL_TISSUE_COORDS)
        out = tmp_path / "out"
        out.mkdir()
        export_tiles(
            patch_h5_path=h5_path,
            slide_path="tests/testdata/948176.svs",
            output_png_path=str(out),
            patch_size=256,
            mpp=0.5,
            num_workers=1,
        )
        pngs = list(out.glob("*.png"))
        assert len(pngs) == 3

    def test_slide_mpp_override_skips_metadata(self, tmp_path):
        """slide_mpp_override bypasses metadata and drives native_patch_size correctly."""
        h5_path = str(tmp_path / "real.patch.h5")
        _make_patch_h5(h5_path, _REAL_TISSUE_COORDS[:1])
        out = tmp_path / "out"
        out.mkdir()
        # slide actual mpp≈0.5026; override with exact 0.5 → native_patch_size=256
        export_tiles(
            patch_h5_path=h5_path,
            slide_path="tests/testdata/948176.svs",
            output_png_path=str(out),
            patch_size=256,
            mpp=0.5,
            num_workers=1,
            slide_mpp_override=0.5,
        )
        assert len(list(out.glob("*.png"))) == 1


# ---------------------------------------------------------------------------
# TestExportTilesCLI — CLI routing
# ---------------------------------------------------------------------------


class TestExportTilesCLI:
    def test_main_routes_to_export_tiles(self, tmp_path):
        """main() should delegate to export_tiles with all config fields forwarded."""
        cfg = ExportTilesConfig(
            slide_path="test.svs",
            patch_h5_path="test.patch.h5",
            output_png_path=str(tmp_path / "out"),
            patch_size=256,
            mpp=0.5,
            num_workers=2,
            slide_mpp_override=None,
        )
        with patch("mussel.cli.export_tiles.export_tiles") as mock_et:
            export_tiles_module.main(cfg)
        mock_et.assert_called_once_with(
            patch_h5_path="test.patch.h5",
            slide_path="test.svs",
            output_png_path=cfg.output_png_path,
            patch_size=256,
            mpp=0.5,
            num_workers=2,
            slide_mpp_override=None,
        )

    def test_main_slide_mpp_override_forwarded(self, tmp_path):
        """slide_mpp_override=0.25 from config should reach export_tiles."""
        cfg = ExportTilesConfig(
            slide_path="test.svs",
            patch_h5_path="test.patch.h5",
            output_png_path=str(tmp_path / "out"),
            slide_mpp_override=0.25,
        )
        with patch("mussel.cli.export_tiles.export_tiles") as mock_et:
            export_tiles_module.main(cfg)
        assert mock_et.call_args.kwargs["slide_mpp_override"] == pytest.approx(0.25)

    def test_main_default_num_workers(self, tmp_path):
        """Default num_workers=16 should be forwarded unchanged."""
        cfg = ExportTilesConfig(
            slide_path="s.svs",
            patch_h5_path="p.h5",
            output_png_path=str(tmp_path / "out"),
        )
        with patch("mussel.cli.export_tiles.export_tiles") as mock_et:
            export_tiles_module.main(cfg)
        assert mock_et.call_args.kwargs["num_workers"] == 16
