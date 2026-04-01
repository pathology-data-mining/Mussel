"""Tests for AnyToTiffConverter (mussel/utils/converter.py)."""

import csv
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from mussel.utils.converter import (
    BIOFORMAT_EXTENSIONS,
    CZI_EXTENSIONS,
    PIL_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    AnyToTiffConverter,
    _splitext,
)


# ---------------------------------------------------------------------------
# _splitext
# ---------------------------------------------------------------------------


class TestSplitExt:
    def test_simple_extension(self):
        stem, ext = _splitext("slide.svs")
        assert stem == "slide"
        assert ext == ".svs"

    def test_tiff_extension(self):
        stem, ext = _splitext("/path/to/file.tiff")
        assert stem == "file"
        assert ext == ".tiff"

    def test_compound_ome_tiff(self):
        stem, ext = _splitext("slide.ome.tiff")
        assert stem == "slide"
        assert ext == ".ome.tiff"

    def test_compound_ome_tif(self):
        stem, ext = _splitext("my_slide.ome.tif")
        assert stem == "my_slide"
        assert ext == ".ome.tif"

    def test_compound_ome_btf(self):
        stem, ext = _splitext("/a/b/c.ome.btf")
        assert stem == "c"
        assert ext == ".ome.btf"

    def test_mixed_case_compound(self):
        """Compound extension detection is case-insensitive."""
        stem, ext = _splitext("Slide.OME.TIFF")
        assert stem == "Slide"
        assert ext == ".ome.tiff"

    def test_no_extension(self):
        stem, ext = _splitext("noext")
        assert stem == "noext"
        assert ext == ""

    def test_dot_in_stem(self):
        """A dot in the filename stem should not confuse the parser."""
        stem, ext = _splitext("slide.v2.svs")
        assert ext == ".svs"
        assert "slide" in stem


# ---------------------------------------------------------------------------
# Extension set constants
# ---------------------------------------------------------------------------


class TestExtensionSets:
    def test_png_in_pil(self):
        assert ".png" in PIL_EXTENSIONS

    def test_jpeg_in_pil(self):
        assert ".jpeg" in PIL_EXTENSIONS

    def test_czi_in_czi_set(self):
        assert ".czi" in CZI_EXTENSIONS

    def test_dicom_in_bioformats(self):
        assert ".dcm" in BIOFORMAT_EXTENSIONS or ".dicom" in BIOFORMAT_EXTENSIONS

    def test_supported_is_union(self):
        assert PIL_EXTENSIONS.issubset(SUPPORTED_EXTENSIONS)
        assert CZI_EXTENSIONS.issubset(SUPPORTED_EXTENSIONS)
        assert BIOFORMAT_EXTENSIONS.issubset(SUPPORTED_EXTENSIONS)


# ---------------------------------------------------------------------------
# AnyToTiffConverter initialisation
# ---------------------------------------------------------------------------


class TestAnyToTiffConverterInit:
    def test_creates_output_dir(self, tmp_path):
        """__init__ should create the job_dir if it does not exist."""
        new_dir = tmp_path / "output" / "nested"
        assert not new_dir.exists()
        AnyToTiffConverter(job_dir=str(new_dir))
        assert new_dir.exists()

    def test_bigtiff_flag_stored(self, tmp_path):
        c = AnyToTiffConverter(job_dir=str(tmp_path), bigtiff=True)
        assert c.bigtiff is True


# ---------------------------------------------------------------------------
# process_file — PNG → TIFF via _save_tiff (numpy path)
# ---------------------------------------------------------------------------


def _make_png(tmp_path: Path, name: str = "test.png", size=(64, 64)) -> str:
    """Create a small synthetic PNG image and return its path."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    path = str(tmp_path / name)
    Image.fromarray(arr).save(path)
    return path


class TestProcessFile:
    def test_png_converted_via_numpy_path(self, tmp_path):
        """process_file converts a PNG to TIFF using the numpy fallback."""
        png_path = _make_png(tmp_path)
        converter = AnyToTiffConverter(job_dir=str(tmp_path / "out"))

        # Patch pyvips so we always hit the numpy read path, then mock _save_tiff
        with (
            patch.object(converter, "_try_pyvips_convert", return_value=False),
            patch.object(converter, "_save_tiff") as mock_save,
        ):
            converter.process_file(png_path, mpp=0.5)

        mock_save.assert_called_once()
        call_args = mock_save.call_args
        # Second arg is the stem (without extension), third is the mpp
        assert call_args.args[1] == "test"
        assert call_args.args[2] == pytest.approx(0.5)

    def test_output_tiff_is_readable(self, tmp_path):
        """The output TIFF should be a valid image readable by PIL (requires pyvips)."""
        pytest.importorskip("pyvips", reason="pyvips not installed")
        png_path = _make_png(tmp_path, size=(32, 32))
        out_dir = tmp_path / "out"
        converter = AnyToTiffConverter(job_dir=str(out_dir))

        with patch.object(converter, "_try_pyvips_convert", return_value=False):
            converter.process_file(png_path, mpp=0.25)

        tiff_path = out_dir / "test.tiff"
        img = Image.open(str(tiff_path))
        assert img.size == (32, 32)

    def test_exception_in_process_file_is_caught(self, tmp_path):
        """process_file should not propagate exceptions (logs instead)."""
        converter = AnyToTiffConverter(job_dir=str(tmp_path))
        # Pass a non-existent file — will raise internally but should be caught
        converter.process_file("/does/not/exist.png", mpp=0.5)  # should not raise

    def test_pyvips_fast_path_used_when_available(self, tmp_path):
        """If _try_pyvips_convert returns True, _read_image is not called."""
        png_path = _make_png(tmp_path)
        converter = AnyToTiffConverter(job_dir=str(tmp_path / "out"))

        with (
            patch.object(converter, "_try_pyvips_convert", return_value=True) as mock_pyvips,
            patch.object(converter, "_read_image") as mock_read,
        ):
            converter.process_file(png_path, mpp=0.5)

        mock_pyvips.assert_called_once()
        mock_read.assert_not_called()


# ---------------------------------------------------------------------------
# process_all — batch mode
# ---------------------------------------------------------------------------


def _make_mpp_csv(tmp_path: Path, rows) -> str:
    csv_path = str(tmp_path / "mpp.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi", "mpp"])
        writer.writerows(rows)
    return csv_path


class TestProcessAll:
    def test_raises_if_csv_missing(self, tmp_path):
        converter = AnyToTiffConverter(job_dir=str(tmp_path))
        with pytest.raises(ValueError, match="MPP CSV not found"):
            converter.process_all(
                input_dir=str(tmp_path),
                mpp_csv=str(tmp_path / "nonexistent.csv"),
            )

    def test_raises_if_csv_missing_columns(self, tmp_path):
        bad_csv = str(tmp_path / "bad.csv")
        with open(bad_csv, "w") as f:
            f.write("filename,resolution\nslide.svs,0.5\n")
        converter = AnyToTiffConverter(job_dir=str(tmp_path))
        with pytest.raises(ValueError, match="missing columns"):
            converter.process_all(input_dir=str(tmp_path), mpp_csv=bad_csv)

    def test_raises_if_csv_empty(self, tmp_path):
        csv_path = _make_mpp_csv(tmp_path, [])
        converter = AnyToTiffConverter(job_dir=str(tmp_path))
        with pytest.raises(ValueError, match="empty"):
            converter.process_all(input_dir=str(tmp_path), mpp_csv=csv_path)

    def test_raises_if_no_valid_tasks(self, tmp_path):
        """All files in CSV are absent → should raise ValueError."""
        csv_path = _make_mpp_csv(tmp_path, [("missing.svs", 0.5)])
        converter = AnyToTiffConverter(job_dir=str(tmp_path))
        with pytest.raises(ValueError, match="No valid conversion tasks"):
            converter.process_all(input_dir=str(tmp_path), mpp_csv=csv_path)

    def test_raises_if_downscale_by_less_than_1(self, tmp_path):
        csv_path = _make_mpp_csv(tmp_path, [("slide.png", 0.5)])
        converter = AnyToTiffConverter(job_dir=str(tmp_path))
        with pytest.raises(ValueError, match="downscale_by"):
            converter.process_all(
                input_dir=str(tmp_path), mpp_csv=csv_path, downscale_by=0
            )

    def test_converts_png_via_csv(self, tmp_path):
        """process_all converts a PNG listed in the CSV."""
        png_path = _make_png(tmp_path, name="slide.png")
        csv_path = _make_mpp_csv(tmp_path, [("slide.png", 0.5)])
        out_dir = tmp_path / "out"
        converter = AnyToTiffConverter(job_dir=str(out_dir))

        # Force numpy read path + mock save to avoid pyvips dependency in CI
        with (
            patch.object(converter, "_try_pyvips_convert", return_value=False),
            patch.object(converter, "_save_tiff"),
        ):
            converter.process_all(
                input_dir=str(tmp_path),
                mpp_csv=csv_path,
                num_workers=1,
            )

    def test_unsupported_extension_is_skipped(self, tmp_path):
        """Files with unsupported extensions are skipped, not raised."""
        weird_path = tmp_path / "slide.xyz123"
        weird_path.write_text("dummy")
        csv_path = _make_mpp_csv(tmp_path, [("slide.xyz123", 0.5)])
        converter = AnyToTiffConverter(job_dir=str(tmp_path / "out"))
        with pytest.raises(ValueError, match="No valid conversion tasks"):
            converter.process_all(input_dir=str(tmp_path), mpp_csv=csv_path)
