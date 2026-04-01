"""Tests for the mussel convert CLI (mussel/cli/convert.py)."""

import os
import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf
from PIL import Image
import numpy as np

import mussel.cli.convert as convert_module
from mussel.cli.convert import ConvertConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png(tmp_path: Path, name: str = "test.png", size=(32, 32)) -> str:
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    path = str(tmp_path / name)
    Image.fromarray(arr).save(path)
    return path


def _make_mpp_csv(tmp_path: Path, rows) -> str:
    csv_path = str(tmp_path / "mpp.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi", "mpp"])
        writer.writerows(rows)
    return csv_path


def _run_main(cfg_dict: dict):
    """Run convert main() with an OmegaConf-created config."""
    cfg = OmegaConf.structured(ConvertConfig(**cfg_dict))
    convert_module.main(cfg)


# ---------------------------------------------------------------------------
# Error-path tests (no I/O needed)
# ---------------------------------------------------------------------------


class TestConvertCLIErrors:
    def test_single_file_no_mpp_raises(self, tmp_path):
        """Single-file mode without mpp= should raise ValueError."""
        png = _make_png(tmp_path)
        cfg = ConvertConfig(
            input_path=png,
            output_dir=str(tmp_path / "out"),
            mpp=None,
        )
        with pytest.raises(ValueError, match="mpp is required"):
            convert_module.main(cfg)

    def test_directory_no_mpp_csv_raises(self, tmp_path):
        """Directory mode without mpp_csv= should raise ValueError."""
        (tmp_path / "slides").mkdir()
        cfg = ConvertConfig(
            input_path=str(tmp_path / "slides"),
            output_dir=str(tmp_path / "out"),
            mpp_csv=None,
        )
        with pytest.raises(ValueError, match="mpp_csv is required"):
            convert_module.main(cfg)

    def test_nonexistent_path_raises(self, tmp_path):
        """A path that is neither a file nor a directory should raise ValueError."""
        cfg = ConvertConfig(
            input_path=str(tmp_path / "does_not_exist.svs"),
            output_dir=str(tmp_path / "out"),
            mpp=0.5,
        )
        with pytest.raises(ValueError, match="does not exist"):
            convert_module.main(cfg)


# ---------------------------------------------------------------------------
# Single-file mode with mocked converter
# ---------------------------------------------------------------------------


class TestConvertCLISingleFile:
    def test_single_file_calls_process_file(self, tmp_path):
        """In single-file mode, AnyToTiffConverter.process_file is called once."""
        png = _make_png(tmp_path)
        out_dir = tmp_path / "out"

        cfg = ConvertConfig(
            input_path=png,
            output_dir=str(out_dir),
            mpp=0.25,
        )

        with patch("mussel.utils.converter.AnyToTiffConverter") as MockConverter:
            mock_instance = MagicMock()
            MockConverter.return_value = mock_instance
            convert_module.main(cfg)

        MockConverter.assert_called_once_with(job_dir=str(out_dir), bigtiff=False)
        mock_instance.process_file.assert_called_once_with(png, mpp=0.25)
        mock_instance.process_all.assert_not_called()

    def test_single_file_bigtiff_forwarded(self, tmp_path):
        """bigtiff=True should be forwarded to the converter constructor."""
        png = _make_png(tmp_path)
        cfg = ConvertConfig(
            input_path=png,
            output_dir=str(tmp_path / "out"),
            mpp=0.5,
            bigtiff=True,
        )

        with patch("mussel.utils.converter.AnyToTiffConverter") as MockConverter:
            MockConverter.return_value = MagicMock()
            convert_module.main(cfg)

        MockConverter.assert_called_once_with(job_dir=cfg.output_dir, bigtiff=True)


# ---------------------------------------------------------------------------
# Batch mode with mocked converter
# ---------------------------------------------------------------------------


class TestConvertCLIBatchMode:
    def test_batch_mode_calls_process_all(self, tmp_path):
        """In batch mode, AnyToTiffConverter.process_all is called once."""
        slides_dir = tmp_path / "slides"
        slides_dir.mkdir()
        csv_path = _make_mpp_csv(tmp_path, [("slide1.svs", 0.5)])
        out_dir = tmp_path / "out"

        cfg = ConvertConfig(
            input_path=str(slides_dir),
            output_dir=str(out_dir),
            mpp_csv=csv_path,
            num_workers=2,
            downscale_by=1,
        )

        with patch("mussel.utils.converter.AnyToTiffConverter") as MockConverter:
            mock_instance = MagicMock()
            MockConverter.return_value = mock_instance
            convert_module.main(cfg)

        mock_instance.process_all.assert_called_once_with(
            input_dir=str(slides_dir),
            mpp_csv=csv_path,
            downscale_by=1,
            num_workers=2,
        )
        mock_instance.process_file.assert_not_called()

    def test_batch_mode_downscale_by_forwarded(self, tmp_path):
        """downscale_by parameter should be forwarded to process_all."""
        slides_dir = tmp_path / "slides"
        slides_dir.mkdir()
        csv_path = _make_mpp_csv(tmp_path, [("slide.svs", 0.5)])

        cfg = ConvertConfig(
            input_path=str(slides_dir),
            output_dir=str(tmp_path / "out"),
            mpp_csv=csv_path,
            downscale_by=2,
        )

        with patch("mussel.utils.converter.AnyToTiffConverter") as MockConverter:
            mock_instance = MagicMock()
            MockConverter.return_value = mock_instance
            convert_module.main(cfg)

        call_kwargs = mock_instance.process_all.call_args.kwargs
        assert call_kwargs.get("downscale_by") == 2


# ---------------------------------------------------------------------------
# End-to-end: real PNG → TIFF (no exotic formats needed)
# ---------------------------------------------------------------------------


class TestConvertCLIEndToEnd:
    def test_single_png_converted(self, tmp_path):
        """Full pipeline: PNG → pyramidal TIFF via the convert CLI (mocked _save_tiff)."""
        png = _make_png(tmp_path, name="real.png", size=(64, 64))
        out_dir = tmp_path / "out"

        cfg = ConvertConfig(
            input_path=png,
            output_dir=str(out_dir),
            mpp=0.5,
        )

        from mussel.utils import converter as conv_mod
        with (
            patch.object(conv_mod.AnyToTiffConverter, "_try_pyvips_convert", return_value=False),
            patch.object(conv_mod.AnyToTiffConverter, "_save_tiff") as mock_save,
        ):
            convert_module.main(cfg)

        mock_save.assert_called_once()
        call_args = mock_save.call_args
        assert call_args.args[1] == "real"
        assert call_args.args[2] == pytest.approx(0.5)
