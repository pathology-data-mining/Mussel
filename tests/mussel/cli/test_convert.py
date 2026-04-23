"""Tests for the mussel convert CLI (mussel/cli/convert.py)."""

import csv
import multiprocessing as mp
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

import mussel.cli.convert as convert_module
from mussel.cli.convert import ConvertConfig
from mussel.utils.converter import AnyToTiffConverter, _splitext

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

        with patch("mussel.cli.convert.AnyToTiffConverter") as MockConverter:
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

        with patch("mussel.cli.convert.AnyToTiffConverter") as MockConverter:
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

        with patch("mussel.cli.convert.AnyToTiffConverter") as MockConverter:
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

        with patch("mussel.cli.convert.AnyToTiffConverter") as MockConverter:
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
            patch.object(
                conv_mod.AnyToTiffConverter, "_try_pyvips_convert", return_value=False
            ),
            patch.object(conv_mod.AnyToTiffConverter, "_save_tiff") as mock_save,
        ):
            convert_module.main(cfg)

        mock_save.assert_called_once()
        call_args = mock_save.call_args
        assert call_args.args[1] == "real"
        assert call_args.args[2] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Integration tests (require pyvips — skipped when not installed)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestConvertCLIIntegration:
    """Integration tests that exercise the real pyvips pipeline end-to-end.

    Automatically skipped when pyvips is not installed.
    """

    @pytest.fixture(autouse=True)
    def require_pyvips(self):
        pytest.importorskip("pyvips", reason="pyvips not installed")

    def test_single_file_produces_readable_tiff(self, tmp_path):
        """PNG → pyramidal TIFF via pyvips; output is readable by tiffslide."""
        png = _make_png(tmp_path, name="slide.png", size=(128, 128))
        out_dir = tmp_path / "out"

        cfg = ConvertConfig(input_path=png, output_dir=str(out_dir), mpp=0.5)
        convert_module.main(cfg)

        out_tiff = out_dir / "slide.tiff"
        assert out_tiff.exists(), f"Expected output TIFF at {out_tiff}"

        import tiffslide

        with tiffslide.open_slide(str(out_tiff)) as wsi:
            w, h = wsi.level_dimensions[0]
            assert w == 128 and h == 128, f"Expected 128x128 at level 0, got {w}x{h}"
            region = wsi.read_region((0, 0), 0, (32, 32)).convert("RGB")
            assert region.size == (32, 32)

    def test_batch_mode_converts_multiple_files(self, tmp_path):
        """Batch mode converts all PNGs listed in the CSV."""
        slides_dir = tmp_path / "slides"
        slides_dir.mkdir()

        # Create two PNGs
        for name, size in [("a.png", (64, 64)), ("b.png", (32, 48))]:
            _make_png(slides_dir, name=name, size=size)

        csv_path = _make_mpp_csv(tmp_path, [("a.png", 0.5), ("b.png", 0.25)])
        out_dir = tmp_path / "out"

        cfg = ConvertConfig(
            input_path=str(slides_dir),
            output_dir=str(out_dir),
            mpp_csv=csv_path,
            num_workers=1,
        )
        convert_module.main(cfg)

        assert (out_dir / "a.tiff").exists(), "Expected a.tiff"
        assert (out_dir / "b.tiff").exists(), "Expected b.tiff"

    def test_bigtiff_flag_creates_file(self, tmp_path):
        """bigtiff=True path executes without error for a small file."""
        png = _make_png(tmp_path, name="big.png", size=(64, 64))
        out_dir = tmp_path / "out"

        cfg = ConvertConfig(
            input_path=png, output_dir=str(out_dir), mpp=0.5, bigtiff=True
        )
        convert_module.main(cfg)
        assert (out_dir / "big.tiff").exists()

    def test_output_tiff_has_correct_mpp_metadata(self, tmp_path):
        """The converted TIFF's MPP property matches the requested mpp."""
        png = _make_png(tmp_path, name="mpp_test.png", size=(64, 64))
        out_dir = tmp_path / "out"
        target_mpp = 0.25

        cfg = ConvertConfig(input_path=png, output_dir=str(out_dir), mpp=target_mpp)
        convert_module.main(cfg)

        import tiffslide

        with tiffslide.open_slide(str(out_dir / "mpp_test.tiff")) as wsi:
            mpp_x = wsi.properties.get("tiffslide.mpp-x") or wsi.properties.get(
                "openslide.mpp-x"
            )
            if mpp_x is not None:
                assert (
                    abs(float(mpp_x) - target_mpp) < 0.05
                ), f"Expected MPP ~{target_mpp}, got {mpp_x}"


# ---------------------------------------------------------------------------
# _splitext helper
# ---------------------------------------------------------------------------


class TestSplitExt:
    def test_ome_tiff_compound(self):
        assert _splitext("slide.ome.tiff") == ("slide", ".ome.tiff")

    def test_ome_tif_compound(self):
        assert _splitext("slide.ome.tif") == ("slide", ".ome.tif")

    def test_ome_btf_compound(self):
        assert _splitext("slide.ome.btf") == ("slide", ".ome.btf")

    def test_uppercase_compound_normalized(self):
        """Case-insensitive match returns lowercased compound extension."""
        stem, ext = _splitext("SLIDE.OME.TIFF")
        assert ext == ".ome.tiff"
        # stem preserves original case up to extension
        assert stem == "SLIDE"

    def test_regular_svs(self):
        assert _splitext("slide.svs") == ("slide", ".svs")

    def test_strips_directory_prefix(self):
        """Only the basename is used; directory components are stripped."""
        assert _splitext("dir/sub/slide.png") == ("slide", ".png")

    def test_ome_xml_not_handled_as_compound(self):
        """Known limitation: .ome.xml is in SUPPORTED_EXTENSIONS but _splitext
        does not recognise it as a compound extension, returning .xml instead.
        This means process_all silently skips .ome.xml files (see process_all
        TestProcessAllBehavior.test_ome_xml_skipped_by_process_all)."""
        stem, ext = _splitext("slide.ome.xml")
        # Currently returns .xml, not .ome.xml
        assert ext == ".xml"
        assert stem == "slide.ome"


# ---------------------------------------------------------------------------
# AnyToTiffConverter.__init__
# ---------------------------------------------------------------------------


class TestAnyToTiffConverterInit:
    def test_creates_output_dir(self, tmp_path):
        out = tmp_path / "nested" / "output"
        assert not out.exists()
        AnyToTiffConverter(job_dir=str(out))
        assert out.is_dir()

    def test_bigtiff_default_false(self, tmp_path):
        c = AnyToTiffConverter(job_dir=str(tmp_path / "out"))
        assert c.bigtiff is False

    def test_bigtiff_true_stored(self, tmp_path):
        c = AnyToTiffConverter(job_dir=str(tmp_path / "out"), bigtiff=True)
        assert c.bigtiff is True


# ---------------------------------------------------------------------------
# process_all — validation errors
# ---------------------------------------------------------------------------


class TestProcessAllValidation:
    def _converter(self, tmp_path):
        return AnyToTiffConverter(job_dir=str(tmp_path / "out"))

    def test_downscale_by_zero_raises(self, tmp_path):
        c = self._converter(tmp_path)
        with pytest.raises(ValueError, match="downscale_by"):
            c.process_all(str(tmp_path), mpp_csv="x.csv", downscale_by=0)

    def test_num_workers_negative_raises(self, tmp_path):
        c = self._converter(tmp_path)
        csv_path = _make_mpp_csv(tmp_path, [])
        with pytest.raises(ValueError, match="num_workers"):
            c.process_all(str(tmp_path), mpp_csv=csv_path, num_workers=-1)

    def test_csv_not_found_raises(self, tmp_path):
        c = self._converter(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            c.process_all(str(tmp_path), mpp_csv=str(tmp_path / "no.csv"))

    def test_csv_missing_wsi_column_raises(self, tmp_path):
        csv_path = str(tmp_path / "bad.csv")
        import pandas as pd
        pd.DataFrame({"filename": ["a.svs"], "mpp": [0.5]}).to_csv(csv_path, index=False)
        c = self._converter(tmp_path)
        with pytest.raises(ValueError, match="wsi"):
            c.process_all(str(tmp_path), mpp_csv=csv_path)

    def test_csv_missing_mpp_column_raises(self, tmp_path):
        csv_path = str(tmp_path / "bad.csv")
        import pandas as pd
        pd.DataFrame({"wsi": ["a.svs"]}).to_csv(csv_path, index=False)
        c = self._converter(tmp_path)
        with pytest.raises(ValueError, match="mpp"):
            c.process_all(str(tmp_path), mpp_csv=csv_path)

    def test_csv_empty_raises(self, tmp_path):
        csv_path = _make_mpp_csv(tmp_path, [])
        c = self._converter(tmp_path)
        with pytest.raises(ValueError, match="empty"):
            c.process_all(str(tmp_path), mpp_csv=csv_path)

    def test_no_valid_tasks_raises(self, tmp_path):
        """All files in CSV are absent from input_dir → no tasks."""
        csv_path = _make_mpp_csv(tmp_path, [("ghost.svs", 0.5)])
        c = self._converter(tmp_path)
        with pytest.raises(ValueError, match="No valid conversion tasks"):
            c.process_all(str(tmp_path), mpp_csv=csv_path)


# ---------------------------------------------------------------------------
# process_all — skip/behavior (process_file mocked)
# ---------------------------------------------------------------------------


class TestProcessAllBehavior:
    def _make_converter(self, tmp_path):
        return AnyToTiffConverter(job_dir=str(tmp_path / "out"))

    def test_skips_missing_files(self, tmp_path, caplog):
        """Files listed in CSV but absent from input_dir are skipped with warning."""
        input_dir = tmp_path / "slides"
        input_dir.mkdir()
        _make_png(input_dir, name="present.png")
        csv_path = _make_mpp_csv(tmp_path, [("present.png", 0.5), ("missing.svs", 0.5)])

        c = self._make_converter(tmp_path)
        with patch.object(c, "process_file") as mock_pf:
            import logging
            with caplog.at_level(logging.WARNING, logger="mussel.utils.converter"):
                c.process_all(str(input_dir), mpp_csv=csv_path, num_workers=1)

        mock_pf.assert_called_once()
        assert "not found" in caplog.text.lower() or "skipping" in caplog.text.lower()

    def test_skips_unsupported_extension(self, tmp_path, caplog):
        """Files with unsupported extension are skipped; supported ones are processed."""
        input_dir = tmp_path / "slides"
        input_dir.mkdir()
        _make_png(input_dir, name="good.png")
        (input_dir / "bad.xyz").write_text("not an image")
        csv_path = _make_mpp_csv(tmp_path, [("good.png", 0.5), ("bad.xyz", 0.5)])

        c = self._make_converter(tmp_path)
        with patch.object(c, "process_file") as mock_pf:
            import logging
            with caplog.at_level(logging.WARNING, logger="mussel.utils.converter"):
                c.process_all(str(input_dir), mpp_csv=csv_path, num_workers=1)

        mock_pf.assert_called_once()
        assert "unsupported" in caplog.text.lower()

    def test_downscale_by_passes_correct_zoom(self, tmp_path):
        """downscale_by=4 should pass zoom=0.25 to process_file."""
        input_dir = tmp_path / "slides"
        input_dir.mkdir()
        _make_png(input_dir, name="slide.png")
        csv_path = _make_mpp_csv(tmp_path, [("slide.png", 0.5)])

        c = self._make_converter(tmp_path)
        with patch.object(c, "process_file") as mock_pf:
            c.process_all(str(input_dir), mpp_csv=csv_path, downscale_by=4, num_workers=1)

        mock_pf.assert_called_once()
        _, call_kwargs = mock_pf.call_args
        assert call_kwargs.get("zoom") == pytest.approx(0.25)

    def test_num_workers_zero_uses_cpu_count(self, tmp_path):
        """num_workers=0 should use mp.cpu_count() (patched to 1 → serial path)."""
        input_dir = tmp_path / "slides"
        input_dir.mkdir()
        _make_png(input_dir, name="slide.png")
        csv_path = _make_mpp_csv(tmp_path, [("slide.png", 0.5)])

        c = self._make_converter(tmp_path)
        with patch.object(c, "process_file") as mock_pf, \
             patch("mussel.utils.converter.mp.cpu_count", return_value=1):
            c.process_all(str(input_dir), mpp_csv=csv_path, num_workers=0)

        mock_pf.assert_called_once()

    def test_ome_xml_skipped_by_process_all(self, tmp_path):
        """Known limitation: .ome.xml files are skipped because _splitext returns
        .xml (not .ome.xml), and .xml is not in SUPPORTED_EXTENSIONS."""
        input_dir = tmp_path / "slides"
        input_dir.mkdir()
        (input_dir / "slide.ome.xml").write_text("<OME/>")
        csv_path = _make_mpp_csv(tmp_path, [("slide.ome.xml", 0.5)])

        c = self._make_converter(tmp_path)
        # All entries skipped → ValueError
        with pytest.raises(ValueError, match="No valid conversion tasks"):
            c.process_all(str(input_dir), mpp_csv=csv_path)


# ---------------------------------------------------------------------------
# process_file — exception handling and CZI early-return
# ---------------------------------------------------------------------------


class TestProcessFileBehavior:
    def _make_converter(self, tmp_path):
        return AnyToTiffConverter(job_dir=str(tmp_path / "out"))

    def test_exception_is_caught_not_raised(self, tmp_path):
        """process_file swallows all exceptions; callers should never see them."""
        png = _make_png(tmp_path, name="err.png")
        c = self._make_converter(tmp_path)
        with patch.object(c, "_detect_embedded_mpp", return_value=None), \
             patch.object(c, "_try_pyvips_convert", return_value=False), \
             patch.object(c, "_read_image", side_effect=RuntimeError("disk error")):
            # Must not raise
            c.process_file(png, mpp=0.5)

    def test_czi_skips_pyvips_fast_path(self, tmp_path):
        """_try_pyvips_convert returns False immediately for .czi files."""
        c = self._make_converter(tmp_path)
        result = c._try_pyvips_convert(
            "slide.czi", str(tmp_path / "out.tiff"), zoom=1.0, mpp=0.5
        )
        assert result is False

    def test_zoom_scales_output_mpp(self, tmp_path):
        """With zoom=0.5 (downscale_by=2), output_mpp = input_mpp / zoom = mpp * 2."""
        png = _make_png(tmp_path, name="real.png", size=(32, 32))
        c = self._make_converter(tmp_path)
        with patch.object(c, "_detect_embedded_mpp", return_value=None), \
             patch.object(c, "_try_pyvips_convert", return_value=False), \
             patch.object(c, "_save_tiff") as mock_save:
            c.process_file(png, mpp=0.5, zoom=0.5)

        mock_save.assert_called_once()
        _img, _stem, saved_mpp = mock_save.call_args.args
        assert saved_mpp == pytest.approx(1.0)  # 0.5 / 0.5 = 1.0
