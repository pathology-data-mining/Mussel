"""TIFF resolution tags written by AnyToTiffConverter (pyvips is mocked)."""
from unittest.mock import MagicMock, patch

import pytest

from mussel.utils.converter import AnyToTiffConverter


@pytest.mark.parametrize("mpp", [0.25, 0.5, 1.0])
def test_tiffsave_writes_pixels_per_mm(tmp_path, mpp):
    """libvips' tiffsave xres/yres are pixels per millimetre, whatever resunit says."""
    converter = AnyToTiffConverter(job_dir=str(tmp_path))
    img = MagicMock()
    pyvips_mod = MagicMock()
    converter._save_pyvips_tiff(img, str(tmp_path / "x.tiff"), mpp, pyvips_mod)
    kwargs = img.tiffsave.call_args.kwargs
    assert kwargs["xres"] == pytest.approx(1000.0 / mpp)
    assert kwargs["yres"] == pytest.approx(1000.0 / mpp)
    assert kwargs["resunit"] is pyvips_mod.enums.ForeignTiffResunit.CM


@pytest.mark.parametrize("mpp", [0.25, 0.5])
def test_written_mpp_round_trips_through_the_reader(tmp_path, mpp):
    """What _save_pyvips_tiff writes, _detect_mpp_pyvips (libvips: xres in px/mm) reads back."""
    converter = AnyToTiffConverter(job_dir=str(tmp_path))
    img = MagicMock()
    converter._save_pyvips_tiff(img, str(tmp_path / "x.tiff"), mpp, MagicMock())
    written_xres = img.tiffsave.call_args.kwargs["xres"]

    reread = MagicMock()
    reread.get_typeof.return_value = 1
    reread.get.return_value = written_xres
    fake_pyvips = MagicMock()
    fake_pyvips.Image.new_from_file.return_value = reread
    with patch.dict("sys.modules", {"pyvips": fake_pyvips}):
        assert converter._detect_mpp_pyvips("x.tiff") == pytest.approx(mpp)
