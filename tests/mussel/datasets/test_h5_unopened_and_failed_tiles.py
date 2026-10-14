"""WholeSlideImageH5Dataset must not silently skip tiles of a slide it never opened,
and feature extraction must report tiles that failed to load."""
import logging
from pathlib import Path
from unittest.mock import patch

import h5py
import pytest
from torchvision import transforms

import mussel.datasets.h5 as h5mod
from mussel.datasets import WholeSlideImageH5Dataset
from mussel.models import ModelType
from mussel.utils.feature_extract import extract_patch_features_batch

_TESTDATA = Path(__file__).parent.parent.parent / "testdata"
_SLIDE = str(_TESTDATA / "948176.svs")
_H5 = str(_TESTDATA / "948176.patch.h5")
_PRE = transforms.Compose([transforms.Resize(16), transforms.ToTensor()])


def _small_h5(tmp_path, n=20):
    with h5py.File(_H5) as f:
        coords, attrs = f["coords"][:n], dict(f["coords"].attrs)
    p = tmp_path / "s.patch.h5"
    with h5py.File(p, "w") as f:
        ds = f.create_dataset("coords", data=coords)
        for k, v in attrs.items():
            ds.attrs[k] = v
    return str(p)


def test_reading_an_unopened_slide_raises(tmp_path):
    ds = WholeSlideImageH5Dataset(_small_h5(tmp_path), _SLIDE, preprocess=_PRE,
                                  use_imagenet_rgb_dist=False, init_wsi_in_worker=True)
    with pytest.raises(RuntimeError, match="not open"):
        ds[0]
    ds.worker_init()
    img, coord = ds[0]
    assert img is not None and img.shape[0] == 1


class _FlakySlide:
    """A slide whose every third tile read fails, like a corrupt JPEG tile."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def read_region(self, *args):
        self.calls += 1
        if self.calls % 3 == 0:
            raise ValueError("imagecodecs: corrupt tile")
        return self.inner.read_region(*args)


class _FakeModel:
    def get_preprocessing_fun(self):
        return _PRE

    def get_model_fun(self):
        return lambda x: x.mean((2, 3))


class _FakeFactory:
    def get_model(self, *args, **kwargs):
        return _FakeModel()


def test_failed_tiles_are_reported(tmp_path, caplog):
    h5 = _small_h5(tmp_path, n=20)
    out = str(tmp_path / "out.h5")
    real_open = h5mod.openslide.open_slide

    def open_flaky(path):
        return _FlakySlide(real_open(path))

    with patch("mussel.utils.feature_extract.get_model_factory", return_value=_FakeFactory()), \
            patch.object(h5mod.openslide, "open_slide", side_effect=open_flaky), \
            caplog.at_level(logging.WARNING):
        extract_patch_features_batch([h5], [_SLIDE], [out], model_type=ModelType.HOPTIMUS1, batch_size=7,
                                     use_gpu=False, num_workers=0, pin_memory=False, tile_size_check="off")
    with h5py.File(out) as f:
        rows = len(f["features"])
    assert rows == 20 - 20 // 3
    assert f"{20 // 3} of 20 tiles failed to load" in caplog.text
