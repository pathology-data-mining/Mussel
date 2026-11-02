"""limit_to_indices semantics and which tile errors the datasets swallow.

* ``limit_to_indices=[]`` selects no tiles (it used to select every tile), and a
  numpy index array works (it used to raise: arrays have no truth value).
* Only reading / decoding a tile is guarded: an unreadable tile is skipped
  (None), but preprocessing errors propagate instead of silently dropping tiles.
"""
from pathlib import Path

import h5py
import numpy as np
import pytest
from torchvision import transforms

from mussel.datasets import WholeSlideImageH5Dataset, WholeSlideImageTileCoordDataset

_TESTDATA = Path(__file__).parent.parent.parent / "testdata"
_SLIDE = str(_TESTDATA / "948176.svs")
_H5 = str(_TESTDATA / "948176.patch.h5")
_PRE = transforms.Compose([transforms.Resize(16), transforms.ToTensor()])


def _coords(n=10):
    with h5py.File(_H5) as f:
        return f["coords"][:n], dict(f["coords"].attrs)


def _h5_dataset(limit, preprocess=_PRE):
    return WholeSlideImageH5Dataset(_H5, _SLIDE, preprocess=preprocess, use_imagenet_rgb_dist=False,
                                    limit_to_indices=limit, init_wsi_in_worker=False)


def _tc_dataset(limit, preprocess=_PRE):
    coords, attrs = _coords()
    return WholeSlideImageTileCoordDataset(coords, attrs, _SLIDE, preprocess=preprocess,
                                           use_imagenet_rgb_dist=False, limit_to_indices=limit,
                                           init_wsi_in_worker=False)


@pytest.mark.parametrize("make", [_h5_dataset, _tc_dataset])
def test_empty_limit_selects_no_tiles(make):
    assert len(make([])) == 0
    assert len(make(np.array([], dtype=np.int64))) == 0


@pytest.mark.parametrize("make", [_h5_dataset, _tc_dataset])
def test_numpy_limit_to_indices(make):
    coords, _ = _coords()
    ds = make(np.array([3, 1]))
    assert len(ds) == 2
    img, coord = ds[0]
    assert img is not None
    np.testing.assert_array_equal(coord, coords[3])
    np.testing.assert_array_equal(ds[1][1], coords[1])


def _failing_preprocess(img):
    raise RuntimeError("bad transform")


@pytest.mark.parametrize("make", [_h5_dataset, _tc_dataset])
def test_preprocessing_errors_propagate(make):
    ds = make(None, preprocess=_failing_preprocess)
    with pytest.raises(RuntimeError, match="bad transform"):
        ds[0]


@pytest.mark.parametrize("make", [_h5_dataset, _tc_dataset])
def test_unreadable_tile_is_still_skipped(make):
    ds = make(None)

    class _Broken:
        def read_region(self, *a):
            raise ValueError("imagecodecs: corrupt tile")

    ds.wsi = _Broken()
    img, coord = ds[0]
    assert img is None


@pytest.mark.parametrize("make", [_h5_dataset, _tc_dataset])
def test_memory_error_while_reading_propagates(make):
    ds = make(None)

    class _OOM:
        def read_region(self, *a):
            raise MemoryError()

    ds.wsi = _OOM()
    with pytest.raises(MemoryError):
        ds[0]
