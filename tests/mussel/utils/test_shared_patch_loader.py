"""extract_patch_features_batch with one DataLoader for all slides (shared_loader).

The shared loader must write exactly what the per-slide loaders write: same
rows, same order, same coords and attributes, also with is_test_run.
"""
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
from torchvision import transforms

from mussel.models import ModelType
from mussel.utils.feature_extract import (_OneSlideOpenConcat, _PerSlideBatches,
                                          extract_patch_features_batch)

_TESTDATA = Path(__file__).parent.parent.parent / "testdata"
_SLIDE = str(_TESTDATA / "948176.svs")
_H5 = str(_TESTDATA / "948176.patch.h5")


class _FakeModel:
    """A deterministic stand-in encoder: per-tile channel means and a few pixels."""

    def get_preprocessing_fun(self):
        return transforms.Compose([transforms.Resize(16), transforms.ToTensor()])

    def get_model_fun(self):
        return lambda x: torch.cat([x.mean((2, 3)), x[:, :, 0, :5].flatten(1)], 1)


class _FakeFactory:
    def get_model(self, *args, **kwargs):
        return _FakeModel()


def _slides(tmp_path):
    """Three 'slides' of different lengths: the test slide's tiles, a subset, a reversed copy."""
    with h5py.File(_H5) as f:
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs)
    out = []
    for name, c in (("full", coords), ("part", coords[:37]), ("rev", coords[::-1][:200])):
        p = tmp_path / f"{name}.patch.h5"
        with h5py.File(p, "w") as f:
            ds = f.create_dataset("coords", data=c)
            for k, v in attrs.items():
                ds.attrs[k] = v
        out.append(str(p))
    return out


def _run(tmp_path, h5s, tag, **kwargs):
    outs = [str(tmp_path / f"{tag}_{i}.h5") for i in range(len(h5s))]
    with patch("mussel.utils.feature_extract.get_model_factory", return_value=_FakeFactory()):
        extract_patch_features_batch(h5s, [_SLIDE] * len(h5s), outs, model_type=ModelType.HOPTIMUS1,
                                     batch_size=7, use_gpu=False, num_workers=2, pin_memory=False,
                                     tile_size_check="off", **kwargs)
    return outs


def _read(path):
    with h5py.File(path) as f:
        return f["features"][:], f["coords"][:], dict(f["coords"].attrs)


@pytest.mark.parametrize("is_test_run", [False, True])
def test_shared_loader_writes_what_per_slide_loaders_write(tmp_path, is_test_run):
    h5s = _slides(tmp_path)
    shared = _run(tmp_path, h5s, "shared", shared_loader=True, is_test_run=is_test_run)
    per_slide = _run(tmp_path, h5s, "per", shared_loader=False, is_test_run=is_test_run)
    for a, b, h5 in zip(shared, per_slide, h5s):
        fa, ca, aa = _read(a)
        fb, cb, ab = _read(b)
        np.testing.assert_array_equal(fa, fb)
        np.testing.assert_array_equal(ca, cb)
        assert aa.keys() == ab.keys()
        n = len(h5py.File(h5)["coords"])
        assert len(fa) == (min(n, 3 * 7) if is_test_run else n)


def test_per_slide_batches_never_mix_slides():
    s = _PerSlideBatches([5, 1, 7, 0, 3], batch_size=3)
    assert list(s) == [[0, 1, 2], [3, 4], [5], [6, 7, 8], [9, 10, 11], [12], [13, 14, 15]]
    assert s.per_slide() == [2, 1, 3, 0, 1] and len(s) == 7


def test_test_run_reads_only_three_batches_per_slide():
    s = _PerSlideBatches([10, 2, 30], batch_size=3, max_batches_per_slide=3)
    assert s.per_slide() == [3, 1, 3]
    assert list(s) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20]]


class _FakeSlide:
    def __init__(self, log):
        self.log = log

    def close(self):
        self.log.append("close")


class _FakeDataset(torch.utils.data.Dataset):
    """WholeSlideImageH5Dataset's worker_init / wsi contract."""

    def __init__(self, name, n, log):
        self.name, self.n, self.log, self.wsi = name, n, log, None

    def __len__(self):
        return self.n

    def worker_init(self, *args):
        self.log.append(f"open {self.name}")
        self.wsi = _FakeSlide(self.log)

    def __getitem__(self, i):
        assert self.wsi is not None, "read before the slide was opened"
        return self.name, i


def test_one_slide_open_at_a_time():
    log = []
    datasets = [_FakeDataset(n, k, log) for n, k in (("a", 4), ("b", 1), ("c", 3))]
    concat = _OneSlideOpenConcat(datasets)
    got = [concat[i] for b in _PerSlideBatches([4, 1, 3], 2) for i in b]
    assert got == [("a", 0), ("a", 1), ("a", 2), ("a", 3), ("b", 0), ("c", 0), ("c", 1), ("c", 2)]
    assert log == ["open a", "close", "open b", "close", "open c"]
    assert [d.wsi is not None for d in datasets] == [False, False, True]


def _write_tiles(path, coords, attrs):
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("coords", data=coords)
        for k, v in attrs.items():
            ds.attrs[k] = v
    return str(path)


def _tiles(tmp_path):
    with h5py.File(_H5) as f:
        coords, attrs = f["coords"][:], dict(f["coords"].attrs)
    return (_write_tiles(tmp_path / "a.patch.h5", coords[:30], attrs),
            _write_tiles(tmp_path / "empty.patch.h5", coords[:0], attrs),
            _write_tiles(tmp_path / "c.patch.h5", coords[30:52], attrs))


@pytest.mark.parametrize("shared_loader", [True, False])
def test_slide_without_tiles_keeps_later_slides_aligned(tmp_path, shared_loader):
    """A zero-tile slide gets no output, and the next slide still gets its own tiles."""
    a, empty, c = _tiles(tmp_path)
    outs = _run(tmp_path, [a, empty, c], f"z{shared_loader}", shared_loader=shared_loader)
    assert not Path(outs[1]).exists()
    for out, src in ((outs[0], a), (outs[2], c)):
        _, coords, _ = _read(out)
        np.testing.assert_array_equal(coords, h5py.File(src)["coords"][:])


@pytest.mark.parametrize("shared_loader", [True, False])
def test_unopenable_slide_fails_at_that_slide(tmp_path, shared_loader):
    """A slide that can't be opened raises while it is processed; earlier slides are written."""
    a, _, c = _tiles(tmp_path)
    outs = [str(tmp_path / f"u{shared_loader}_{i}.h5") for i in range(3)]
    with patch("mussel.utils.feature_extract.get_model_factory", return_value=_FakeFactory()):
        with pytest.raises(Exception):
            extract_patch_features_batch([a, c, a], [_SLIDE, str(tmp_path / "missing.svs"), _SLIDE], outs,
                                         model_type=ModelType.HOPTIMUS1, batch_size=7, use_gpu=False,
                                         num_workers=2, pin_memory=False, tile_size_check="off",
                                         shared_loader=shared_loader)
    assert [Path(o).exists() for o in outs] == [True, False, False]
