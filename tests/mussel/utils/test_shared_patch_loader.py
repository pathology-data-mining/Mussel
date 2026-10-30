"""extract_patch_features_batch with one DataLoader for all slides (shared_loader).

The shared loader must write exactly what the per-slide loaders write, and both
what the single-slide path (extract_patch_features, process_dataset) writes:
same rows, same order, same coords and attributes, also with is_test_run.
"""
import gc
import multiprocessing
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from mussel.models import ModelType
from mussel.utils import feature_extract as fe
from mussel.utils.feature_extract import (_OneSlideOpenConcat, _PerSlideBatches,
                                          extract_patch_features, extract_patch_features_batch)


@pytest.fixture(autouse=True)
def _reap_dataloader_workers():
    """Let no test's DataLoader workers outlive it.

    Spawned workers still starting when their pool is torn down (an error-path test)
    compete with the next test's workers on a small CI runner.
    """
    yield
    gc.collect()
    for child in multiprocessing.active_children():
        child.join(timeout=30)
        if child.is_alive():  # hung: don't let it leak into the next test
            child.terminate()
            child.join(timeout=10)


_COMMON = dict(model_type=ModelType.HOPTIMUS1, batch_size=7, use_gpu=False, num_workers=2,
               pin_memory=False, tile_size_check="off")


def _slides(fakes, tmp_path):
    """Three 'slides' of different lengths: the test slide's tiles, a subset, a reversed copy."""
    c = fakes.coords
    return [fakes.write_tiles(tmp_path / f"{name}.patch.h5", coords)
            for name, coords in (("full", c), ("part", c[:37]), ("rev", c[::-1][:200]))]


def _run(fakes, tmp_path, h5s, tag, slides=None, **kwargs):
    outs = [str(tmp_path / f"{tag}_{i}.h5") for i in range(len(h5s))]
    with patch.object(fe, "get_model_factory", side_effect=fakes.factories()):
        extract_patch_features_batch(h5s, slides or [fakes.slide] * len(h5s), outs, **{**_COMMON, **kwargs})
    return outs


@pytest.mark.parametrize("is_test_run", [False, True])
def test_shared_and_per_slide_loaders_write_what_the_single_slide_path_writes(fakes, tmp_path, is_test_run):
    h5s = _slides(fakes, tmp_path)
    shared = _run(fakes, tmp_path, h5s, "shared", shared_loader=True, is_test_run=is_test_run)
    per_slide = _run(fakes, tmp_path, h5s, "per", shared_loader=False, is_test_run=is_test_run)
    for i, h5 in enumerate(h5s):
        ref = str(tmp_path / f"ref_{i}.h5")
        with patch.object(fe, "get_model_factory", side_effect=fakes.factories()):
            extract_patch_features(h5, fakes.slide, ref, is_test_run=is_test_run, **_COMMON)
        fr, cr, ar = fakes.read(ref)
        for out in (shared[i], per_slide[i]):
            f, c, a = fakes.read(out)
            np.testing.assert_array_equal(f, fr)
            np.testing.assert_array_equal(c, cr)
            assert a.keys() == ar.keys()
        n = len(h5py.File(h5)["coords"])
        assert len(fr) == (min(n, 3 * 7) if is_test_run else n)


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


def _tiles(fakes, tmp_path):
    c = fakes.coords
    return (fakes.write_tiles(tmp_path / "a.patch.h5", c[:30]),
            fakes.write_tiles(tmp_path / "empty.patch.h5", c[:0]),
            fakes.write_tiles(tmp_path / "c.patch.h5", c[30:52]))


@pytest.mark.parametrize("shared_loader", [True, False])
def test_slide_without_tiles_keeps_later_slides_aligned(fakes, tmp_path, shared_loader):
    """A zero-tile slide gets no output, and the next slide still gets its own tiles."""
    a, empty, c = _tiles(fakes, tmp_path)
    outs = _run(fakes, tmp_path, [a, empty, c], f"z{shared_loader}", shared_loader=shared_loader)
    assert not Path(outs[1]).exists()
    for out, src in ((outs[0], a), (outs[2], c)):
        np.testing.assert_array_equal(fakes.read(out)[1], h5py.File(src)["coords"][:])


@pytest.mark.parametrize("shared_loader", [True, False])
def test_unopenable_slide_fails_only_that_slide(fakes, tmp_path, shared_loader):
    """A slide that can't be opened fails; the slides before and after it are written."""
    a, _, c = _tiles(fakes, tmp_path)
    slides = [fakes.slide, str(tmp_path / "missing.svs"), fakes.slide]
    # The slide's own error, not e.g. a DataLoader worker that died.
    with pytest.raises(fe.SlideBatchError, match="missing.svs") as info:
        _run(fakes, tmp_path, [a, c, a], f"u{shared_loader}", slides=slides, shared_loader=shared_loader)
    assert list(info.value.failures) == [1]
    assert [Path(tmp_path / f"u{shared_loader}_{i}.h5").exists() for i in range(3)] == [True, False, True]
    np.testing.assert_array_equal(fakes.read(tmp_path / f"u{shared_loader}_2.h5")[1], h5py.File(a)["coords"][:])


@pytest.mark.parametrize("shared_loader", [True, False])
def test_unreadable_patch_h5_fails_only_that_slide(fakes, tmp_path, shared_loader):
    """A patch H5 that can't be read fails its slide; the slides before and after it are written."""
    a, _, c = _tiles(fakes, tmp_path)
    # The patch H5's own error, not e.g. a DataLoader worker that died.
    with pytest.raises(fe.SlideBatchError, match="missing.patch.h5") as info:
        _run(fakes, tmp_path, [a, str(tmp_path / "missing.patch.h5"), c], f"h{shared_loader}",
             shared_loader=shared_loader)
    assert list(info.value.failures) == [1]
    assert [Path(tmp_path / f"h{shared_loader}_{i}.h5").exists() for i in range(3)] == [True, False, True]
    np.testing.assert_array_equal(fakes.read(tmp_path / f"h{shared_loader}_2.h5")[1], h5py.File(c)["coords"][:])
