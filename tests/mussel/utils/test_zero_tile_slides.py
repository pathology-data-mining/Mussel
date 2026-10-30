"""Slides with no tile features (no tissue tiles, or every tile failed to load).

Patch-feature extraction must still write an output H5 (empty ``features``
with the model's feature dim, and ``coords`` with the source coords attributes)
at the path it returns, so callers that open it don't crash. Aggregating one
slide of 0 tiles is a clear NoTilesError; batch aggregation skips such slides
with a warning ("skipped (no tiles)"), not counting them as failed.
"""
import logging
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from mussel.cli.extract_features import _write_pts
from mussel.datasets import WholeSlideImageH5Dataset
from mussel.models import ModelType
from mussel.utils.feature_extract import (NoTilesError, _write_patch_batches,
                                          aggregate_sample_features,
                                          aggregate_slide_features,
                                          aggregate_slide_features_batch,
                                          process_dataset, save_features)
from mussel.utils.ml import collate_features

_TESTDATA = Path(__file__).parent.parent.parent / "testdata"
_SLIDE = str(_TESTDATA / "948176.svs")
_H5 = str(_TESTDATA / "948176.patch.h5")
_PRE = transforms.Compose([transforms.Resize(16), transforms.ToTensor()])


def _patch_h5(tmp_path, n):
    with h5py.File(_H5) as f:
        coords, attrs = f["coords"][:n], dict(f["coords"].attrs)
    p = tmp_path / f"s{n}.patch.h5"
    with h5py.File(p, "w") as f:
        ds = f.create_dataset("coords", data=coords)
        for k, v in attrs.items():
            ds.attrs[k] = v
    return str(p), attrs


class _Broken:
    def read_region(self, *a):
        raise ValueError("imagecodecs: corrupt tile")


class _FakeModel:
    def get_preprocessing_fun(self):
        return _PRE

    def get_model_fun(self):
        return lambda x: x.mean((2, 3))


class _FakeFactory:
    def get_model(self, *args, **kwargs):
        return _FakeModel()


def _assert_empty_output(path, attrs, feature_shape=(0, 3)):
    """Empty features keep the model's feature dim (_FakeModel: 3 channels -> D=3)."""
    with h5py.File(path, "r") as f:
        assert f["features"].shape == feature_shape
        assert f["coords"].shape == (0, 2)
        assert f["coords"].attrs["patch_size"] == attrs["patch_size"]


@pytest.mark.parametrize("n_tiles", [0, 4])  # no tiles / every tile fails to load
def test_h5_processor_writes_empty_output(tmp_path, n_tiles):
    patch_h5, attrs = _patch_h5(tmp_path, n_tiles)
    ds = WholeSlideImageH5Dataset(patch_h5, _SLIDE, preprocess=_PRE, use_imagenet_rgb_dist=False,
                                  init_wsi_in_worker=False)
    ds.wsi = _Broken()
    loader = DataLoader(ds, batch_size=2, num_workers=0, collate_fn=collate_features)
    out = str(tmp_path / "out.h5")
    result = process_dataset(ds, loader, _FakeModel().get_model_fun(), output_h5_path=out,
                             patch_h5_path=patch_h5)
    assert result.output_path == out
    _assert_empty_output(out, attrs)


def test_save_features_single_step_writes_empty_h5_and_pt(tmp_path):
    patch_h5, attrs = _patch_h5(tmp_path, 0)
    out_h5, out_pt = str(tmp_path / "s.features.h5"), str(tmp_path / "s.features.pt")
    with patch("mussel.utils.feature_extract.get_model_factory", return_value=_FakeFactory()):
        save_features(slide_path=_SLIDE, patch_h5_path=patch_h5, output_h5_path=out_h5,
                      output_pt_path=out_pt, model_type=ModelType.HOPTIMUS1, use_gpu=False,
                      num_workers=0, batch_size=4, tile_size_check="off")
    _assert_empty_output(out_h5, attrs)
    assert torch.load(out_pt, weights_only=True).shape[0] == 0


def test_write_patch_batches_writes_empty_outputs(tmp_path):
    patch_h5, attrs = _patch_h5(tmp_path, 3)
    outs = [str(tmp_path / "a.h5"), str(tmp_path / "b.h5")]
    empty_batches = [(torch.empty(0), np.empty((0, 2)))]
    fun = _FakeModel().get_model_fun()
    _write_patch_batches(empty_batches, [fun, fun], [None, np.float16], patch_h5, outs, False,
                         _SLIDE, _PRE)
    for out in outs:
        _assert_empty_output(out, attrs)
    with h5py.File(outs[1]) as f:
        assert f["features"].dtype == np.float16


def _features_h5(tmp_path, name, n, dim=4):
    p = tmp_path / f"{name}.patch_features.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("features", data=np.ones((n, dim) if n else (0, 0), np.float32))
        c = f.create_dataset("coords", data=np.arange(2 * n, dtype=np.int64).reshape(n, 2))
        c.attrs["patch_size"] = 224
    return str(p)


@pytest.mark.parametrize("method", ["mean", "max"])
def test_aggregating_zero_tiles_is_a_clear_error(tmp_path, method):
    with pytest.raises(NoTilesError, match="0 tiles"):
        aggregate_slide_features(_features_h5(tmp_path, "empty", 0), aggregation_method=method,
                                 output_pt_path=str(tmp_path / "x.pt"))


def test_identity_of_zero_tiles_is_empty(tmp_path):
    out_pt = str(tmp_path / "x.pt")
    aggregate_slide_features(_features_h5(tmp_path, "empty", 0), aggregation_method="identity",
                             output_pt_path=out_pt)
    assert torch.load(out_pt, weights_only=True).shape[0] == 0


def _assert_skipped_not_failed(caplog, name="empty.patch_features"):
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert f"Skipped (no tiles): 1 slides: {name}" in caplog.text
    assert "Failed" not in caplog.text


@pytest.mark.parametrize("method", ["mean", "max"])
def test_batch_mean_max_skips_empty_slide(tmp_path, caplog, method):
    inputs = [_features_h5(tmp_path, "empty", 0), _features_h5(tmp_path, "full", 3)]
    pts = [str(tmp_path / "empty.pt"), str(tmp_path / "full.pt")]
    with caplog.at_level(logging.WARNING):
        aggregate_slide_features_batch(inputs, output_pt_paths=pts, aggregation_method=method)
    assert not Path(pts[0]).exists()
    assert torch.load(pts[1], weights_only=True).shape == (1, 4)
    _assert_skipped_not_failed(caplog)


def test_batch_of_only_empty_slides_raises(tmp_path, caplog):
    """No slide with tiles is more likely a wrong mpp or slide list than empty slides."""
    inputs = [_features_h5(tmp_path, "empty", 0)]
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="none of the 1 slide.*had tiles"):
        aggregate_slide_features_batch(inputs, output_pt_paths=[str(tmp_path / "e.pt")],
                                       aggregation_method="mean")
    assert "Skipped (no tiles): 1 slides: empty.patch_features" in caplog.text


def test_batch_slide_encoder_of_only_empty_slides_raises(tmp_path):
    inputs = [_features_h5(tmp_path, "empty", 0)]
    with patch("mussel.utils.feature_extract._load_model", return_value=_FakeTitanModel()), \
            pytest.raises(RuntimeError, match="none of the 1 slide.*had tiles"):
        aggregate_slide_features_batch(inputs, output_pt_paths=[str(tmp_path / "e.pt")],
                                       aggregation_method="model", model_type=ModelType.TITAN_SLIDE,
                                       use_gpu=False, tile_size_check="off")


class _FakeSlideModel:
    def get_model_fun(self):
        return lambda features, coords: features.mean(1)


class _FakeTitanModel:
    def get_model_fun(self):
        return lambda features, coords, patch_size: features.mean(1)


class _FakeGenericSlideModel:
    def get_model_fun(self):
        return lambda features: features.mean(1)


@pytest.mark.parametrize("model_type, fake", [
    (ModelType.GIGAPATH_SLIDE, _FakeSlideModel()),
    (ModelType.TITAN_SLIDE, _FakeTitanModel()),
    (ModelType.PRISM_SLIDE, _FakeGenericSlideModel()),  # generic (stacked batch) branch
])
def test_batch_slide_encoder_skips_empty_slide(tmp_path, caplog, model_type, fake):
    inputs = [_features_h5(tmp_path, "empty", 0), _features_h5(tmp_path, "full", 3)]
    pts = [str(tmp_path / "empty.pt"), str(tmp_path / "full.pt")]
    with patch("mussel.utils.feature_extract._load_model", return_value=fake), \
            caplog.at_level(logging.WARNING):
        aggregate_slide_features_batch(inputs, output_pt_paths=pts, aggregation_method="model",
                                       model_type=model_type, use_gpu=False,
                                       tile_size_check="off")
    assert not Path(pts[0]).exists()
    assert torch.load(pts[1], weights_only=True).shape == (1, 4)
    assert "empty.patch_features: no tile features to aggregate" in caplog.text
    _assert_skipped_not_failed(caplog)


def test_write_pts_all_empty_slides_raises(tmp_path, caplog):
    """Each empty slide still gets its empty .pt, then the run fails: no slide had tiles."""
    h5s = [_features_h5(tmp_path, "a", 0), _features_h5(tmp_path, "b", 0)]
    pts = [str(tmp_path / "a.pt"), str(tmp_path / "b.pt")]
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="no output files"):
        _write_pts(h5s, pts)
    assert all(torch.load(p, weights_only=True).shape[0] == 0 for p in pts)
    assert "Skipped (no tiles): 2 slides" in caplog.text


def test_sample_aggregation_skips_empty_slides(tmp_path, caplog):
    """An empty slide's (0, 0) features don't break concatenation with (N, D) slides."""
    feats = [np.empty((0, 0), np.float32), np.ones((3, 4), np.float32)]
    coords = [np.empty((0, 2), np.int64), np.zeros((3, 2), np.int64)]
    with caplog.at_level(logging.WARNING):
        result = aggregate_sample_features(feats, coords, ["s", "s"])
    assert result["s"][0].shape == (3, 4)
    assert result["s"][1].shape == (3, 2)
    assert "no tiles" in caplog.text


def test_sample_aggregation_of_only_empty_slides_is_a_clear_error():
    with pytest.raises(NoTilesError, match="sample 's'"):
        aggregate_sample_features([np.empty((0, 0), np.float32)] * 2, [np.empty((0, 2))] * 2, ["s", "s"])
