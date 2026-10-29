"""A slide that fails in a batch doesn't stop the others, and the batch then raises SlideBatchError.

Covers extract_patch_features_batch(_multi), aggregate_slide_features_batch and
the extract_features CLI that calls them. Models are the deterministic fakes of
conftest.py; no weights are loaded.
"""
import contextlib
import gc
import weakref
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import mussel.cli.extract_features as cli
from mussel.models import ModelType
from mussel.utils import feature_extract as fe

_COMMON = dict(batch_size=7, use_gpu=False, pin_memory=False, tile_size_check="off")


def _two_slides(fakes, tmp_path):
    return [fakes.write_tiles(tmp_path / "a.patch.h5", fakes.coords[:30]),
            fakes.write_tiles(tmp_path / "b.patch.h5", fakes.coords[30:52])]


def _reference(fakes, tmp_path, h5, num_workers):
    ref = str(tmp_path / "ref.h5")
    with patch.object(fe, "get_model_factory", side_effect=fakes.factories()):
        fe.extract_patch_features(h5, fakes.slide, ref, model_type=ModelType.UNI, num_workers=num_workers,
                                  **_COMMON)
    return fakes.read(ref)


def test_unreadable_patch_h5_fails_only_its_slide(fakes, tmp_path):
    h5s = [str(tmp_path / "missing.patch.h5"), _two_slides(fakes, tmp_path)[1]]
    outs = [str(tmp_path / "0.h5"), str(tmp_path / "1.h5")]
    with patch.object(fe, "get_model_factory", side_effect=fakes.factories()):
        with pytest.raises(fe.SlideBatchError, match="1 of 2 slide.*missing") as info:
            fe.extract_patch_features_batch(h5s, [fakes.slide] * 2, outs, model_type=ModelType.UNI,
                                            num_workers=0, **_COMMON)
    assert list(info.value.failures) == [0]
    assert info.value.outputs == [None, outs[1]]
    np.testing.assert_array_equal(fakes.read(outs[1])[0], _reference(fakes, tmp_path, h5s[1], 0)[0])


@pytest.mark.parametrize("num_workers", [0, 2])
def test_slide_failing_mid_extraction_does_not_shift_the_next_slides_tiles(fakes, tmp_path, num_workers):
    """With the shared loader, the failed slide's unread batches are skipped, not read as the next slide's."""
    calls = []

    class FailsFirstBatch(fakes.FakeModel):
        def get_model_fun(self):
            inner = super().get_model_fun()

            def fun(x):
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError("CUDA out of memory")
                return inner(x)
            return fun

    h5s = _two_slides(fakes, tmp_path)
    outs = [str(tmp_path / "0.h5"), str(tmp_path / "1.h5")]
    with patch.object(fe, "get_model_factory", side_effect=fakes.factories(model_cls=FailsFirstBatch)):
        with pytest.raises(fe.SlideBatchError, match="CUDA out of memory") as info:
            fe.extract_patch_features_batch(h5s, [fakes.slide] * 2, outs, model_type=ModelType.UNI,
                                            num_workers=num_workers, **_COMMON)
    assert list(info.value.failures) == [0]
    assert not (tmp_path / "0.h5").exists()
    features, coords, _ = fakes.read(outs[1])
    ref_features, ref_coords, _ = _reference(fakes, tmp_path, h5s[1], num_workers)
    np.testing.assert_array_equal(coords, ref_coords)
    np.testing.assert_array_equal(features, ref_features)


@pytest.mark.parametrize("failure", ["model_load", "loader"])
def test_models_are_released_when_extraction_raises(fakes, tmp_path, failure):
    alive = weakref.WeakSet()

    class Tracked(fakes.FakeModel):
        def __init__(self, *args):
            super().__init__(*args)
            alive.add(self)

    def factory(model_type):
        if failure == "model_load" and model_type == ModelType.OPTIMUS:
            raise OSError("boom")
        return fakes.factories(model_cls=Tracked)(model_type)

    def broken_loader(*args, **kwargs):
        raise OSError("boom")
        yield  # a generator, like _per_slide_batches

    models = (ModelType.UNI, ModelType.OPTIMUS) if failure == "model_load" else (ModelType.UNI,)
    outs = {m: [str(tmp_path / f"{m.code}_{i}.h5") for i in range(2)] for m in models}
    loader = (patch.object(fe, "_per_slide_batches", side_effect=broken_loader) if failure == "loader"
              else contextlib.nullcontext())
    with patch.object(fe, "get_model_factory", side_effect=factory), loader:
        with pytest.raises(OSError, match="boom") as info:
            fe.extract_patch_features_batch_multi(_two_slides(fakes, tmp_path), [fakes.slide] * 2, outs,
                                                  num_workers=0, **_COMMON)
        gc.collect()
        assert info.value.__traceback__ is not None
        assert len(alive) == 0  # while the traceback (and its frames) is still alive


def _patch_features(path, n=6, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=rng.random((n, dim), dtype=np.float32))
        f.create_dataset("coords", data=np.arange(2 * n).reshape(n, 2))
    return str(path)


class _MeanSlideEncoder:
    def get_model_fun(self):
        return lambda feats: feats.mean(1)


def test_model_aggregation_unreadable_input_fails_only_its_slide(tmp_path):
    h5s = [_patch_features(tmp_path / "a.h5"), str(tmp_path / "missing.h5"), _patch_features(tmp_path / "c.h5", seed=1)]
    pts = [str(tmp_path / f"{i}.pt") for i in range(3)]
    alive = weakref.WeakSet()

    def load(*args, **kwargs):
        model = _MeanSlideEncoder()
        alive.add(model)
        return model

    with patch.object(fe, "_load_model", side_effect=load):
        with pytest.raises(fe.SlideBatchError, match="1 of 3 slide.*missing") as info:
            fe.aggregate_slide_features_batch(h5s, output_pt_paths=pts, aggregation_method="model",
                                              model_type=ModelType.CHIEF_SLIDE, model_path="x", use_gpu=False,
                                              tile_size_check="off", slide_batch_size=8)
        gc.collect()
        assert len(alive) == 0
    assert list(info.value.failures) == [1]
    assert info.value.outputs == (None, [pts[0], None, pts[2]])
    for i in (0, 2):
        with h5py.File(h5s[i]) as f:
            expected = f["features"][:].mean(0)
        np.testing.assert_allclose(torch.load(pts[i], weights_only=True).numpy().ravel(), expected, rtol=1e-6)


def test_mean_aggregation_partial_failure_raises_after_the_others(tmp_path):
    h5s = [_patch_features(tmp_path / "a.h5"), str(tmp_path / "missing.h5")]
    pts = [str(tmp_path / "a.pt"), str(tmp_path / "b.pt")]
    with pytest.raises(fe.SlideBatchError, match="1 of 2 slide.*'mean' aggregation.*missing"):
        fe.aggregate_slide_features_batch(h5s, output_pt_paths=pts, aggregation_method="mean")
    assert (tmp_path / "a.pt").exists() and not (tmp_path / "b.pt").exists()


def test_mean_aggregation_all_failing_keeps_the_all_failed_message(tmp_path):
    with pytest.raises(RuntimeError, match="All 1 slide\\(s\\) failed during 'mean' aggregation"):
        fe.aggregate_slide_features_batch([str(tmp_path / "missing.h5")], aggregation_method="mean")


@pytest.mark.parametrize("aggregation_method", ["identity", "mean"])
def test_extract_features_cli_processes_the_other_slides_then_raises(fakes, tmp_path, aggregation_method):
    h5s = [str(tmp_path / "missing.patch.h5"), _two_slides(fakes, tmp_path)[1]]
    cfg = OmegaConf.structured(cli.ExtractFeaturesConfig(
        patch_h5_paths=h5s, slide_paths=[fakes.slide] * 2, slide_ids=["B", "G"], output_dir=str(tmp_path / "out"),
        model_type=ModelType.UNI, aggregation_method=aggregation_method, batch_size=7, use_gpu=False,
        num_workers=0, tile_size_check="off"))
    with patch.object(fe, "get_model_factory", side_effect=fakes.factories()):
        with pytest.raises(RuntimeError, match="1 of 2 slide.*B: ") as info:
            cli._main_batch(OmegaConf.to_object(cfg))
    assert "G:" not in str(info.value)
    assert (tmp_path / "out" / "G.features.pt").exists()
    assert not (tmp_path / "out" / "B.features.pt").exists()
