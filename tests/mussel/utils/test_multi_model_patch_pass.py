"""Several patch models in one pass (extract_patch_features_batch_multi, extract_features model_types).

Each model's output must be exactly what the single-slide path
(extract_patch_features, process_dataset) writes for it alone; models with
identical preprocessing must share one decode.
"""
import gc
import logging
import weakref
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torchvision import transforms

import mussel.cli.extract_features as cli
from mussel.models import ModelType
from mussel.utils import feature_extract as fe

# Two models sharing a preprocessing (like H-optimus-0/1) and one with its own.
_PRE_A = transforms.Compose([transforms.Resize(16), transforms.ToTensor()])
_PRE_B = transforms.Compose([transforms.Resize(12), transforms.ToTensor()])
_SPEC = {
    ModelType.HOPTIMUS1: (_PRE_A, 1.0),
    ModelType.OPTIMUS: (_PRE_A, -2.0),
    ModelType.UNI: (_PRE_B, 3.0),
}


def _slides(fakes, tmp_path):
    return [fakes.write_tiles(tmp_path / "a.patch.h5", fakes.coords[:150]),
            fakes.write_tiles(tmp_path / "b.patch.h5", fakes.coords[::-1][:61])]


def _outs(tmp_path, models, tag=""):
    return {m: [str(tmp_path / f"{tag}{m.code}_{i}.h5") for i in range(2)] for m in models}


def _multi(fakes, h5s, outs, spec=_SPEC, **kwargs):
    common = dict(batch_size=7, use_gpu=False, num_workers=0, pin_memory=False, tile_size_check="off")
    with patch.object(fe, "get_model_factory", side_effect=fakes.factories(spec)):
        fe.extract_patch_features_batch_multi(h5s, [fakes.slide] * len(h5s), outs, **{**common, **kwargs})


@pytest.mark.parametrize("num_workers, is_test_run", [(2, False), (0, False), (2, True)])
def test_each_model_matches_the_single_slide_path(fakes, tmp_path, num_workers, is_test_run):
    h5s = _slides(fakes, tmp_path)
    prec = {ModelType.HOPTIMUS1: "bfloat16", ModelType.OPTIMUS: "float32", ModelType.UNI: "float16"}
    outs = _outs(tmp_path, _SPEC, "multi_")
    _multi(fakes, h5s, outs, embedding_precision=prec, num_workers=num_workers, is_test_run=is_test_run)
    for m in _SPEC:
        for i, h5 in enumerate(h5s):
            ref = str(tmp_path / f"ref_{m.code}_{i}.h5")
            with patch.object(fe, "get_model_factory", side_effect=fakes.factories(_SPEC)):
                fe.extract_patch_features(h5, fakes.slide, ref, model_type=m, batch_size=7, use_gpu=False,
                                          num_workers=num_workers, pin_memory=False, is_test_run=is_test_run,
                                          embedding_precision=prec[m], tile_size_check="off")
            fa, ca, aa = fakes.read(outs[m][i])
            fb, cb, ab = fakes.read(ref)
            assert fa.dtype == fb.dtype
            np.testing.assert_array_equal(fa.view(np.uint8), fb.view(np.uint8))
            np.testing.assert_array_equal(ca, cb)
            assert aa.keys() == ab.keys()


def test_models_with_identical_preprocessing_share_one_decode(fakes, tmp_path):
    h5s = _slides(fakes, tmp_path)
    reads = []
    real = fe.WholeSlideImageH5Dataset.__getitem__

    def counting(self, idx):
        reads.append(1)
        return real(self, idx)

    n_tiles = sum(len(h5py.File(h)["coords"]) for h in h5s)
    with patch.object(fe.WholeSlideImageH5Dataset, "__getitem__", counting):
        _multi(fakes, h5s, _outs(tmp_path, _SPEC))
    assert len(reads) == 2 * n_tiles  # one pass for the two H-optimus models, one for UNI (not 3)


def test_cli_model_types_writes_per_model_dirs(fakes, tmp_path):
    h5s = _slides(fakes, tmp_path)
    cfg = OmegaConf.structured(cli.ExtractFeaturesConfig(
        patch_h5_paths=h5s, slide_paths=[fakes.slide] * 2, slide_ids=["A", "B"], output_dir=str(tmp_path / "out"),
        model_types=[ModelType.HOPTIMUS1, ModelType.OPTIMUS], model_precisions={"hoptimus1": "bfloat16"},
        batch_size=7, use_gpu=False, num_workers=0, tile_size_check="off"))
    with patch.object(fe, "get_model_factory", side_effect=fakes.factories(_SPEC)):
        cli._main_batch(OmegaConf.to_object(cfg))
    for code in ("hoptimus1", "optimus"):
        for sid in ("A", "B"):
            pt = torch.load(tmp_path / "out" / code / f"{sid}.features.pt", weights_only=True)
            assert pt.shape[0] == len(fakes.read(tmp_path / "out" / code / f"{sid}.features.h5")[0])
    assert fakes.read(tmp_path / "out" / "hoptimus1" / "A.features.h5")[0].dtype != np.float32  # bfloat16 stored


def test_cli_rejects_unknown_precision_model(fakes, tmp_path):
    cfg = cli.ExtractFeaturesConfig(patch_h5_paths=[fakes.h5], slide_paths=[fakes.slide], slide_ids=["A"],
                                    output_dir=str(tmp_path), model_types=[ModelType.HOPTIMUS1],
                                    model_precisions={"uni": "float16"})
    with pytest.raises(ValueError, match="not in model_types"):
        cli._main_batch(cfg)


def test_model_dir_and_model_paths_resolve_weights(fakes, tmp_path):
    seen = {}
    base = fakes.factories(_SPEC)

    class Recording(base):
        def get_model(self, model_path, *args, **kwargs):
            seen[self.model_type] = model_path
            return super().get_model(model_path, *args, **kwargs)

    with patch.object(fe, "get_model_factory", side_effect=Recording), \
            patch.object(fe, "get_model_path_from_dir", side_effect=lambda d, m: f"{d}/{m.code}.bin"):
        fe.extract_patch_features_batch_multi(
            _slides(fakes, tmp_path), [fakes.slide] * 2, _outs(tmp_path, (ModelType.HOPTIMUS1, ModelType.OPTIMUS)),
            model_dir="/models", model_paths={ModelType.OPTIMUS: "/explicit/optimus.pth"},
            batch_size=7, use_gpu=False, num_workers=0, pin_memory=False, tile_size_check="off")
    assert seen == {ModelType.HOPTIMUS1: "/models/hoptimus1.bin", ModelType.OPTIMUS: "/explicit/optimus.pth"}


def test_cli_rejects_single_model_path_with_model_types(fakes, tmp_path):
    cfg = cli.ExtractFeaturesConfig(patch_h5_paths=[fakes.h5], slide_paths=[fakes.slide], slide_ids=["A"],
                                    output_dir=str(tmp_path), model_types=[ModelType.HOPTIMUS1, ModelType.OPTIMUS],
                                    model_path="/models/one.pth")
    with pytest.raises(ValueError, match="model_paths"):
        cli._main_batch(cfg)


def test_preprocessing_key_groups_only_provably_identical_transforms():
    a, b = ModelType.HOPTIMUS1, ModelType.OPTIMUS
    key = fe._preprocessing_key
    resize = lambda: transforms.Compose([transforms.Resize(16), transforms.ToTensor()])  # noqa: E731
    assert key(resize(), a) == key(resize(), b)  # standard transforms: equal parameters, equal key
    assert key(_PRE_A, a) != key(_PRE_B, b)
    # Lambda's repr is "Lambda()" whatever it wraps: never grouped.
    la = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x)])
    lb = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: -x)])
    assert repr(la) == repr(lb) and key(la, a) != key(lb, b)
    custom = lambda img: img  # noqa: E731
    assert key(custom, a) != key(lambda img: img, b)
    assert key(None, a) == key(None, b) == "default"


def test_freed_custom_transforms_are_not_grouped_when_their_address_is_reused(fakes, tmp_path):
    """Each model's custom transform is freed before the next model loads, so the
    next one can land at the same address; the models must still get separate passes."""
    class Custom:  # not a torchvision Compose: no provable equality
        def __init__(self, sign):
            self.sign, self.to_tensor = sign, transforms.Compose([transforms.Resize(16), transforms.ToTensor()])

        def __call__(self, img):
            x = self.to_tensor(img)
            return x if self.sign > 0 else 1 - x

    class FreshCustom(fakes.FakeModel):
        def get_preprocessing_fun(self):  # a new object per call, like timm's create_transform
            return Custom(1 if self.model_type == ModelType.HOPTIMUS1 else -1)

    outs = _outs(tmp_path, (ModelType.HOPTIMUS1, ModelType.OPTIMUS))
    # Every object at one address: the worst case of address reuse, made deterministic.
    with patch.object(fe, "get_model_factory", side_effect=fakes.factories(model_cls=FreshCustom)), \
            patch.object(fe, "id", create=True, side_effect=lambda o: 1):
        fe.extract_patch_features_batch_multi(
            _slides(fakes, tmp_path), [fakes.slide] * 2, outs,
            batch_size=7, use_gpu=False, num_workers=0, pin_memory=False, tile_size_check="off")
    a, b = fakes.read(outs[ModelType.HOPTIMUS1][0])[0], fakes.read(outs[ModelType.OPTIMUS][0])[0]
    assert not np.allclose(a, b)  # OPTIMUS got its own (1 - x) preprocessing


def test_models_differing_only_in_a_lambda_get_separate_passes(fakes, tmp_path):
    lam = lambda f: transforms.Compose([transforms.Resize(16), transforms.ToTensor(), transforms.Lambda(f)])  # noqa: E731
    spec = {ModelType.HOPTIMUS1: (lam(lambda x: x), 1.0), ModelType.OPTIMUS: (lam(lambda x: 1 - x), 1.0)}
    outs = _outs(tmp_path, spec)
    _multi(fakes, _slides(fakes, tmp_path), outs, spec=spec)
    a, b = fakes.read(outs[ModelType.HOPTIMUS1][0])[0], fakes.read(outs[ModelType.OPTIMUS][0])[0]
    assert not np.allclose(a, b)  # each model got its own preprocessing: 1 - x shifts the means


@pytest.mark.parametrize("slide_encoder", [ModelType.TITAN_SLIDE, ModelType.ABMIL_SLIDE])
def test_slide_encoders_are_rejected(fakes, tmp_path, slide_encoder):
    with pytest.raises(ValueError, match="slide encoders"):
        _multi(fakes, _slides(fakes, tmp_path), _outs(tmp_path, (ModelType.HOPTIMUS1, slide_encoder)))


def test_only_one_groups_models_are_alive_at_a_time(fakes, tmp_path):
    """Models of other groups are released, and reloaded when their group runs."""
    alive, loads, seen = weakref.WeakSet(), [], []

    class Tracked(fakes.FakeModel):
        def __init__(self, *args):
            super().__init__(*args)
            loads.append(self.model_type)
            alive.add(self)

        def get_model_fun(self):
            inner, me = super().get_model_fun(), self.model_type

            def fun(x):
                gc.collect()
                seen.append((me, tuple(sorted(m.model_type.name for m in alive))))
                return inner(x)
            return fun

    with patch.object(fe, "get_model_factory", side_effect=fakes.factories(_SPEC, model_cls=Tracked)):
        fe.extract_patch_features_batch_multi(_slides(fakes, tmp_path), [fakes.slide] * 2, _outs(tmp_path, _SPEC),
                                              batch_size=7, use_gpu=False, num_workers=0, pin_memory=False,
                                              tile_size_check="off")
    by_model = {}
    for me, live in seen:
        by_model.setdefault(me, set()).add(live)
    # H-optimus-0/1 share a pass with UNI released; UNI runs alone after they are released.
    assert by_model[ModelType.HOPTIMUS1] == by_model[ModelType.OPTIMUS] == {("HOPTIMUS1", "OPTIMUS")}
    assert by_model[ModelType.UNI] == {("UNI",)}
    assert sorted(m.name for m in loads) == ["HOPTIMUS1", "OPTIMUS", "UNI", "UNI"]  # UNI reloaded once


@pytest.mark.parametrize("multi", [True, False])
def test_cli_slide_without_tiles_does_not_abort_the_batch(fakes, tmp_path, multi, caplog):
    c = fakes.coords
    paths = [fakes.write_tiles(tmp_path / f"{n}.patch.h5", coords)
             for n, coords in (("A", c[:20]), ("E", c[:0]), ("C", c[20:35]))]
    extra = (dict(model_types=[ModelType.HOPTIMUS1, ModelType.OPTIMUS]) if multi
             else dict(model_type=ModelType.HOPTIMUS1))
    cfg = OmegaConf.structured(cli.ExtractFeaturesConfig(
        patch_h5_paths=paths, slide_paths=[fakes.slide] * 3, slide_ids=["A", "E", "C"], output_dir=str(tmp_path / "out"),
        batch_size=7, use_gpu=False, num_workers=0, tile_size_check="off", **extra))
    with patch.object(fe, "get_model_factory", side_effect=fakes.factories(_SPEC)), caplog.at_level(logging.WARNING):
        cli._main_batch(OmegaConf.to_object(cfg))
    out = tmp_path / "out" / ("hoptimus1" if multi else "")
    assert (out / "A.features.pt").exists() and (out / "C.features.pt").exists()
    assert not (out / "E.features.pt").exists()
    assert "skipped (no tiles)" in caplog.text


@pytest.mark.parametrize("multi", [True, False])
def test_cli_fails_when_no_slide_writes_features(fakes, tmp_path, multi):
    paths = [fakes.write_tiles(tmp_path / f"{n}.patch.h5", fakes.coords[:0]) for n in ("E1", "E2")]
    extra = (dict(model_types=[ModelType.HOPTIMUS1, ModelType.OPTIMUS]) if multi
             else dict(model_type=ModelType.HOPTIMUS1))
    cfg = OmegaConf.structured(cli.ExtractFeaturesConfig(
        patch_h5_paths=paths, slide_paths=[fakes.slide] * 2, slide_ids=["E1", "E2"], output_dir=str(tmp_path / "out"),
        batch_size=7, use_gpu=False, num_workers=0, tile_size_check="off", **extra))
    with patch.object(fe, "get_model_factory", side_effect=fakes.factories(_SPEC)):
        with pytest.raises(RuntimeError, match="no output files"):
            cli._main_batch(OmegaConf.to_object(cfg))


def test_cli_model_types_passes_shared_loader(fakes, tmp_path):
    cfg = OmegaConf.structured(cli.ExtractFeaturesConfig(
        patch_h5_paths=[fakes.h5], slide_paths=[fakes.slide], slide_ids=["A"], output_dir=str(tmp_path / "out"),
        model_types=[ModelType.HOPTIMUS1, ModelType.OPTIMUS], shared_loader=False, use_gpu=False))
    with patch.object(cli, "extract_patch_features_batch_multi") as run, patch.object(cli, "_write_pts"), \
            patch.object(cli, "_raise_if_no_features"):
        cli._main_batch(OmegaConf.to_object(cfg))
    assert run.call_args.kwargs["shared_loader"] is False
