"""Several patch models in one pass (extract_patch_features_batch_multi, extract_features model_types).

Each model's output must be exactly what extract_patch_features_batch writes for
it alone; models with identical preprocessing must share one decode.
"""
from pathlib import Path
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

_TESTDATA = Path(__file__).parent.parent.parent / "testdata"
_SLIDE = str(_TESTDATA / "948176.svs")
_H5 = str(_TESTDATA / "948176.patch.h5")

# Two models sharing a preprocessing (like H-optimus-0/1) and one with its own.
_PRE_A = transforms.Compose([transforms.Resize(16), transforms.ToTensor()])
_PRE_B = transforms.Compose([transforms.Resize(12), transforms.ToTensor()])
_SPEC = {
    ModelType.HOPTIMUS1: (_PRE_A, 1.0),
    ModelType.OPTIMUS: (_PRE_A, -2.0),
    ModelType.UNI: (_PRE_B, 3.0),
}


class _FakeModel:
    def __init__(self, model_type):
        self.pre, self.scale = _SPEC[model_type]

    def get_preprocessing_fun(self):
        return self.pre

    def get_model_fun(self):
        scale = self.scale
        return lambda x: torch.cat([x.mean((2, 3)), x[:, :, 0, :4].flatten(1)], 1) * scale


class _FakeFactory:
    def __init__(self, model_type):
        self.model_type = model_type

    def get_model(self, *args, **kwargs):
        return _FakeModel(self.model_type)


def _factory(model_type):
    return _FakeFactory(model_type)


def _slides(tmp_path):
    with h5py.File(_H5) as f:
        coords, attrs = f["coords"][:], dict(f["coords"].attrs)
    paths = []
    for name, c in (("a", coords[:150]), ("b", coords[::-1][:61])):
        p = tmp_path / f"{name}.patch.h5"
        with h5py.File(p, "w") as f:
            ds = f.create_dataset("coords", data=c)
            for k, v in attrs.items():
                ds.attrs[k] = v
        paths.append(str(p))
    return paths


def _read(p):
    with h5py.File(p) as f:
        return f["features"][:], f["coords"][:], dict(f["coords"].attrs)


@pytest.mark.parametrize("num_workers, is_test_run", [(2, False), (0, False), (2, True)])
def test_each_model_matches_its_single_model_run(tmp_path, num_workers, is_test_run):
    h5s = _slides(tmp_path)
    prec = {ModelType.HOPTIMUS1: "bfloat16", ModelType.OPTIMUS: "float32", ModelType.UNI: "float16"}
    common = dict(batch_size=7, use_gpu=False, num_workers=num_workers, pin_memory=False,
                  is_test_run=is_test_run, tile_size_check="off")
    with patch.object(fe, "get_model_factory", side_effect=_factory):
        outs = {m: [str(tmp_path / f"multi_{m.code}_{i}.h5") for i in range(2)] for m in _SPEC}
        fe.extract_patch_features_batch_multi(h5s, [_SLIDE] * 2, outs, embedding_precision=prec, **common)
        for m in _SPEC:
            single = [str(tmp_path / f"single_{m.code}_{i}.h5") for i in range(2)]
            fe.extract_patch_features_batch(h5s, [_SLIDE] * 2, single, model_type=m,
                                            embedding_precision=prec[m], **common)
            for a, b in zip(outs[m], single):
                fa, ca, aa = _read(a)
                fb, cb, ab = _read(b)
                assert fa.dtype == fb.dtype
                np.testing.assert_array_equal(fa.view(np.uint8), fb.view(np.uint8))
                np.testing.assert_array_equal(ca, cb)
                assert aa.keys() == ab.keys()


def test_models_with_identical_preprocessing_share_one_decode(tmp_path):
    h5s = _slides(tmp_path)
    reads = []
    real = fe.WholeSlideImageH5Dataset.__getitem__

    def counting(self, idx):
        reads.append(1)
        return real(self, idx)

    n_tiles = sum(len(h5py.File(h)["coords"]) for h in h5s)
    outs = {m: [str(tmp_path / f"{m.code}_{i}.h5") for i in range(2)] for m in _SPEC}
    with patch.object(fe, "get_model_factory", side_effect=_factory), \
            patch.object(fe.WholeSlideImageH5Dataset, "__getitem__", counting):
        fe.extract_patch_features_batch_multi(h5s, [_SLIDE] * 2, outs, batch_size=7, use_gpu=False,
                                              num_workers=0, pin_memory=False, tile_size_check="off")
    assert len(reads) == 2 * n_tiles  # one pass for the two H-optimus models, one for UNI (not 3)


def test_cli_model_types_writes_per_model_dirs(tmp_path):
    h5s = _slides(tmp_path)
    cfg = OmegaConf.structured(cli.ExtractFeaturesConfig(
        patch_h5_paths=h5s, slide_paths=[_SLIDE] * 2, slide_ids=["A", "B"], output_dir=str(tmp_path / "out"),
        model_types=[ModelType.HOPTIMUS1, ModelType.OPTIMUS], model_precisions={"hoptimus1": "bfloat16"},
        batch_size=7, use_gpu=False, num_workers=0, tile_size_check="off"))
    with patch.object(fe, "get_model_factory", side_effect=_factory):
        cli._main_batch(OmegaConf.to_object(cfg))
    for code in ("hoptimus1", "optimus"):
        for sid in ("A", "B"):
            h5 = tmp_path / "out" / code / f"{sid}.features.h5"
            pt = torch.load(tmp_path / "out" / code / f"{sid}.features.pt", weights_only=True)
            feats = _read(h5)[0]
            assert pt.shape[0] == len(feats)
    assert _read(tmp_path / "out" / "hoptimus1" / "A.features.h5")[0].dtype != np.float32  # bfloat16 stored


def test_cli_rejects_unknown_precision_model(tmp_path):
    cfg = cli.ExtractFeaturesConfig(patch_h5_paths=[_H5], slide_paths=[_SLIDE], slide_ids=["A"],
                                    output_dir=str(tmp_path), model_types=[ModelType.HOPTIMUS1],
                                    model_precisions={"uni": "float16"})
    with pytest.raises(ValueError, match="not in model_types"):
        cli._main_batch(cfg)
