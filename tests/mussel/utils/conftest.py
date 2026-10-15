"""Shared fakes for the patch-feature extraction tests (shared loader, multi-model pass, timings).

The ``fakes`` fixture gives the test slide, its patch H5 and a deterministic
stand-in for Mussel's patch encoders, so the tests exercise Mussel's real
dataset, loaders and writers without model weights.
"""
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
import torch
from torchvision import transforms

_TESTDATA = Path(__file__).parent.parent.parent / "testdata"
_SLIDE = str(_TESTDATA / "948176.svs")
_H5 = str(_TESTDATA / "948176.patch.h5")
_DEFAULT_PRE = transforms.Compose([transforms.Resize(16), transforms.ToTensor()])


class FakeModel:
    """Deterministic stand-in encoder: per-tile channel means and a few pixels, times ``scale``."""

    def __init__(self, preprocess=_DEFAULT_PRE, scale=1.0, model_type=None):
        self.pre, self.scale, self.model_type = preprocess, scale, model_type

    def get_preprocessing_fun(self):
        return self.pre

    def get_model_fun(self):
        scale = self.scale
        return lambda x: torch.cat([x.mean((2, 3)), x[:, :, 0, :4].flatten(1)], 1) * scale


def factories(spec=None, model_cls=FakeModel):
    """A stand-in for ``get_model_factory``: ``spec`` maps ModelType -> (preprocessing, scale).

    Use as ``patch.object(fe, "get_model_factory", side_effect=factories(spec))``.
    """

    class Factory:
        def __init__(self, model_type):
            self.model_type = model_type

        def get_model(self, *args, **kwargs):
            pre, scale = (spec or {}).get(self.model_type, (_DEFAULT_PRE, 1.0))
            return model_cls(pre, scale, self.model_type)

    return Factory


def write_tiles(path, coords):
    """A patch H5 at ``path`` with ``coords`` and the test slide's tile attributes."""
    with h5py.File(_H5) as f:
        attrs = dict(f["coords"].attrs)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("coords", data=coords)
        for k, v in attrs.items():
            ds.attrs[k] = v
    return str(path)


def read(path):
    """(features, coords, coords attributes) of a features H5."""
    with h5py.File(path) as f:
        return f["features"][:], f["coords"][:], dict(f["coords"].attrs)


@pytest.fixture
def fakes():
    with h5py.File(_H5) as f:
        coords = f["coords"][:]
    return SimpleNamespace(slide=_SLIDE, h5=_H5, coords=coords, default_pre=_DEFAULT_PRE,
                           FakeModel=FakeModel, factories=factories, write_tiles=write_tiles, read=read)
