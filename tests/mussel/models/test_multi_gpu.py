"""Tests for multi-GPU (``nn.DataParallel``) wrapping in TorchModel.

No GPU is needed: ``nn.DataParallel`` is constructible on a CPU-only host (it
then just forwards to the wrapped module), which is enough to exercise
attribute access, ``save()`` and preprocessing through the wrapper.  The
wrapping decision in ``TorchModel.__init__`` is exercised by faking CUDA
availability with a parameter-less module, so no tensor is ever moved to a
CUDA device.
"""

import importlib
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

import mussel.models  # noqa: F401  # triggers all @register_model decorators
from mussel.models.abmil import ABMILSlideModel
from mussel.models.base import TorchModel, _TimmHfHubBase
from mussel.models.chief import CHIEFSlideEncoderModel
from mussel.models.clip import ClipModel
from mussel.models.conch import TitanSlideEncoderModel
from mussel.models.feather import FeatherSlideEncoderModel
from mussel.models.genbio import GenBioPathFMModel
from mussel.models.gigapath import GigapathSlideEncoderModel
from mussel.models.gpfm import GPFMModel
from mussel.models.madeleine import MadeleineSlideEncoderModel
from mussel.models.optimus import HOptimus1Model, OptimusModel
from mussel.models.prism import PRISMSlideEncoderModel
from mussel.models.uni import Uni2Model, UniModel
from mussel.models.virchow import Virchow2Model, VirchowModel

# ``torch.nn.parallel.data_parallel`` (attribute access) is the function of that
# name, which shadows the submodule; fetch the module itself.
dp_module = importlib.import_module("torch.nn.parallel.data_parallel")

SLIDE_ENCODERS = [
    TitanSlideEncoderModel,
    GigapathSlideEncoderModel,
    CHIEFSlideEncoderModel,
    FeatherSlideEncoderModel,
    MadeleineSlideEncoderModel,
    PRISMSlideEncoderModel,
    ABMILSlideModel,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Tiny(nn.Module):
    """Small deterministic model with a timm-style ``pretrained_cfg``."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear = nn.Linear(4, 3)
        self.pretrained_cfg = {
            "input_size": (3, 32, 32),
            "interpolation": "bicubic",
            "mean": (0.5, 0.5, 0.5),
            "std": (0.25, 0.25, 0.25),
            "crop_pct": 1.0,
        }
        self.transform = "sentinel-transform"

    def forward(self, x):
        return self.linear(x)


class _ParamFree(nn.Module):
    """Parameter-less module: ``.to("cuda:0")`` is a no-op without CUDA."""

    def forward(self, x):
        return x


def _tiny_inference():
    return _Tiny().train(False)


def _make(cls, obj):
    """Instantiate *cls* without __init__, as test_model_classes does."""
    inst = cls.__new__(cls)
    inst.obj = obj
    inst.device = torch.device("cpu")
    inst.use_gpu = False
    return inst


@pytest.fixture
def restore_torch_globals():
    """TorchModel(use_gpu=True) flips global torch flags; restore them."""
    benchmark = torch.backends.cudnn.benchmark
    precision = torch.get_float32_matmul_precision()
    yield
    torch.backends.cudnn.benchmark = benchmark
    torch.set_float32_matmul_precision(precision)


def _init_with_fake_cuda(cls, gpu_device_id):
    """Run TorchModel.__init__ for *cls* as if CUDA with 2 GPUs were present."""
    inst = cls.__new__(cls)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=2),
        patch.object(dp_module, "_check_balance", lambda device_ids: None),
    ):
        TorchModel.__init__(
            inst, "unused", _ParamFree(), use_gpu=True, gpu_device_id=gpu_device_id
        )
    return inst


# ---------------------------------------------------------------------------
# Wrapping decision
# ---------------------------------------------------------------------------


class TestWrappingDecision:
    def test_patch_encoder_wrapped_for_multiple_gpus(self, restore_torch_globals):
        m = _init_with_fake_cuda(TorchModel, [0, 1])
        assert isinstance(m.obj, nn.DataParallel)
        assert m.obj.device_ids == [0, 1]
        assert isinstance(m.unwrapped, _ParamFree)
        assert m.device == torch.device("cuda", 0)

    @pytest.mark.parametrize("cls", SLIDE_ENCODERS, ids=lambda c: c.__name__)
    def test_slide_encoders_never_wrapped(self, cls, restore_torch_globals):
        assert cls.supports_data_parallel is False
        m = _init_with_fake_cuda(cls, [0, 1])
        assert not isinstance(m.obj, nn.DataParallel)
        assert m.device == torch.device("cuda", 0)

    @pytest.mark.parametrize(
        "cls",
        [
            TorchModel,
            VirchowModel,
            Virchow2Model,
            UniModel,
            Uni2Model,
            GPFMModel,
            ClipModel,
            HOptimus1Model,
            OptimusModel,
            GenBioPathFMModel,
        ],
        ids=lambda c: c.__name__,
    )
    def test_patch_encoders_support_data_parallel(self, cls):
        assert cls.supports_data_parallel is True

    @pytest.mark.parametrize("gpu_device_id", [None, 0, [0], [1]])
    def test_single_gpu_not_wrapped(self, gpu_device_id, restore_torch_globals):
        m = _init_with_fake_cuda(TorchModel, gpu_device_id)
        assert not isinstance(m.obj, nn.DataParallel)

    def test_cpu_with_gpu_list_not_wrapped(self):
        m = TorchModel("unused", _Tiny(), use_gpu=False, gpu_device_id=[0, 1])
        assert not isinstance(m.obj, nn.DataParallel)


# ---------------------------------------------------------------------------
# Single-device behaviour is unchanged
# ---------------------------------------------------------------------------


class TestSingleDeviceUnchanged:
    @pytest.mark.parametrize("gpu_device_id", [None, 0])
    def test_not_wrapped_and_output_identical(self, gpu_device_id):
        ref = _tiny_inference()
        x = torch.randn(5, 4)
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.float16):
            expected = ref(x).float()

        m = TorchModel("unused", _Tiny(), use_gpu=False, gpu_device_id=gpu_device_id)
        assert not isinstance(m.obj, nn.DataParallel)
        assert m.unwrapped is m.obj
        out = m.get_model_fun()(x)
        assert torch.equal(out, expected)

    def test_dataparallel_wrapper_forward_identical(self):
        """Forward through the (CPU pass-through) wrapper gives the same output."""
        x = torch.randn(5, 4)
        plain = _make(TorchModel, _tiny_inference())
        wrapped = _make(
            TorchModel, nn.DataParallel(_tiny_inference(), device_ids=[0, 1])
        )
        assert torch.equal(plain.get_model_fun()(x), wrapped.get_model_fun()(x))


# ---------------------------------------------------------------------------
# Attribute access / save through a DataParallel wrapper
# ---------------------------------------------------------------------------


def _wrapped_tiny():
    return nn.DataParallel(_tiny_inference(), device_ids=[0, 1])


class TestAttributeAccessThroughWrapper:
    def test_unwrapped_property(self):
        inner = _Tiny()
        m = _make(TorchModel, nn.DataParallel(inner, device_ids=[0, 1]))
        assert m.unwrapped is inner

    @pytest.mark.parametrize(
        "cls",
        [_TimmHfHubBase, VirchowModel, Virchow2Model, UniModel, Uni2Model, GPFMModel],
        ids=lambda c: c.__name__,
    )
    def test_timm_preprocessing(self, cls):
        m = _make(cls, _wrapped_tiny())
        transform = m.get_preprocessing_fun()
        assert callable(transform)

    def test_genbio_preprocessing(self):
        m = _make(GenBioPathFMModel, _wrapped_tiny())
        assert m.get_preprocessing_fun() == "sentinel-transform"

    def test_save_pickles_unwrapped_model(self, tmp_path):
        m = _make(TorchModel, _wrapped_tiny())
        path = tmp_path / "model.pkl"
        m.save(str(path))
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        assert not isinstance(loaded, nn.DataParallel)
        assert isinstance(loaded, _Tiny)


# ---------------------------------------------------------------------------
# CLIP forward must survive DataParallel replication
# ---------------------------------------------------------------------------


class _FakeClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(4, 3, bias=False)

    def encode_image(self, x):
        return self.visual(x)

    def forward(self, *args, **kwargs):  # open_clip's forward needs image+text
        raise AssertionError("CLIP.forward must not be called for image features")


def _cpu_replicate(root: nn.Module, scale: float) -> nn.Module:
    """Emulate ``torch.nn.parallel.replicate`` for one replica on CPU.

    Mirrors what replicate() does: ``_replicate_for_data_parallel()`` (a
    shallow ``__dict__`` copy) on every module, children re-pointed at their
    replicas, and parameters replaced by per-device copies (here: scaled
    copies, so a replica that wrongly reuses the original's weights is
    detectable).
    """
    modules = list(root.modules())
    replicas = {id(m): m._replicate_for_data_parallel() for m in modules}
    for m in modules:
        r = replicas[id(m)]
        for name, child in m._modules.items():
            if child is not None:
                r._modules[name] = replicas[id(child)]
        for name, p in m._parameters.items():
            setattr(r, name, None if p is None else p.detach() * scale)
    return replicas[id(root)]


class TestClipDataParallel:
    def _make_clip(self):
        fake = _FakeClip()
        preprocess = object()
        with patch(
            "mussel.models.clip.open_clip.create_model_and_transforms",
            return_value=(fake, None, preprocess),
        ):
            m = ClipModel("hf-hub:fake/clip", use_gpu=False)
        return m, fake, preprocess

    def test_forward_is_encode_image_and_not_patched_on_instance(self):
        m, fake, preprocess = self._make_clip()
        assert m.get_preprocessing_fun() is preprocess
        assert "forward" not in m.obj.__dict__
        x = torch.randn(2, 4)
        with torch.no_grad():
            assert torch.equal(m.obj(x), fake.encode_image(x))

    def test_replica_uses_its_own_weights(self):
        m, fake, _ = self._make_clip()
        replica = _cpu_replicate(m.obj, scale=2.0)
        x = torch.randn(2, 4)
        with torch.no_grad():
            expected = fake.encode_image(x) * 2.0
            assert torch.allclose(replica(x), expected)


# ---------------------------------------------------------------------------
# GenBio loads onto the requested GPU
# ---------------------------------------------------------------------------


class TestGenBioDevice:
    @pytest.mark.parametrize(
        "use_gpu, gpu_device_id, expected",
        [
            (True, [2, 3], "cuda:2"),
            (True, 1, "cuda:1"),
            (True, None, "cuda"),
            (False, [2, 3], "cpu"),
            (False, None, "cpu"),
        ],
    )
    def test_load_device(self, use_gpu, gpu_device_id, expected):
        with (
            patch.object(
                GenBioPathFMModel, "_load_genbio", return_value=SimpleNamespace()
            ) as load,
            patch.object(TorchModel, "__init__", return_value=None),
        ):
            GenBioPathFMModel(
                "hf-hub:x/y", use_gpu=use_gpu, gpu_device_id=gpu_device_id
            )
        load.assert_called_once_with("hf-hub:x/y", expected)
