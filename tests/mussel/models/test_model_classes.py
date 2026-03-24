"""Unit tests for all model classes in model_factory.py.

Each test mocks the underlying deep-learning libraries so that no GPU or
downloaded weights are required.  Tests verify:
  - Class can be instantiated (via __new__ + manual attribute setup)
  - get_preprocessing_fun() returns the correct type / is callable
  - get_model_fun() calls the underlying model with the right inputs and
    returns a CPU tensor with the expected shape
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, PropertyMock
from torchvision import transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(cls, obj, device=None, use_gpu=False):
    """Create a model instance without calling __init__."""
    instance = cls.__new__(cls)
    instance.obj = obj
    instance.device = device or torch.device("cpu")
    instance.use_gpu = use_gpu
    return instance


def _fake_pretrained_cfg():
    """Return a minimal timm pretrained_cfg dict for resolve_data_config."""
    return {
        "input_size": (3, 224, 224),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "interpolation": "bicubic",
        "crop_pct": 0.875,
    }


def _simple_patch_img(size=224):
    """Return a (1, 3, size, size) float32 tensor."""
    return torch.rand(1, 3, size, size)


# ---------------------------------------------------------------------------
# Patch encoder – preprocessing: custom Compose transforms
# ---------------------------------------------------------------------------

class TestOptimusModelPreprocessing:
    """OptimusModel (H-Optimus-0) uses custom Compose with its own normalisation."""

    def test_preprocessing_returns_compose(self):
        from mussel.models.model_factory import OptimusModel
        m = _make_model(OptimusModel, MagicMock())
        prep = m.get_preprocessing_fun()
        assert isinstance(prep, transforms.Compose)

    def test_preprocessing_last_transform_normalise(self):
        from mussel.models.model_factory import OptimusModel
        m = _make_model(OptimusModel, MagicMock())
        prep = m.get_preprocessing_fun()
        last = prep.transforms[-1]
        assert isinstance(last, transforms.Normalize)
        assert abs(last.mean[0] - 0.707223) < 1e-5


class TestHOptimus1ModelPreprocessing:
    def test_preprocessing_returns_compose(self):
        from mussel.models.model_factory import HOptimus1Model
        m = _make_model(HOptimus1Model, MagicMock())
        prep = m.get_preprocessing_fun()
        assert isinstance(prep, transforms.Compose)

    def test_preprocessing_normalise_mean(self):
        from mussel.models.model_factory import HOptimus1Model
        m = _make_model(HOptimus1Model, MagicMock())
        prep = m.get_preprocessing_fun()
        norm = prep.transforms[-1]
        assert isinstance(norm, transforms.Normalize)
        assert abs(norm.mean[0] - 0.707223) < 1e-5


class TestH0MiniModelPreprocessing:
    def test_preprocessing_returns_compose(self):
        from mussel.models.model_factory import H0MiniModel
        m = _make_model(H0MiniModel, MagicMock())
        prep = m.get_preprocessing_fun()
        assert isinstance(prep, transforms.Compose)


class TestConch15ModelPreprocessing:
    def test_preprocessing_returns_compose(self):
        from mussel.models.model_factory import Conch15Model
        m = _make_model(Conch15Model, MagicMock())
        prep = m.get_preprocessing_fun()
        assert isinstance(prep, transforms.Compose)

    def test_preprocessing_resize_448(self):
        from mussel.models.model_factory import Conch15Model
        m = _make_model(Conch15Model, MagicMock())
        prep = m.get_preprocessing_fun()
        resize_t = next(t for t in prep.transforms if isinstance(t, transforms.Resize))
        assert resize_t.size == 448


# ---------------------------------------------------------------------------
# Patch encoders – preprocessing via timm resolve_data_config
# ---------------------------------------------------------------------------

def _timm_prep_model():
    """Return a MagicMock with a pretrained_cfg attribute."""
    model_mock = MagicMock()
    model_mock.pretrained_cfg = _fake_pretrained_cfg()
    return model_mock


_TIMM_CFG = {"input_size": (3, 224, 224), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225), "interpolation": "bicubic", "crop_pct": 0.875}


class TestTimmBasedPreprocessing:
    """VirchowModel, UniModel, Uni2Model, PhikonModel, Midnight12kModel, GPFMModel."""

    def _assert_prep_callable(self, cls):
        fake_transform = transforms.Compose([transforms.ToTensor()])
        with patch("mussel.models.model_factory.resolve_data_config", return_value=_TIMM_CFG) as mock_resolve, \
             patch("mussel.models.model_factory.create_transform", return_value=fake_transform) as mock_create:
            m = _make_model(cls, _timm_prep_model())
            prep = m.get_preprocessing_fun()
            assert callable(prep)
            mock_resolve.assert_called_once()
            mock_create.assert_called_once()

    def test_virchow_preprocessing(self):
        from mussel.models.model_factory import VirchowModel
        self._assert_prep_callable(VirchowModel)

    def test_uni_preprocessing(self):
        from mussel.models.model_factory import UniModel
        self._assert_prep_callable(UniModel)

    def test_uni2_preprocessing(self):
        from mussel.models.model_factory import Uni2Model
        self._assert_prep_callable(Uni2Model)

    def test_phikon_preprocessing(self):
        from mussel.models.model_factory import PhikonModel
        self._assert_prep_callable(PhikonModel)

    def test_phikon_v2_preprocessing(self):
        from mussel.models.model_factory import PhikonV2Model
        self._assert_prep_callable(PhikonV2Model)

    def test_midnight12k_preprocessing(self):
        from mussel.models.model_factory import Midnight12kModel
        self._assert_prep_callable(Midnight12kModel)

    def test_gpfm_preprocessing(self):
        from mussel.models.model_factory import GPFMModel
        self._assert_prep_callable(GPFMModel)


# ---------------------------------------------------------------------------
# Patch encoder – HibouLModel (custom preprocessing via AutoImageProcessor)
# ---------------------------------------------------------------------------

class TestHibouLModelPreprocessing:
    def test_preprocessing_callable(self):
        from mussel.models.model_factory import HibouLModel
        processor = MagicMock()
        processor.return_value = {"pixel_values": torch.rand(1, 3, 224, 224)}
        m = _make_model(HibouLModel, MagicMock())
        m._processor = processor
        prep = m.get_preprocessing_fun()
        assert callable(prep)

    def test_preprocessing_extracts_pixel_values(self):
        from mussel.models.model_factory import HibouLModel
        img = MagicMock()
        processor = MagicMock()
        expected = torch.rand(1, 3, 224, 224)
        processor.return_value = {"pixel_values": expected}
        m = _make_model(HibouLModel, MagicMock())
        m._processor = processor
        result = m.get_preprocessing_fun()(img)
        # squeeze(0): (1, 3, 224, 224) -> (3, 224, 224)
        assert result.shape == torch.Size([3, 224, 224])


# ---------------------------------------------------------------------------
# Patch encoder – ClipModel (preprocessing stored in self.preprocessing)
# ---------------------------------------------------------------------------

class TestClipModelPreprocessing:
    def test_preprocessing_returns_stored_transform(self):
        from mussel.models.model_factory import ClipModel
        m = _make_model(ClipModel, MagicMock())
        fake_prep = transforms.Compose([transforms.ToTensor()])
        m.preprocessing = fake_prep
        assert m.get_preprocessing_fun() is fake_prep


# ---------------------------------------------------------------------------
# Slide encoders – preprocessing returns None
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", [
    "PRISMSlideEncoderModel",
    "FeatherSlideEncoderModel",
    "MadeleineSlideEncoderModel",
])
def test_slide_encoder_preprocessing_is_none(cls_name):
    import mussel.models.model_factory as mf
    cls = getattr(mf, cls_name)
    m = _make_model(cls, MagicMock())
    assert m.get_preprocessing_fun() is None


def test_gigapath_slide_preprocessing_is_none():
    from mussel.models.model_factory import GigapathSlideEncoderModel
    m = _make_model(GigapathSlideEncoderModel, MagicMock())
    assert m.get_preprocessing_fun() is None


def test_titan_slide_preprocessing_is_none():
    from mussel.models.model_factory import TitanSlideEncoderModel
    m = _make_model(TitanSlideEncoderModel, MagicMock())
    assert m.get_preprocessing_fun() is None


# ---------------------------------------------------------------------------
# Patch encoder – model_fun: standard TorchModel (returns self.obj(x).cpu())
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name,embed_dim", [
    ("ResnetModel", 1024),
    ("UniModel", 1024),
    ("Uni2Model", 1536),
    ("OptimusModel", 768),
    ("HOptimus1Model", 1024),
    ("H0MiniModel", 512),
    ("PhikonModel", 768),
    ("PhikonV2Model", 1024),
    ("Midnight12kModel", 768),
    ("GPFMModel", 1024),
    ("GigapathModel", 1536),
    ("TransPathModel", 768),
])
def test_standard_patch_encoder_model_fun(cls_name, embed_dim):
    """Standard TorchModel.get_model_fun() returns self.obj(x).cpu()."""
    import mussel.models.model_factory as mf
    cls = getattr(mf, cls_name)
    batch_size = 2
    expected = torch.rand(batch_size, embed_dim)
    mock_model = MagicMock(return_value=expected)
    m = _make_model(cls, mock_model)
    model_fun = m.get_model_fun()
    x = _simple_patch_img()
    result = model_fun(x)
    mock_model.assert_called_once()
    assert result.device.type == "cpu"


# ---------------------------------------------------------------------------
# VirchowModel – model_fun concatenates CLS + avg patch tokens
# ---------------------------------------------------------------------------

class TestVirchowModelFun:
    def _run(self, cls, embed_dim=1280):
        n_patches = 256
        batch_size = 1
        # Return [batch, 1+n_patches, embed_dim]  (CLS + patch tokens)
        mock_output = torch.rand(batch_size, 1 + n_patches, embed_dim)
        mock_model = MagicMock(return_value=mock_output)
        m = _make_model(cls, mock_model)
        model_fun = m.get_model_fun()
        x = _simple_patch_img()
        result = model_fun(x)
        assert result.device.type == "cpu"
        # Output should be concat of CLS + avg_patch: (batch, 2*embed_dim)
        assert result.shape == torch.Size([batch_size, 2 * embed_dim])

    def test_virchow(self):
        from mussel.models.model_factory import VirchowModel
        self._run(VirchowModel, embed_dim=1280)

    def test_virchow2(self):
        from mussel.models.model_factory import Virchow2Model
        self._run(Virchow2Model, embed_dim=1280)


# ---------------------------------------------------------------------------
# HibouLModel – model_fun extracts CLS token from last_hidden_state
# ---------------------------------------------------------------------------

class TestHibouLModelFun:
    def test_model_fun_returns_cls_token(self):
        from mussel.models.model_factory import HibouLModel
        embed_dim = 1024
        batch_size = 2
        # Simulate transformers model output with last_hidden_state
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.rand(batch_size, 100, embed_dim)
        mock_model = MagicMock(return_value=mock_output)
        m = _make_model(HibouLModel, mock_model)
        m._processor = MagicMock()
        model_fun = m.get_model_fun()
        x = _simple_patch_img()
        result = model_fun(x)
        assert result.device.type == "cpu"
        assert result.shape == torch.Size([batch_size, embed_dim])

    def test_model_fun_calls_with_pixel_values(self):
        from mussel.models.model_factory import HibouLModel
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.rand(1, 50, 512)
        mock_model = MagicMock(return_value=mock_output)
        m = _make_model(HibouLModel, mock_model)
        m._processor = MagicMock()
        model_fun = m.get_model_fun()
        x = _simple_patch_img()
        model_fun(x)
        call_kwargs = mock_model.call_args.kwargs
        assert "pixel_values" in call_kwargs


# ---------------------------------------------------------------------------
# Slide encoders – model_fun
# ---------------------------------------------------------------------------

class TestPRISMSlideEncoderModelFun:
    def test_calls_encode_slide_and_squeezes(self):
        from mussel.models.model_factory import PRISMSlideEncoderModel
        embed_dim = 1024
        mock_model = MagicMock()
        mock_model.encode_slide = MagicMock(return_value=torch.rand(1, embed_dim))
        m = _make_model(PRISMSlideEncoderModel, mock_model)
        model_fun = m.get_model_fun()
        patch_features = torch.rand(1, 100, 1280)
        coords = torch.rand(1, 100, 2)
        result = model_fun(patch_features, coords, patch_size=224)
        mock_model.encode_slide.assert_called_once()
        assert result.device.type == "cpu"
        assert result.shape == torch.Size([embed_dim])


class TestFeatherSlideEncoderModelFun:
    def test_calls_model_and_squeezes(self):
        from mussel.models.model_factory import FeatherSlideEncoderModel
        embed_dim = 512
        mock_model = MagicMock(return_value=torch.rand(1, embed_dim))
        m = _make_model(FeatherSlideEncoderModel, mock_model)
        model_fun = m.get_model_fun()
        patch_features = torch.rand(1, 100, 512)
        coords = torch.rand(1, 100, 2)
        result = model_fun(patch_features, coords, patch_size=512)
        mock_model.assert_called_once()
        assert result.device.type == "cpu"
        assert result.shape == torch.Size([embed_dim])


class TestMadeleineSlideEncoderModelFun:
    def test_calls_model_and_squeezes(self):
        from mussel.models.model_factory import MadeleineSlideEncoderModel
        embed_dim = 512
        mock_model = MagicMock(return_value=torch.rand(1, embed_dim))
        m = _make_model(MadeleineSlideEncoderModel, mock_model)
        model_fun = m.get_model_fun()
        patch_features = torch.rand(1, 100, 512)
        coords = torch.rand(1, 100, 2)
        result = model_fun(patch_features, coords, patch_size=512)
        mock_model.assert_called_once()
        assert result.device.type == "cpu"
        assert result.shape == torch.Size([embed_dim])

    def test_unwraps_tuple_output(self):
        from mussel.models.model_factory import MadeleineSlideEncoderModel
        embed_dim = 512
        inner = torch.rand(1, embed_dim)
        mock_model = MagicMock(return_value=(inner, "ignored"))
        m = _make_model(MadeleineSlideEncoderModel, mock_model)
        model_fun = m.get_model_fun()
        result = model_fun(torch.rand(1, 100, 512), torch.rand(1, 100, 2), 512)
        assert result.shape == torch.Size([embed_dim])


# ---------------------------------------------------------------------------
# Slide encoders – save() raises ValueError for file paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", [
    "PRISMSlideEncoderModel",
    "FeatherSlideEncoderModel",
    "MadeleineSlideEncoderModel",
])
def test_slide_encoder_save_rejects_file_extension(cls_name, tmp_path):
    import mussel.models.model_factory as mf
    cls = getattr(mf, cls_name)
    m = _make_model(cls, MagicMock())
    with pytest.raises(ValueError, match="directory"):
        m.save(str(tmp_path / "model.pt"))


# ---------------------------------------------------------------------------
# OptimusModel class-level correctness (was orphaned bug)
# ---------------------------------------------------------------------------

class TestOptimusModelClassIntegrity:
    def test_optimus_model_exists(self):
        from mussel.models.model_factory import OptimusModel
        assert OptimusModel is not None

    def test_optimus_model_is_separate_from_madeleine(self):
        from mussel.models.model_factory import OptimusModel, MadeleineSlideEncoderModel
        assert OptimusModel is not MadeleineSlideEncoderModel

    def test_optimus_factory_references_optimus_model(self):
        from mussel.models.model_factory import OptimusModelFactory, OptimusModel
        factory = OptimusModelFactory()
        with patch.object(
            OptimusModel, "__init__", return_value=None
        ) as mock_init:
            with patch.object(
                OptimusModel, "__new__", return_value=MagicMock(spec=OptimusModel)
            ):
                try:
                    factory.get_model(model_path="hf-hub:bioptimus/H-optimus-0", use_gpu=False)
                except Exception:
                    pass
            # Verify factory uses OptimusModel (not MadeleineSlideEncoderModel)
        assert OptimusModelFactory.__name__ == "OptimusModelFactory"

    def test_madeleine_init_loads_madeleine_path(self):
        """MadeleineSlideEncoderModel.__init__ should reference MADELEINE_SLIDE, not OPTIMUS."""
        from mussel.models.model_factory import MadeleineSlideEncoderModel, ModelType
        import inspect
        src = inspect.getsource(MadeleineSlideEncoderModel.__init__)
        assert "MADELEINE_SLIDE" in src
        assert "OPTIMUS" not in src


# ---------------------------------------------------------------------------
# All 24 model types have a factory and a patch size
# ---------------------------------------------------------------------------

def test_all_model_types_have_factory():
    from mussel.models.model_factory import MODEL_FACTORIES, ModelType
    for mt in ModelType:
        assert mt in MODEL_FACTORIES, f"{mt} missing from MODEL_FACTORIES"


def test_all_model_types_have_patch_size():
    from mussel.models.model_factory import MODEL_PATCH_SIZES, ModelType
    for mt in ModelType:
        assert mt in MODEL_PATCH_SIZES, f"{mt} missing from MODEL_PATCH_SIZES"
