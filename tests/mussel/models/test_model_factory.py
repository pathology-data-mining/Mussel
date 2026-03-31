import pytest
import torch
from unittest.mock import MagicMock, patch

from mussel.models.model_factory import (
    ModelType,
    get_model_factory,
    get_default_patch_size,
    MODEL_PATCH_SIZES,
)
# Ensure all model modules are registered
import mussel.models  # noqa: F401


def test_get_model_factory_returns_factory():
    """get_model_factory returns a non-None factory for RESNET50 and the default."""
    assert get_model_factory(ModelType.RESNET50) is not None
    assert get_model_factory(ModelType.CTRANSPATH) is not None
    assert get_model_factory() is not None  # default


def test_get_default_patch_size():
    """get_default_patch_size returns correct values for each model."""
    assert get_default_patch_size(ModelType.RESNET50) == 256
    assert get_default_patch_size(ModelType.GIGAPATH) == 256
    assert get_default_patch_size(ModelType.GIGAPATH_SLIDE) == 256
    assert get_default_patch_size(ModelType.UNI) == 256
    assert get_default_patch_size(ModelType.UNI2) == 256

    assert get_default_patch_size(ModelType.CTRANSPATH) == 224
    assert get_default_patch_size(ModelType.VIRCHOW) == 224
    assert get_default_patch_size(ModelType.VIRCHOW2) == 224
    assert get_default_patch_size(ModelType.OPTIMUS) == 224
    assert get_default_patch_size(ModelType.CLIP) == 224
    assert get_default_patch_size(ModelType.GOOGLEPATH) == 224

    assert get_default_patch_size(ModelType.CONCH1_5) == 512
    assert get_default_patch_size(ModelType.TITAN_SLIDE) == 512

    # New patch encoders (all 224 px)
    for mt in (
        ModelType.PHIKON, ModelType.PHIKON_V2, ModelType.H_OPTIMUS_1,
        ModelType.H0_MINI, ModelType.MIDNIGHT12K, ModelType.GPFM, ModelType.HIBOU_L,
    ):
        assert get_default_patch_size(mt) == 224, f"{mt}: expected 224"


def test_get_default_patch_size_invalid():
    """get_default_patch_size raises ValueError for an unknown type."""
    class _Fake:
        pass
    with pytest.raises(ValueError, match="Unknown model type"):
        get_default_patch_size(_Fake())


def test_titan_slide_encoder_model_fun():
    """TitanSlideEncoderModel.get_model_fun() squeezes batch dim from (1, 768) → (768,)."""
    from mussel.models.conch import TitanSlideEncoderModel

    mock_model = MagicMock()
    mock_model.encode_slide_from_patch_features = MagicMock(return_value=torch.randn(1, 768))

    with patch.object(TitanSlideEncoderModel, '__init__', lambda self, *a, **kw: None):
        encoder = TitanSlideEncoderModel.__new__(TitanSlideEncoderModel)
        encoder.obj = mock_model
        encoder.device = torch.device('cpu')
        encoder.use_gpu = False

        model_fun = encoder.get_model_fun()
        result = model_fun(torch.randn(1, 100, 768), torch.randint(0, 4096, (1, 100, 2)), 512)

    assert mock_model.encode_slide_from_patch_features.called
    assert result.shape == torch.Size([768]), f"Expected (768,), got {result.shape}"


def test_gigapath_slide_encoder_model_fun():
    """GigapathSlideEncoderModel.get_model_fun() squeezes batch dim from (1, 768) → (768,)."""
    from mussel.models.gigapath import GigapathSlideEncoderModel

    mock_model = MagicMock(return_value=[torch.randn(1, 768)])

    with patch.object(GigapathSlideEncoderModel, '__init__', lambda self, *a, **kw: None):
        encoder = GigapathSlideEncoderModel.__new__(GigapathSlideEncoderModel)
        encoder.obj = mock_model
        encoder.device = torch.device('cpu')
        encoder.use_gpu = False

        model_fun = encoder.get_model_fun()
        result = model_fun(torch.randn(1, 100, 1536), torch.randn(1, 100, 2))

    assert mock_model.called
    assert result.shape == torch.Size([768]), f"Expected (768,), got {result.shape}"


def test_new_slide_encoders_compatibility():
    """New slide encoders map to the correct required patch encoder."""
    from mussel.models.model_factory import SLIDE_ENCODER_COMPATIBILITY, get_required_patch_encoder

    assert SLIDE_ENCODER_COMPATIBILITY[ModelType.PRISM_SLIDE]     == ModelType.VIRCHOW
    assert SLIDE_ENCODER_COMPATIBILITY[ModelType.FEATHER_SLIDE]   == ModelType.CONCH1_5
    assert SLIDE_ENCODER_COMPATIBILITY[ModelType.CHIEF_SLIDE]     == ModelType.CTRANSPATH
    assert SLIDE_ENCODER_COMPATIBILITY[ModelType.MADELEINE_SLIDE] == ModelType.CLIP

    assert get_required_patch_encoder(ModelType.PRISM_SLIDE)     == ModelType.VIRCHOW
    assert get_required_patch_encoder(ModelType.MADELEINE_SLIDE) == ModelType.CLIP


def test_chief_slide_raises_on_invalid_checkpoint():
    """CHIEF_SLIDE raises an error when given an invalid checkpoint file."""
    from mussel.models.chief import CHIEFSlideEncoderModel
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        fake_path = os.path.join(d, "chief.pt")
        open(fake_path, "w").close()
        with pytest.raises(Exception):
            CHIEFSlideEncoderModel(model_path=fake_path)

