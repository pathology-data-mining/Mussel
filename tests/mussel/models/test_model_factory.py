import pytest
import torch
from unittest.mock import MagicMock, patch

from mussel.models.model_factory import (
    MODEL_FACTORIES,
    ModelType,
    get_model_factory,
    get_default_patch_size,
    MODEL_PATCH_SIZES,
)


def test_model_type_enum():
    """Test ModelType enum has expected values"""
    assert hasattr(ModelType, 'RESNET50')
    assert hasattr(ModelType, 'CTRANSPATH')
    assert hasattr(ModelType, 'GIGAPATH')
    assert hasattr(ModelType, 'VIRCHOW')
    assert hasattr(ModelType, 'OPTIMUS')
    assert hasattr(ModelType, 'CLIP')
    assert hasattr(ModelType, 'GOOGLEPATH')
    assert hasattr(ModelType, 'CONCH1_5')
    assert hasattr(ModelType, 'VIRCHOW2')
    assert hasattr(ModelType, 'UNI')
    assert hasattr(ModelType, 'UNI2')


def test_model_type_properties():
    """Test ModelType enum properties"""
    resnet_type = ModelType.RESNET50
    assert resnet_type.code == "resnet50"
    assert resnet_type.id == 1
    
    ctranspath_type = ModelType.CTRANSPATH
    assert ctranspath_type.code == "ctranspath"
    assert ctranspath_type.id == 2
    
    uni_type = ModelType.UNI
    assert uni_type.code == "uni"
    assert uni_type.id == 11
    
    uni2_type = ModelType.UNI2
    assert uni2_type.code == "uni2h"
    assert uni2_type.id == 10


def test_model_factories_registered():
    """Test that model factories are registered"""
    # Check that factories are registered for main model types
    assert ModelType.RESNET50 in MODEL_FACTORIES
    assert ModelType.CTRANSPATH in MODEL_FACTORIES
    assert ModelType.GIGAPATH in MODEL_FACTORIES
    assert ModelType.VIRCHOW in MODEL_FACTORIES
    assert ModelType.OPTIMUS in MODEL_FACTORIES
    assert ModelType.CLIP in MODEL_FACTORIES
    assert ModelType.GOOGLEPATH in MODEL_FACTORIES
    assert ModelType.CONCH1_5 in MODEL_FACTORIES
    assert ModelType.UNI in MODEL_FACTORIES
    assert ModelType.UNI2 in MODEL_FACTORIES


def test_get_model_factory():
    """Test get_model_factory function"""
    # Get factory for ResNet50
    factory = get_model_factory(ModelType.RESNET50)
    assert factory is not None
    
    # Get factory for CTransPath
    factory = get_model_factory(ModelType.CTRANSPATH)
    assert factory is not None


def test_get_model_factory_default():
    """Test get_model_factory with default parameter"""
    # Default should be CTRANSPATH
    factory = get_model_factory()
    assert factory is not None


def test_titan_slide_encoder_model_fun():
    """Test that TitanSlideEncoderModel.get_model_fun() properly moves inputs to device."""
    from mussel.models.model_factory import TitanSlideEncoderModel
    
    # Create a mock model object
    mock_model = MagicMock()
    mock_model.encode_slide_from_patch_features = MagicMock(return_value=torch.randn(1, 768))
    
    # Create TitanSlideEncoderModel instance with mock
    with patch('mussel.models.model_factory.AutoModel.from_pretrained', return_value=mock_model):
        # We need to mock the parent __init__ to avoid device setup
        with patch.object(TitanSlideEncoderModel, '__init__', lambda self, *args, **kwargs: None):
            encoder = TitanSlideEncoderModel.__new__(TitanSlideEncoderModel)
            encoder.obj = mock_model
            encoder.device = torch.device('cpu')
            encoder.use_gpu = False
            
            # Get the model function
            model_fun = encoder.get_model_fun()
            
            # Create test inputs
            patch_features = torch.randn(1, 100, 768)
            coords = torch.randint(0, 4096, (1, 100, 2))
            patch_size = 512
            
            # Call the model function
            result = model_fun(patch_features, coords, patch_size)
            
            # Verify that encode_slide_from_patch_features was called
            assert mock_model.encode_slide_from_patch_features.called
            
            # Get the call arguments
            call_args = mock_model.encode_slide_from_patch_features.call_args
            
            # Verify that the inputs were passed (they should be on the device)
            assert call_args[0][0] is not None  # patch_features
            assert call_args[0][1] is not None  # coords
            assert call_args[0][2] == patch_size  # patch_size
            
            # Verify output shape is (768,) not (1, 768)
            assert result.shape == torch.Size([768]), f"Expected shape (768,), got {result.shape}"


def test_gigapath_slide_encoder_model_fun():
    """Test that GigapathSlideEncoderModel.get_model_fun() properly moves inputs to device."""
    from mussel.models.model_factory import GigapathSlideEncoderModel
    
    # Create a mock model object
    mock_model = MagicMock()
    mock_model.return_value = [torch.randn(1, 768)]
    
    # Create GigapathSlideEncoderModel instance with mock
    with patch('timm.create_model', return_value=mock_model):
        # We need to mock the parent __init__ to avoid device setup
        with patch.object(GigapathSlideEncoderModel, '__init__', lambda self, *args, **kwargs: None):
            encoder = GigapathSlideEncoderModel.__new__(GigapathSlideEncoderModel)
            encoder.obj = mock_model
            encoder.device = torch.device('cpu')
            encoder.use_gpu = False
            
            # Get the model function
            model_fun = encoder.get_model_fun()
            
            # Create test inputs
            features = torch.randn(1, 100, 1536)
            coords = torch.randn(1, 100, 2)
            
            # Call the model function
            result = model_fun(features, coords)
            
            # Verify that the model was called
            assert mock_model.called
            
            # Get the call arguments
            call_args = mock_model.call_args
            
            # Verify that the inputs were passed (they should be on the device)
            assert call_args[0][0] is not None  # features
            assert call_args[0][1] is not None  # coords
            
            # Verify output shape is (768,) not (1, 768)
            assert result.shape == torch.Size([768]), f"Expected shape (768,), got {result.shape}"


def test_model_patch_sizes_mapping():
    """Test that MODEL_PATCH_SIZES contains all model types."""
    # Check that all patch encoder models have a patch size mapping
    assert ModelType.RESNET50 in MODEL_PATCH_SIZES
    assert ModelType.CTRANSPATH in MODEL_PATCH_SIZES
    assert ModelType.GIGAPATH in MODEL_PATCH_SIZES
    assert ModelType.VIRCHOW in MODEL_PATCH_SIZES
    assert ModelType.VIRCHOW2 in MODEL_PATCH_SIZES
    assert ModelType.OPTIMUS in MODEL_PATCH_SIZES
    assert ModelType.CLIP in MODEL_PATCH_SIZES
    assert ModelType.GOOGLEPATH in MODEL_PATCH_SIZES
    assert ModelType.CONCH1_5 in MODEL_PATCH_SIZES
    assert ModelType.UNI in MODEL_PATCH_SIZES
    assert ModelType.UNI2 in MODEL_PATCH_SIZES
    assert ModelType.GIGAPATH_SLIDE in MODEL_PATCH_SIZES
    assert ModelType.TITAN_SLIDE in MODEL_PATCH_SIZES


def test_get_default_patch_size():
    """Test get_default_patch_size function returns correct values."""
    # Test 256 pixel models
    assert get_default_patch_size(ModelType.RESNET50) == 256
    assert get_default_patch_size(ModelType.GIGAPATH) == 256
    assert get_default_patch_size(ModelType.GIGAPATH_SLIDE) == 256
    assert get_default_patch_size(ModelType.UNI) == 256
    assert get_default_patch_size(ModelType.UNI2) == 256
    
    # Test 224 pixel models
    assert get_default_patch_size(ModelType.CTRANSPATH) == 224
    assert get_default_patch_size(ModelType.VIRCHOW) == 224
    assert get_default_patch_size(ModelType.VIRCHOW2) == 224
    assert get_default_patch_size(ModelType.OPTIMUS) == 224
    assert get_default_patch_size(ModelType.CLIP) == 224
    assert get_default_patch_size(ModelType.GOOGLEPATH) == 224
    
    # Test 512 pixel models
    assert get_default_patch_size(ModelType.CONCH1_5) == 512
    assert get_default_patch_size(ModelType.TITAN_SLIDE) == 512


def test_get_default_patch_size_invalid():
    """Test get_default_patch_size raises ValueError for invalid model type."""
    # Create a mock ModelType that doesn't exist in the mapping
    class FakeModelType:
        pass
    
    fake_model = FakeModelType()
    
    with pytest.raises(ValueError, match="Unknown model type"):
        get_default_patch_size(fake_model)


# ---------------------------------------------------------------------------
# Tests for new patch encoder models (added in trident-features branch)
# ---------------------------------------------------------------------------


def test_new_patch_encoder_enum_values():
    """New patch encoders are present in the ModelType enum."""
    assert hasattr(ModelType, "PHIKON")
    assert hasattr(ModelType, "PHIKON_V2")
    assert hasattr(ModelType, "H_OPTIMUS_1")
    assert hasattr(ModelType, "H0_MINI")
    assert hasattr(ModelType, "MIDNIGHT12K")
    assert hasattr(ModelType, "GPFM")
    assert hasattr(ModelType, "HIBOU_L")


def test_new_patch_encoder_properties():
    """New patch encoders have correct codes and IDs."""
    assert ModelType.PHIKON.code == "phikon"
    assert ModelType.PHIKON.id == 14
    assert ModelType.PHIKON_V2.code == "phikon_v2"
    assert ModelType.PHIKON_V2.id == 15
    assert ModelType.H_OPTIMUS_1.code == "hoptimus1"
    assert ModelType.H_OPTIMUS_1.id == 16
    assert ModelType.H0_MINI.code == "h0mini"
    assert ModelType.H0_MINI.id == 17
    assert ModelType.MIDNIGHT12K.code == "midnight12k"
    assert ModelType.MIDNIGHT12K.id == 18
    assert ModelType.GPFM.code == "gpfm"
    assert ModelType.GPFM.id == 19
    assert ModelType.HIBOU_L.code == "hibou_l"
    assert ModelType.HIBOU_L.id == 20


def test_new_patch_encoders_in_model_factories():
    """New patch encoders are registered in MODEL_FACTORIES."""
    assert ModelType.PHIKON in MODEL_FACTORIES
    assert ModelType.PHIKON_V2 in MODEL_FACTORIES
    assert ModelType.H_OPTIMUS_1 in MODEL_FACTORIES
    assert ModelType.H0_MINI in MODEL_FACTORIES
    assert ModelType.MIDNIGHT12K in MODEL_FACTORIES
    assert ModelType.GPFM in MODEL_FACTORIES
    assert ModelType.HIBOU_L in MODEL_FACTORIES


def test_new_patch_encoders_in_model_patch_sizes():
    """New patch encoders have patch size entries (all 224 px)."""
    for model_type in (
        ModelType.PHIKON,
        ModelType.PHIKON_V2,
        ModelType.H_OPTIMUS_1,
        ModelType.H0_MINI,
        ModelType.MIDNIGHT12K,
        ModelType.GPFM,
        ModelType.HIBOU_L,
    ):
        assert model_type in MODEL_PATCH_SIZES, f"{model_type} missing from MODEL_PATCH_SIZES"
        assert MODEL_PATCH_SIZES[model_type] == 224, f"{model_type} patch size should be 224"


def test_new_patch_encoders_get_default_patch_size():
    """get_default_patch_size returns 224 for all new patch encoders."""
    for model_type in (
        ModelType.PHIKON,
        ModelType.PHIKON_V2,
        ModelType.H_OPTIMUS_1,
        ModelType.H0_MINI,
        ModelType.MIDNIGHT12K,
        ModelType.GPFM,
        ModelType.HIBOU_L,
    ):
        assert get_default_patch_size(model_type) == 224, f"{model_type}: expected 224"


def test_new_patch_encoders_get_model_factory():
    """get_model_factory returns a non-None factory for each new patch encoder."""
    for model_type in (
        ModelType.PHIKON,
        ModelType.PHIKON_V2,
        ModelType.H_OPTIMUS_1,
        ModelType.H0_MINI,
        ModelType.MIDNIGHT12K,
        ModelType.GPFM,
        ModelType.HIBOU_L,
    ):
        factory = get_model_factory(model_type)
        assert factory is not None, f"No factory registered for {model_type}"


# ---------------------------------------------------------------------------
# Tests for new slide encoder models
# ---------------------------------------------------------------------------


def test_new_slide_encoder_enum_values():
    """New slide encoders are present in the ModelType enum."""
    assert hasattr(ModelType, "PRISM_SLIDE")
    assert hasattr(ModelType, "FEATHER_SLIDE")
    assert hasattr(ModelType, "CHIEF_SLIDE")
    assert hasattr(ModelType, "MADELEINE_SLIDE")


def test_new_slide_encoder_properties():
    """New slide encoders have correct codes and IDs."""
    assert ModelType.PRISM_SLIDE.id == 21
    assert ModelType.FEATHER_SLIDE.id == 22
    assert ModelType.CHIEF_SLIDE.id == 23
    assert ModelType.MADELEINE_SLIDE.id == 24


def test_new_slide_encoders_in_model_patch_sizes():
    """New slide encoders have patch size entries."""
    assert ModelType.PRISM_SLIDE in MODEL_PATCH_SIZES
    assert ModelType.FEATHER_SLIDE in MODEL_PATCH_SIZES
    assert ModelType.CHIEF_SLIDE in MODEL_PATCH_SIZES
    assert ModelType.MADELEINE_SLIDE in MODEL_PATCH_SIZES

    assert MODEL_PATCH_SIZES[ModelType.PRISM_SLIDE] == 224
    assert MODEL_PATCH_SIZES[ModelType.FEATHER_SLIDE] == 512
    assert MODEL_PATCH_SIZES[ModelType.CHIEF_SLIDE] == 224
    assert MODEL_PATCH_SIZES[ModelType.MADELEINE_SLIDE] == 512


def test_new_slide_encoders_compatibility():
    """New slide encoders map to the correct required patch encoder."""
    from mussel.models.model_factory import SLIDE_ENCODER_COMPATIBILITY, get_required_patch_encoder

    assert SLIDE_ENCODER_COMPATIBILITY[ModelType.PRISM_SLIDE] == ModelType.VIRCHOW
    assert SLIDE_ENCODER_COMPATIBILITY[ModelType.FEATHER_SLIDE] == ModelType.CONCH1_5
    assert SLIDE_ENCODER_COMPATIBILITY[ModelType.CHIEF_SLIDE] == ModelType.CTRANSPATH
    assert SLIDE_ENCODER_COMPATIBILITY[ModelType.MADELEINE_SLIDE] == ModelType.CONCH1_5

    # Via the public API as well
    assert get_required_patch_encoder(ModelType.PRISM_SLIDE) == ModelType.VIRCHOW
    assert get_required_patch_encoder(ModelType.MADELEINE_SLIDE) == ModelType.CONCH1_5


def test_chief_slide_raises_not_implemented():
    """CHIEF_SLIDE raises NotImplementedError — local checkpoint not on HuggingFace."""
    from mussel.models.model_factory import CHIEFSlideEncoderModel

    # __init__ raises ValueError for missing path before NotImplementedError;
    # pass a fake existing path to get to the NotImplementedError branch.
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        fake_path = os.path.join(d, "chief.pt")
        open(fake_path, "w").close()
        with pytest.raises(NotImplementedError):
            CHIEFSlideEncoderModel(model_path=fake_path)
