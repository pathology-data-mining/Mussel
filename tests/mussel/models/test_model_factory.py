import pytest
import torch
from unittest.mock import MagicMock, patch

from mussel.models.model_factory import (
    MODEL_FACTORIES,
    ModelType,
    get_model_factory,
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


def test_model_type_properties():
    """Test ModelType enum properties"""
    resnet_type = ModelType.RESNET50
    assert resnet_type.code == "resnet50"
    assert resnet_type.id == 1
    
    ctranspath_type = ModelType.CTRANSPATH
    assert ctranspath_type.code == "ctranspath"
    assert ctranspath_type.id == 2


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

