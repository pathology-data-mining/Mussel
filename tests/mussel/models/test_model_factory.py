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
