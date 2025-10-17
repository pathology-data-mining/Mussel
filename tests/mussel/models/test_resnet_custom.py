import torch

from mussel.models.resnet_custom import (
    Bottleneck_Baseline,
    ResNet_Baseline,
    resnet50_baseline,
)


def test_bottleneck_baseline_forward():
    """Test Bottleneck_Baseline forward pass"""
    # Create bottleneck block with matching input/output channels
    # inplanes must equal planes * expansion for residual connection to work
    block = Bottleneck_Baseline(inplanes=256, planes=64)
    
    # Create random input
    x = torch.randn(1, 256, 56, 56)
    
    # Forward pass
    output = block(x)
    
    # Check output shape (expansion=4, so output channels = 64*4 = 256)
    assert output.shape == (1, 256, 56, 56)


def test_bottleneck_baseline_with_downsample():
    """Test Bottleneck_Baseline with downsampling"""
    # Create downsample layer
    downsample = torch.nn.Sequential(
        torch.nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
        torch.nn.BatchNorm2d(128)
    )
    
    block = Bottleneck_Baseline(inplanes=64, planes=32, stride=2, downsample=downsample)
    
    x = torch.randn(1, 64, 56, 56)
    output = block(x)
    
    # Check that spatial dimensions are halved and channels match
    assert output.shape == (1, 128, 28, 28)


def test_resnet_baseline_forward():
    """Test ResNet_Baseline forward pass"""
    model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])
    
    # Create random input (batch_size=2, channels=3, height=224, width=224)
    x = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    output = model(x)
    
    # Check output shape - should be flattened feature vector
    assert len(output.shape) == 2
    assert output.shape[0] == 2  # batch size


def test_resnet50_baseline_creation():
    """Test resnet50_baseline model creation"""
    model = resnet50_baseline(pretrained=False)
    
    # Check model is created
    assert isinstance(model, ResNet_Baseline)
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    
    # Should produce feature vector
    assert len(output.shape) == 2


def test_resnet_baseline_layers():
    """Test that ResNet_Baseline has expected layers"""
    model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])
    
    # Check that required layers exist
    assert hasattr(model, 'conv1')
    assert hasattr(model, 'bn1')
    assert hasattr(model, 'relu')
    assert hasattr(model, 'maxpool')
    assert hasattr(model, 'layer1')
    assert hasattr(model, 'layer2')
    assert hasattr(model, 'layer3')
    assert hasattr(model, 'avgpool')


def test_bottleneck_baseline_residual_connection():
    """Test that residual connection works correctly"""
    # Create bottleneck without stride change
    block = Bottleneck_Baseline(inplanes=256, planes=64)
    
    x = torch.randn(1, 256, 28, 28)
    
    # Forward pass
    output = block(x)
    
    # Output should have same spatial dimensions as input
    assert output.shape[2:] == x.shape[2:]
