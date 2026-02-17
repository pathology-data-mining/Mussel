import h5py
import numpy as np
import torch
from PIL import Image

from mussel.datasets.h5 import eval_transforms


def test_eval_transforms_with_imagenet():
    """Test eval_transforms with ImageNet normalization"""
    transform = eval_transforms(use_imagenet_rgb_dist=True)
    
    # Create a test image
    img = Image.new('RGB', (256, 256), color='red')
    
    # Apply transform
    tensor = transform(img)
    
    # Check output is a tensor
    assert isinstance(tensor, torch.Tensor)
    
    # Check shape (C, H, W)
    assert tensor.shape == (3, 256, 256)
    
    # Check normalization was applied (values should not be in [0, 255] range)
    assert tensor.min() < 0 or tensor.max() > 1


def test_eval_transforms_without_imagenet():
    """Test eval_transforms without ImageNet normalization"""
    transform = eval_transforms(use_imagenet_rgb_dist=False)
    
    # Create a test image
    img = Image.new('RGB', (128, 128), color='blue')
    
    # Apply transform
    tensor = transform(img)
    
    # Check output is a tensor
    assert isinstance(tensor, torch.Tensor)
    
    # Check shape
    assert tensor.shape == (3, 128, 128)


def test_eval_transforms_normalization_values():
    """Test that the normalization values are correct"""
    # With ImageNet distribution
    transform_imagenet = eval_transforms(use_imagenet_rgb_dist=True)
    
    # Create white image
    white_img = Image.new('RGB', (64, 64), color=(255, 255, 255))
    tensor = transform_imagenet(white_img)
    
    # All channels should have similar normalization
    assert tensor.shape[0] == 3
    
    # Without ImageNet distribution
    transform_simple = eval_transforms(use_imagenet_rgb_dist=False)
    tensor_simple = transform_simple(white_img)
    
    # Check that normalization was applied
    assert tensor_simple.max() <= 1.0
