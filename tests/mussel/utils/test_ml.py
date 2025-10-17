import numpy as np
import torch
import torch.nn as nn

from mussel.utils.ml import (
    SubsetSequentialSampler,
    calculate_error,
    collate_features,
    collate_MIL,
    initialize_weights,
    nth,
)


def test_subset_sequential_sampler():
    """Test SubsetSequentialSampler returns indices in order"""
    indices = [5, 2, 8, 1]
    sampler = SubsetSequentialSampler(indices)
    
    assert len(sampler) == 4
    assert list(sampler) == indices


def test_collate_mil():
    """Test collate_MIL function for batching"""
    # Create sample batch items
    batch = [
        (torch.tensor([[1, 2, 3]]), 0),
        (torch.tensor([[4, 5, 6]]), 1),
    ]
    
    img, labels = collate_MIL(batch)
    
    # Check concatenation
    assert img.shape == (2, 3)
    assert torch.equal(img, torch.tensor([[1, 2, 3], [4, 5, 6]]))
    
    # Check labels
    assert torch.equal(labels, torch.LongTensor([0, 1]))


def test_collate_features():
    """Test collate_features function"""
    # Create sample batch items with features and coordinates
    batch = [
        (torch.tensor([[1.0, 2.0]]), np.array([[0, 0]])),
        (torch.tensor([[3.0, 4.0]]), np.array([[10, 20]])),
    ]
    
    features, coords = collate_features(batch)
    
    # Check features concatenation
    assert features.shape == (2, 2)
    assert torch.equal(features, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    
    # Check coordinates stacking
    assert coords.shape == (2, 2)
    np.testing.assert_array_equal(coords, np.array([[0, 0], [10, 20]]))


def test_calculate_error():
    """Test calculate_error function"""
    # Perfect predictions
    y_hat = torch.tensor([0, 1, 2, 3])
    y = torch.tensor([0, 1, 2, 3])
    error = calculate_error(y_hat, y)
    assert error == 0.0
    
    # All wrong predictions
    y_hat = torch.tensor([1, 0, 3, 2])
    y = torch.tensor([0, 1, 2, 3])
    error = calculate_error(y_hat, y)
    assert error == 1.0
    
    # Half correct predictions
    y_hat = torch.tensor([0, 0, 2, 2])
    y = torch.tensor([0, 1, 2, 3])
    error = calculate_error(y_hat, y)
    assert error == 0.5


def test_nth():
    """Test nth function for getting nth element from iterator"""
    import collections
    
    iterator = iter([10, 20, 30, 40, 50])
    
    # Get 2nd element (0-indexed)
    result = nth(iterator, 2)
    assert result == 30
    
    # Test with None (consume iterator and return empty deque)
    iterator2 = iter([1, 2, 3])
    result = nth(iterator2, None)
    assert isinstance(result, collections.deque)
    assert len(result) == 0
    
    # Test with default value when index is out of range
    iterator3 = iter([1, 2])
    result = nth(iterator3, 10, default="not found")
    assert result == "not found"


def test_initialize_weights():
    """Test initialize_weights function"""
    # Create a simple module
    module = nn.Sequential(
        nn.Linear(10, 20),
        nn.BatchNorm1d(20),
        nn.Linear(20, 5)
    )
    
    # Initialize weights
    initialize_weights(module)
    
    # Check that Linear layers have non-zero weights and zero bias
    for m in module.modules():
        if isinstance(m, nn.Linear):
            assert not torch.all(m.weight == 0)
            assert torch.all(m.bias == 0)
        elif isinstance(m, nn.BatchNorm1d):
            assert torch.all(m.weight == 1)
            assert torch.all(m.bias == 0)
