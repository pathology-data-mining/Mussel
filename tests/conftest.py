"""
Shared pytest configuration and fixtures for Mussel tests.

This conftest.py provides common test fixtures and configuration
used across multiple test files.
"""

import os
from pathlib import Path
from typing import Generator

import pytest
import torch
from omegaconf import OmegaConf

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import mussel modules
import mussel.cli.extract_features
from mussel.cli.extract_features import ExtractFeaturesConfig
from mussel.models import ModelType


@pytest.fixture
def test_data_path() -> str:
    """Return path to test data directory."""
    return os.path.join(os.path.dirname(__file__), "testdata")


@pytest.fixture
def svs_slide_path(test_data_path: str) -> str:
    """Return path to test slide file."""
    return os.path.join(test_data_path, "948176.svs")


@pytest.fixture
def patch_h5_path(test_data_path: str) -> str:
    """Return path to test patch HDF5 file."""
    return os.path.join(test_data_path, "948176.patch.h5")


@pytest.fixture
def classifier_pkl_path(test_data_path: str) -> str:
    """Return path to test classifier pickle file."""
    return os.path.join(test_data_path, "simple_classifier.pkl")


@pytest.fixture
def annotation_csv_path(test_data_path: str) -> str:
    """Return path to test annotation CSV file."""
    return os.path.join(test_data_path, "948176.annotation.csv")


@pytest.fixture
def features_pt_path(test_data_path: str) -> str:
    """Return path to test features pickle file."""
    return os.path.join(test_data_path, "948176.features.pt")


@pytest.fixture
def class_embedding_pt_path(test_data_path: str) -> str:
    """Return path to test class embedding file."""
    return os.path.join(test_data_path, "class_embedding.pt")


@pytest.fixture(autouse=True)
def reset_seed() -> Generator[None, None, None]:
    """Reset random seed to ensure reproducible test results."""
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    yield
    # Cleanup
    torch.cuda.empty_cache()


@pytest.fixture
def cpu_device() -> torch.device:
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def mock_model_factory():
    """Provide a mock model factory for testing without loading models."""
    from unittest.mock import MagicMock

    mock_model = MagicMock()
    mock_model_fun = MagicMock(return_value=torch.randn(1, 2048))
    mock_model.get_model_fun.return_value = mock_model_fun
    mock_model.get_preprocessing_fun.return_value = None

    mock_factory = MagicMock()
    mock_factory.return_value = MagicMock(get_model=MagicMock(return_value=mock_model))

    return mock_factory


@pytest.fixture
def mock_segment_tissue():
    """Provide mock tissue segmentation results."""
    from unittest.mock import MagicMock

    mock_coords = [[0, 0], [256, 0], [0, 256]]
    mock_polygon = MagicMock()
    mock_grid = MagicMock()

    mock_segment = MagicMock(return_value=(mock_polygon, mock_grid, mock_coords, None))

    return mock_segment


# Skip test if required test data is unavailable
skip_if_missing_testdata = pytest.mark.skip(
    reason="Test data not available - requires testdata directory"
)

# Skip test if GPU is unavailable
skip_if_no_gpu = pytest.mark.skip(reason="GPU is required for this test")
