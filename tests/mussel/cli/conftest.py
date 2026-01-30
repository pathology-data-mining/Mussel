"""
Shared pytest configuration and fixtures for CLI tests.

This conftest.py provides common test fixtures and configuration
used across CLI test files.
"""

import os
from pathlib import Path
from typing import Generator

import pytest
import torch
from omegaconf import OmegaConf

# Import fixtures from common conftest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    test_data_path,
    patch_h5_path,
    classifier_pkl_path,
    annotation_csv_path,
    features_pt_path,
    class_embedding_pt_path,
    skip_if_missing_testdata,
)


# Re-export fixtures for convenience
patch_h5_path = patch_h5_path
classifier_pkl_path = classifier_pkl_path
annotation_csv_path = annotation_csv_path
features_pt_path = features_pt_path
class_embedding_pt_path = class_embedding_pt_path


@pytest.fixture
def test_data_path_cli() -> str:
    """Return path to test data directory (alias)."""
    return test_data_path
