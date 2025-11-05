"""
Functional tests for process_dataset and related helper functions.
Tests verify that the refactored code produces correct outputs.
"""

from unittest.mock import Mock

import numpy as np
import pytest
import torch


class TestExtractFeaturesFromLoader:
    """Test the _extract_features_from_loader helper function."""

    def test_extracts_features_correctly(self):
        """Test that features are extracted and converted to numpy arrays."""
        from mussel.utils.feature_extract import _extract_features_from_loader

        # Create mock data
        batch1 = torch.randn(2, 3, 224, 224)
        labels1 = torch.tensor([0, 1])
        batch2 = torch.randn(2, 3, 224, 224)
        labels2 = torch.tensor([1, 0])
        loader = [(batch1, labels1), (batch2, labels2)]

        # Mock model function
        def mock_model_fun(batch):
            return torch.randn(batch.shape[0], 512)

        # Extract features
        results = list(_extract_features_from_loader(loader, mock_model_fun))

        assert len(results) == 2
        for count, batch, labels, features in results:
            assert isinstance(features, np.ndarray)
            assert features.shape[0] == 2  # batch size
            assert features.shape[1] == 512  # feature dimension

    def test_is_test_run_limits_batches(self):
        """Test that is_test_run=True stops after 3 batches."""
        from mussel.utils.feature_extract import _extract_features_from_loader

        # Create 5 batches
        loader = [
            (torch.randn(2, 3, 224, 224), torch.tensor([0, 1])) for _ in range(5)
        ]

        def mock_model_fun(batch):
            return torch.randn(batch.shape[0], 512)

        results = list(
            _extract_features_from_loader(loader, mock_model_fun, is_test_run=True)
        )

        # Should only process first 3 batches (indices 0, 1, 2)
        assert len(results) == 3

    def test_yields_correct_tuple_structure(self):
        """Test that each yielded item has the correct structure."""
        from mussel.utils.feature_extract import _extract_features_from_loader

        batch = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([0, 1])
        loader = [(batch, labels)]

        def mock_model_fun(batch):
            return torch.randn(batch.shape[0], 512)

        results = list(_extract_features_from_loader(loader, mock_model_fun))

        assert len(results) == 1
        count, returned_batch, returned_labels, features = results[0]
        assert count == 0
        assert torch.equal(returned_batch, batch)
        assert torch.equal(returned_labels, labels)
        assert isinstance(features, np.ndarray)


class TestProcessTileCoordDataset:
    """Test _process_tile_coord_dataset function."""

    def test_returns_concatenated_arrays(self):
        """Test that function returns properly concatenated features and labels."""
        from mussel.utils.feature_extract import _process_tile_coord_dataset

        # Create mock dataset and loader
        mock_dataset = Mock()
        loader = [
            (torch.randn(2, 3, 224, 224), torch.tensor([0, 1])),
            (torch.randn(2, 3, 224, 224), torch.tensor([1, 0])),
        ]

        def mock_model_fun(batch):
            return torch.randn(batch.shape[0], 512)

        features, labels = _process_tile_coord_dataset(
            mock_dataset, loader, mock_model_fun
        )

        # Verify output types
        assert isinstance(features, np.ndarray)
        assert isinstance(labels, np.ndarray)

        # Verify shapes - should have 4 samples total (2 batches * 2 samples each)
        assert features.shape[0] == 4
        assert features.shape[1] == 512
        assert labels.shape[0] == 4

    def test_handles_single_batch(self):
        """Test that function handles a single batch correctly."""
        from mussel.utils.feature_extract import _process_tile_coord_dataset

        mock_dataset = Mock()
        loader = [(torch.randn(3, 3, 224, 224), torch.tensor([0, 1, 2]))]

        def mock_model_fun(batch):
            return torch.randn(batch.shape[0], 256)

        features, labels = _process_tile_coord_dataset(
            mock_dataset, loader, mock_model_fun
        )

        assert features.shape == (3, 256)
        assert labels.shape == (3,)


class TestProcessDataset:
    """Test the main process_dataset function with different dataset types."""

    def test_dispatches_to_tile_coord_handler(self):
        """Test that WholeSlideImageTileCoordDataset is handled correctly."""
        from mussel.datasets import WholeSlideImageTileCoordDataset
        from mussel.utils.feature_extract import process_dataset

        mock_dataset = Mock(spec=WholeSlideImageTileCoordDataset)
        loader = [
            (torch.randn(2, 3, 224, 224), torch.tensor([0, 1])),
            (torch.randn(2, 3, 224, 224), torch.tensor([1, 0])),
        ]

        def mock_model_fun(batch):
            return torch.randn(batch.shape[0], 512)

        result = process_dataset(mock_dataset, loader, mock_model_fun)

        # Should return tuple of (features, labels)
        assert isinstance(result, tuple)
        assert len(result) == 2
        features, labels = result
        assert isinstance(features, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert features.shape[0] == 4  # 2 batches * 2 samples
        assert labels.shape[0] == 4

    def test_raises_error_for_unsupported_type(self):
        """Test that TypeError is raised for unsupported dataset types."""
        from mussel.utils.feature_extract import process_dataset

        unsupported_dataset = Mock()  # Generic mock, not a known dataset type
        loader = [(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))]

        def mock_model_fun(batch):
            return torch.randn(batch.shape[0], 512)

        with pytest.raises(TypeError, match="Unsupported dataset type"):
            process_dataset(unsupported_dataset, loader, mock_model_fun)

    def test_preserves_feature_dimensions(self):
        """Test that features maintain their dimensions through processing."""
        from mussel.datasets import WholeSlideImageTileCoordDataset
        from mussel.utils.feature_extract import process_dataset

        mock_dataset = Mock(spec=WholeSlideImageTileCoordDataset)
        feature_dim = 768  # Different dimension to test

        loader = [(torch.randn(3, 3, 224, 224), torch.tensor([0, 1, 2]))]

        def mock_model_fun(batch):
            return torch.randn(batch.shape[0], feature_dim)

        features, labels = process_dataset(mock_dataset, loader, mock_model_fun)

        assert features.shape[1] == feature_dim

    def test_is_test_run_parameter(self):
        """Test that is_test_run parameter is respected."""
        from mussel.datasets import WholeSlideImageTileCoordDataset
        from mussel.utils.feature_extract import process_dataset

        mock_dataset = Mock(spec=WholeSlideImageTileCoordDataset)
        # Create 5 batches
        loader = [
            (torch.randn(2, 3, 224, 224), torch.tensor([0, 1])) for _ in range(5)
        ]

        def mock_model_fun(batch):
            return torch.randn(batch.shape[0], 512)

        features, labels = process_dataset(
            mock_dataset, loader, mock_model_fun, is_test_run=True
        )

        # Should only process 3 batches (6 samples total)
        assert features.shape[0] == 6
        assert labels.shape[0] == 6

