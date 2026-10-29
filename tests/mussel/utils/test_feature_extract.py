"""Tests for feature extraction utilities, particularly batch processing."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import h5py
import numpy as np
import torch
import pytest

from mussel.utils.feature_extract import extract_patch_features_batch
from mussel.models import ModelType


def create_mock_h5_file(h5_path, num_patches=10):
    """Create a mock HDF5 file with patch coordinates."""
    coords = np.array([[i * 256, i * 256] for i in range(num_patches)])
    
    with h5py.File(h5_path, "w") as f:
        coords_dset = f.create_dataset("coords", data=coords)
        coords_dset.attrs["patch_size"] = 256
        coords_dset.attrs["patch_level"] = 0
        coords_dset.attrs["patch_size_to_resize_to_for_desired_mpp"] = 224


def test_extract_patch_features_batch_basic(tmp_path):
    """Test basic batch extraction of patch features from multiple slides."""
    # Create mock input files
    num_slides = 3
    patch_h5_paths = []
    slide_paths = []
    output_h5_paths = []
    
    for i in range(num_slides):
        # Create mock patch coordinates file
        patch_h5_path = tmp_path / f"slide{i}_coords.h5"
        create_mock_h5_file(patch_h5_path, num_patches=5)
        patch_h5_paths.append(str(patch_h5_path))
        
        # Mock slide path (doesn't need to exist due to mocking)
        slide_paths.append(f"slide{i}.svs")
        
        # Output path
        output_h5_paths.append(str(tmp_path / f"slide{i}_features.h5"))
    
    # Mock the model and dataset to avoid loading actual slides
    with (
        patch('mussel.utils.feature_extract.get_model_factory') as mock_factory,
        patch('mussel.utils.feature_extract.WholeSlideImageH5Dataset') as mock_dataset_class,
        # A MagicMock dataset matches no singledispatch registration, so stub it out
        patch('mussel.utils.feature_extract.process_dataset') as mock_process_dataset,
    ):
        # Mock model
        mock_model = MagicMock()
        mock_model_fun = MagicMock(
            side_effect=lambda x: torch.randn(len(x), 2048)  # Return batch of features
        )
        mock_model.get_model_fun.return_value = mock_model_fun
        mock_model.get_preprocessing_fun.return_value = None
        mock_factory.return_value = MagicMock(get_model=MagicMock(return_value=mock_model))
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 5
        mock_dataset_class.return_value = mock_dataset
        
        # Call the batch extraction function
        result_paths = extract_patch_features_batch(
            patch_h5_paths=patch_h5_paths,
            slide_paths=slide_paths,
            output_h5_paths=output_h5_paths,
            model_type=ModelType.RESNET50,
            model_path=None,
            batch_size=32,
            use_gpu=False,
            num_workers=0,
        )
        
        # Verify model was loaded only once
        assert mock_factory.call_count == 1, "Model factory should be called only once"
        
        # Verify dataset was created for each slide
        assert mock_dataset_class.call_count == num_slides, \
            f"Dataset should be created {num_slides} times, got {mock_dataset_class.call_count}"

        # Verify each slide was processed
        assert mock_process_dataset.call_count == num_slides
        
        # Verify result paths match input
        assert result_paths == output_h5_paths


def test_extract_patch_features_batch_empty_list():
    """Test that batch extraction handles empty input gracefully."""
    result = extract_patch_features_batch(
        patch_h5_paths=[],
        slide_paths=[],
        output_h5_paths=[],
        model_type=ModelType.RESNET50,
    )
    
    assert result == [], "Should return empty list for empty input"


def test_extract_patch_features_batch_single_slide(tmp_path):
    """Test batch extraction with a single slide (edge case)."""
    # Create mock input file
    patch_h5_path = tmp_path / "slide_coords.h5"
    create_mock_h5_file(patch_h5_path, num_patches=10)
    
    patch_h5_paths = [str(patch_h5_path)]
    slide_paths = ["slide.svs"]
    output_h5_paths = [str(tmp_path / "slide_features.h5")]
    
    with (
        patch('mussel.utils.feature_extract.get_model_factory') as mock_factory,
        patch('mussel.utils.feature_extract.WholeSlideImageH5Dataset') as mock_dataset_class,
        # A MagicMock dataset matches no singledispatch registration, so stub it out
        patch('mussel.utils.feature_extract.process_dataset') as mock_process_dataset,
    ):
        # Mock model
        mock_model = MagicMock()
        mock_model_fun = MagicMock(
            side_effect=lambda x: torch.randn(len(x), 2048)
        )
        mock_model.get_model_fun.return_value = mock_model_fun
        mock_model.get_preprocessing_fun.return_value = None
        mock_factory.return_value = MagicMock(get_model=MagicMock(return_value=mock_model))
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset
        
        result_paths = extract_patch_features_batch(
            patch_h5_paths=patch_h5_paths,
            slide_paths=slide_paths,
            output_h5_paths=output_h5_paths,
            model_type=ModelType.RESNET50,
            model_path=None,
            batch_size=32,
            use_gpu=False,
            num_workers=0,
        )
        
        # Even with one slide, model should be loaded only once
        assert mock_factory.call_count == 1
        assert mock_process_dataset.call_count == 1
        assert result_paths == output_h5_paths


def test_extract_patch_features_batch_model_reuse():
    """
    Document that batch processing reuses the model across slides.
    
    This is the key benefit of batch processing:
    - Old approach: Load model N times for N slides
    - New approach: Load model 1 time for N slides
    
    For 100 slides with 2s model load time:
    - Old: 100 * 2s = 200s wasted on model loading
    - New: 1 * 2s = 2s for model loading
    - Savings: 198s (99% reduction in model loading time)
    """
    # This is a documentation test to highlight the key benefit
    # In production, this translates to significant time savings
    pass


def _call_get_features_with_preloaded_model(**kwargs):
    """Call get_features with a pre-loaded mock model, stubbing out slide I/O."""
    from mussel.utils.feature_extract import get_features

    mock_model = MagicMock()
    mock_model.get_preprocessing_fun.return_value = None
    coords = np.array([[0, 0], [256, 256]])
    patch_features = np.zeros((2, 8))

    with (
        patch("mussel.utils.feature_extract.get_model_factory") as mock_factory,
        patch("mussel.utils.feature_extract.WholeSlideImageTileCoordDataset"),
        patch("mussel.utils.feature_extract.DataLoader"),
        patch(
            "mussel.utils.feature_extract.process_dataset",
            return_value=(patch_features, np.zeros(2)),
        ),
        patch(
            "mussel.utils.feature_extract._apply_slide_aggregation",
            return_value=np.zeros((1, 8)),
        ) as mock_aggregate,
    ):
        get_features(
            coords,
            "slide.svs",
            {"patch_size": 256},
            model=mock_model,
            use_gpu=False,
            num_workers=0,
            **kwargs,
        )
    return mock_factory, mock_aggregate


def test_get_features_preloaded_model_auto_sets_model_aggregation():
    """A pre-loaded model must still switch aggregation to the slide encoder."""
    mock_factory, mock_aggregate = _call_get_features_with_preloaded_model(
        model_type=ModelType.GIGAPATH,
        use_slide_encoder=True,
        slide_model_type=ModelType.GIGAPATH_SLIDE,
    )

    mock_factory.assert_not_called()
    assert mock_aggregate.call_args.kwargs["aggregation_method"] == "model"


def test_get_features_preloaded_model_validates_slide_encoder_compatibility():
    """A pre-loaded model whose model_type mismatches the slide encoder is rejected."""
    with pytest.raises(ValueError, match="pass model_type="):
        _call_get_features_with_preloaded_model(
            model_type=ModelType.CLIP,
            use_slide_encoder=True,
            slide_model_type=ModelType.GIGAPATH_SLIDE,
        )


def test_get_features_model_param_does_not_shift_positional_args():
    """Adding `model` must not change the position of existing parameters."""
    import inspect

    from mussel.utils.feature_extract import get_features

    params = list(inspect.signature(get_features).parameters)
    assert params.index("batch_size") == params.index("model_path") + 1
    assert params[-1] == "model"
