"""Tests for get_features pre-loaded model parameters."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mussel.models import ModelType
from mussel.utils.feature_extract import get_features


def _make_mock_model(feature_dim=384):
    """Return a mock model with the required interface."""
    mock = MagicMock()
    mock.get_preprocessing_fun.return_value = None
    mock.get_model_fun.return_value = MagicMock(
        side_effect=lambda x: __import__('torch').randn(len(x), feature_dim)
    )
    return mock


def _base_patches():
    """Context managers that stub out all I/O in get_features."""
    return [
        patch("mussel.utils.feature_extract.get_model_factory"),
        patch("mussel.utils.feature_extract.WholeSlideImageTileCoordDataset"),
        patch("mussel.utils.feature_extract.process_dataset"),
        patch("mussel.utils.feature_extract._make_dataloader"),
    ]


# -- model= parameter -------------------------------------------------------

def test_model_factory_not_called_when_model_provided():
    """get_model_factory must not be called when a pre-loaded model is given."""
    coords = np.zeros((10, 2), dtype=np.int32)
    attrs = {"patch_size": 256, "patch_level": 0, "mpp": 0.5,
             "patch_size_to_resize_to_for_desired_mpp": 224}
    mock_model = _make_mock_model()

    with patch("mussel.utils.feature_extract.get_model_factory") as mock_factory,          patch("mussel.utils.feature_extract.WholeSlideImageTileCoordDataset"),          patch("mussel.utils.feature_extract._make_dataloader"),          patch("mussel.utils.feature_extract.process_dataset") as mock_proc:
        mock_proc.return_value = MagicMock(
            features=np.zeros((10, 384)), labels=np.zeros(10)
        )
        get_features(coords, "slide.svs", attrs, model=mock_model)

    mock_factory.assert_not_called()


def test_model_factory_called_when_model_not_provided():
    """get_model_factory must be called when no pre-loaded model is given."""
    coords = np.zeros((10, 2), dtype=np.int32)
    attrs = {"patch_size": 256, "patch_level": 0, "mpp": 0.5,
             "patch_size_to_resize_to_for_desired_mpp": 224}

    with patch("mussel.utils.feature_extract.get_model_factory") as mock_factory,          patch("mussel.utils.feature_extract.WholeSlideImageTileCoordDataset"),          patch("mussel.utils.feature_extract._make_dataloader"),          patch("mussel.utils.feature_extract.process_dataset") as mock_proc:
        mock_model = _make_mock_model()
        mock_factory.return_value = MagicMock(get_model=MagicMock(return_value=mock_model))
        mock_proc.return_value = MagicMock(
            features=np.zeros((10, 384)), labels=np.zeros(10)
        )
        get_features(coords, "slide.svs", attrs, model_type=ModelType.CTRANSPATH)

    mock_factory.assert_called_once()


def test_positional_args_unchanged():
    """Existing positional call pattern must still work after adding model= at end."""
    coords = np.zeros((5, 2), dtype=np.int32)
    attrs = {"patch_size": 256, "patch_level": 0, "mpp": 0.5,
             "patch_size_to_resize_to_for_desired_mpp": 224}

    with patch("mussel.utils.feature_extract.get_model_factory") as mock_factory,          patch("mussel.utils.feature_extract.WholeSlideImageTileCoordDataset"),          patch("mussel.utils.feature_extract._make_dataloader"),          patch("mussel.utils.feature_extract.process_dataset") as mock_proc:
        mock_model = _make_mock_model()
        mock_factory.return_value = MagicMock(get_model=MagicMock(return_value=mock_model))
        mock_proc.return_value = MagicMock(
            features=np.zeros((5, 384)), labels=np.zeros(5)
        )
        # Classic positional call: (coords, slide_path, attrs, model_type, model_path, batch_size)
        # batch_size must not be interpreted as model=
        features, labels = get_features(
            coords, "slide.svs", attrs, ModelType.CTRANSPATH, None, 32
        )
    assert features.shape == (5, 384)


def test_model_invalid_interface_raises_type_error():
    """Passing an object without the required methods must raise TypeError immediately."""
    coords = np.zeros((5, 2), dtype=np.int32)
    attrs = {"patch_size": 256, "patch_level": 0, "mpp": 0.5,
             "patch_size_to_resize_to_for_desired_mpp": 224}
    bad_model = object()

    with pytest.raises(TypeError, match="get_preprocessing_fun"):
        get_features(coords, "slide.svs", attrs, model=bad_model)


# -- slide_model= parameter ------------------------------------------------

def test_slide_model_factory_not_called_when_slide_model_provided():
    """get_model_factory must not be loaded for the slide encoder when slide_model is given."""
    coords = np.zeros((10, 2), dtype=np.int32)
    attrs = {"patch_size": 256, "patch_level": 0, "mpp": 0.5,
             "patch_size_to_resize_to_for_desired_mpp": 224}
    mock_patch_model = _make_mock_model()
    mock_slide_model = MagicMock()
    mock_slide_model.get_model_fun.return_value = MagicMock(
        return_value=__import__('torch').zeros(1, 512)
    )

    call_log = []
    def factory_side_effect(model_type):
        call_log.append(model_type)
        m = MagicMock()
        m.get_model.return_value = mock_patch_model
        return m

    with patch("mussel.utils.feature_extract.get_model_factory", side_effect=factory_side_effect),          patch("mussel.utils.feature_extract.validate_slide_encoder_compatibility"),          patch("mussel.utils.feature_extract.WholeSlideImageTileCoordDataset"),          patch("mussel.utils.feature_extract._make_dataloader"),          patch("mussel.utils.feature_extract.process_dataset") as mock_proc:
        mock_proc.return_value = MagicMock(
            features=np.zeros((10, 384)), labels=np.zeros(10)
        )
        get_features(
            coords, "slide.svs", attrs,
            model=mock_patch_model,
            use_slide_encoder=True,
            slide_model_type=ModelType.GIGAPATH_SLIDE,
            aggregation_method="model",
            slide_model=mock_slide_model,
        )

    # Only the patch encoder factory may be called (for auto-infer check), not the slide encoder
    slide_encoder_calls = [t for t in call_log if t == ModelType.GIGAPATH_SLIDE]
    assert len(slide_encoder_calls) == 0, "Slide encoder factory must not be called when slide_model is provided"


def test_slide_model_invalid_interface_raises_type_error():
    """Passing a slide_model without get_model_fun must raise TypeError."""
    coords = np.zeros((5, 2), dtype=np.int32)
    attrs = {"patch_size": 256, "patch_level": 0, "mpp": 0.5,
             "patch_size_to_resize_to_for_desired_mpp": 224}
    bad_slide_model = object()

    with pytest.raises(TypeError, match="get_model_fun"):
        get_features(coords, "slide.svs", attrs, slide_model=bad_slide_model)


# -- compatibility validation ----b??-----------------------------------------

def test_compatibility_validated_with_preloaded_patch_model():
    """validate_slide_encoder_compatibility must be called even with pre-loaded patch model."""
    coords = np.zeros((5, 2), dtype=np.int32)
    attrs = {"patch_size": 256, "patch_level": 0, "mpp": 0.5,
             "patch_size_to_resize_to_for_desired_mpp": 224}
    mock_patch_model = _make_mock_model()

    with patch("mussel.utils.feature_extract.validate_slide_encoder_compatibility") as mock_validate,          patch("mussel.utils.feature_extract.WholeSlideImageTileCoordDataset"),          patch("mussel.utils.feature_extract._make_dataloader"),          patch("mussel.utils.feature_extract.process_dataset") as mock_proc:
        mock_proc.return_value = MagicMock(
            features=np.zeros((5, 384)), labels=np.zeros(5)
        )
        mock_slide = MagicMock()
        mock_slide.get_model_fun.return_value = MagicMock(return_value=__import__('torch').zeros(1, 512))
        get_features(
            coords, "slide.svs", attrs,
            model_type=ModelType.GIGAPATH,
            model=mock_patch_model,
            use_slide_encoder=True,
            slide_model_type=ModelType.GIGAPATH_SLIDE,
            aggregation_method="model",
            slide_model=mock_slide,
        )

    mock_validate.assert_called_once_with(ModelType.GIGAPATH, ModelType.GIGAPATH_SLIDE)
