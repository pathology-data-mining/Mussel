import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from omegaconf import OmegaConf
import h5py
import torch

from mussel.cli.tessellate_extract_features import TessellateExtractFeaturesConfig, main
from mussel.cli.tessellate import SegConfig
from mussel.models import ModelType


_TEF_REQUIRED = dict(
    slide_path="test.svs",
    output_h5_path="test.h5",
    output_pt_path="test.pt",
)


@pytest.mark.parametrize("model_type,expected_patch_size", [
    (ModelType.CONCH1_5, 512),
    (ModelType.VIRCHOW, 224),
    (ModelType.CLIP, 224),
    (ModelType.GOOGLEPATH, 224),
    (ModelType.GIGAPATH, SegConfig.DEFAULT_PATCH_SIZE),
])
def test_default_patch_size_for_model(model_type, expected_patch_size):
    """Test that patch size is automatically set based on model type."""
    cfg = TessellateExtractFeaturesConfig(
        **_TEF_REQUIRED,
        prefilter_model_type=model_type,
        seg_config=SegConfig(),
    )
    assert cfg.seg_config.patch_size == expected_patch_size


def test_explicit_patch_size_preserved():
    """Test that explicitly set patch size is not overridden."""
    cfg = TessellateExtractFeaturesConfig(
        **_TEF_REQUIRED,
        prefilter_model_type=ModelType.CONCH1_5,
        seg_config=SegConfig(patch_size=384),
    )
    assert cfg.seg_config.patch_size == 384


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_tessellate_extract_features(tmp_path, test_data_path, classifier_pkl_path):
    """Test the integrated tessellate-extract-features workflow with dual extraction."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    classifier_pkl = classifier_pkl_path
    output_h5_path = os.path.join(tmp_path, "filtered_features.h5")
    output_pt_path = os.path.join(tmp_path, "filtered_features.pt")

    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateExtractFeaturesConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=classifier_pkl,
        classifier_threshold=0.75,
        prefilter_model_type=ModelType.RESNET50,  # Use ResNet50 for pre-filtering
        model_type=None,  # Will use same model as prefilter
        seg_config=seg_config,
        num_workers=1,
        batch_size=32,
        use_gpu=False,  # Use CPU for testing
        keep_intermediate_files=False,
    )

    main(OmegaConf.create(cfg))

    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)

    # Check that the output files are not empty
    with h5py.File(output_h5_path, "r") as f:
        assert "features" in f
        assert "coords" in f
        assert f["features"].shape[0] > 0
        assert f["coords"].shape[0] > 0

    data = torch.load(output_pt_path, weights_only=True)
    assert data.shape[0] > 0


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_tessellate_extract_features_with_different_models(tmp_path, test_data_path, classifier_pkl_path):
    """Test with different models for pre-filter and post-filter extraction."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    classifier_pkl = classifier_pkl_path
    output_h5_path = os.path.join(tmp_path, "features.h5")
    output_pt_path = os.path.join(tmp_path, "features.pt")

    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateExtractFeaturesConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=classifier_pkl,
        classifier_threshold=0.75,
        prefilter_model_type=ModelType.RESNET50,  # First extraction with ResNet50
        model_type=ModelType.RESNET50,  # Second extraction also with ResNet50
        seg_config=seg_config,
        num_workers=1,
        batch_size=32,
        use_gpu=False,
        keep_intermediate_files=False,
    )

    main(OmegaConf.create(cfg))

    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_tessellate_extract_features_with_intermediate_files(tmp_path, test_data_path, classifier_pkl_path):
    """Test the integrated workflow while keeping intermediate files."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    classifier_pkl = classifier_pkl_path
    output_h5_path = os.path.join(tmp_path, "features.h5")
    output_pt_path = os.path.join(tmp_path, "features.pt")

    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateExtractFeaturesConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=classifier_pkl,
        classifier_threshold=0.75,
        prefilter_model_type=ModelType.RESNET50,
        seg_config=seg_config,
        num_workers=1,
        batch_size=32,
        use_gpu=False,
        keep_intermediate_files=True,  # Keep intermediate files
    )

    main(OmegaConf.create(cfg))

    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)

    # Check that intermediate files were created
    base_path = Path(output_h5_path).parent
    tessellate_h5_path = base_path / f"{Path(slide_path).stem}.tessellate.h5"
    prefilter_features_h5_path = base_path / f"{Path(slide_path).stem}.prefilter_features.h5"
    filtered_coords_h5_path = base_path / f"{Path(slide_path).stem}.filtered_coords.h5"
    
    assert os.path.exists(tessellate_h5_path)
    assert os.path.exists(prefilter_features_h5_path)
    assert os.path.exists(filtered_coords_h5_path)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_tessellate_extract_features_with_visualizations(tmp_path, test_data_path, classifier_pkl_path):
    """Test the workflow with optional visualization outputs."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    classifier_pkl = classifier_pkl_path
    output_h5_path = os.path.join(tmp_path, "features.h5")
    output_pt_path = os.path.join(tmp_path, "features.pt")
    output_mask_path = os.path.join(tmp_path, "mask.png")
    output_grid_mask_path = os.path.join(tmp_path, "grid.png")
    output_thumbnail_path = os.path.join(tmp_path, "thumbnail.png")

    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateExtractFeaturesConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=classifier_pkl,
        classifier_threshold=0.75,
        output_mask_path=output_mask_path,
        output_grid_mask_path=output_grid_mask_path,
        output_thumbnail_path=output_thumbnail_path,
        prefilter_model_type=ModelType.RESNET50,
        seg_config=seg_config,
        num_workers=1,
        batch_size=32,
        use_gpu=False,
        keep_intermediate_files=False,
    )

    main(OmegaConf.create(cfg))

    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)
    assert os.path.exists(output_mask_path)
    assert os.path.exists(output_grid_mask_path)
    assert os.path.exists(output_thumbnail_path)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_tessellate_extract_features_without_filtering(tmp_path, test_data_path):
    """Test the workflow without filtering (classifier_pkl=None)."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    output_h5_path = os.path.join(tmp_path, "features.h5")
    output_pt_path = os.path.join(tmp_path, "features.pt")

    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateExtractFeaturesConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=None,  # No filtering
        prefilter_model_type=ModelType.RESNET50,
        seg_config=seg_config,
        num_workers=1,
        batch_size=32,
        use_gpu=False,
        keep_intermediate_files=False,
    )

    main(OmegaConf.create(cfg))

    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)

    # Check that the output files are not empty
    with h5py.File(output_h5_path, "r") as f:
        assert "features" in f
        assert "coords" in f
        assert f["features"].shape[0] > 0
        assert f["coords"].shape[0] > 0

    data = torch.load(output_pt_path, weights_only=True)
    assert data.shape[0] > 0


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_tessellate_extract_features_with_slide_encoder_inference(tmp_path, test_data_path):
    """Test that model_type is inferred from slide_model_type when using model aggregation."""

    slide_path = os.path.join(test_data_path, "948176.svs")
    output_h5_path = os.path.join(tmp_path, "features.h5")
    output_pt_path = os.path.join(tmp_path, "features.pt")
    
    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateExtractFeaturesConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=None,  # No filtering
        prefilter_model_type=ModelType.RESNET50,
        # Note: model_type is NOT specified
        model_type=None,
        aggregation_method="model",
        slide_model_type=ModelType.GIGAPATH_SLIDE,  # Requires GIGAPATH patch encoder
        seg_config=seg_config,
        num_workers=1,
        batch_size=32,
        use_gpu=False,
        keep_intermediate_files=False,
    )
    
    # Mock save_features to capture the model_type parameter
    with (
        patch('mussel.cli.tessellate_extract_features_common.save_features') as mock_save_features,
        patch('mussel.cli.tessellate_extract_features_common.segment_tissue') as mock_segment
    ):
        
        # Mock segment_tissue to return fake data
        mock_coords = [[0, 0], [256, 0], [0, 256]]
        mock_polygon = MagicMock()
        mock_grid = MagicMock()
        mock_segment.return_value = (mock_polygon, mock_grid, mock_coords, None)
        
        # Run main
        main(OmegaConf.create(cfg))
        
        # Verify save_features was called with the correct model_type
        assert mock_save_features.called
        call_args = mock_save_features.call_args
        
        # The model_type should be GIGAPATH (inferred from GIGAPATH_SLIDE)
        assert (
            call_args.kwargs['model_type'] == ModelType.GIGAPATH
        ), f"Expected model_type to be GIGAPATH (inferred from slide_model_type), got {call_args.kwargs['model_type']}"
        assert call_args.kwargs['slide_model_type'] == ModelType.GIGAPATH_SLIDE
        assert call_args.kwargs['aggregation_method'] == "model"
