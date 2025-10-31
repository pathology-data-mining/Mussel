import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from omegaconf import OmegaConf
import h5py
import torch

from mussel.cli.tessellate_extract_features import TessellateExtractFeaturesConfig, main
from mussel.cli.tessellate import SegConfig
from mussel.models import ModelType


def test_tessellate_extract_features(tmp_path):
    """Test the integrated tessellate-extract-features workflow with dual extraction."""
    slide_path = "tests/testdata/948176.svs"
    classifier_pkl = "tests/testdata/simple_classifier.pkl"
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
        postfilter_model_type=None,  # Will use same model as prefilter
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


def test_tessellate_extract_features_with_different_models(tmp_path):
    """Test with different models for pre-filter and post-filter extraction."""
    slide_path = "tests/testdata/948176.svs"
    classifier_pkl = "tests/testdata/simple_classifier.pkl"
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
        postfilter_model_type=ModelType.RESNET50,  # Second extraction also with ResNet50
        seg_config=seg_config,
        num_workers=1,
        batch_size=32,
        use_gpu=False,
        keep_intermediate_files=False,
    )

    main(OmegaConf.create(cfg))

    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)


def test_tessellate_extract_features_with_intermediate_files(tmp_path):
    """Test the integrated workflow while keeping intermediate files."""
    slide_path = "tests/testdata/948176.svs"
    classifier_pkl = "tests/testdata/simple_classifier.pkl"
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


def test_tessellate_extract_features_with_visualizations(tmp_path):
    """Test the workflow with optional visualization outputs."""
    slide_path = "tests/testdata/948176.svs"
    classifier_pkl = "tests/testdata/simple_classifier.pkl"
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


def test_tessellate_extract_features_without_filtering(tmp_path):
    """Test the workflow without filtering (classifier_pkl=None)."""
    slide_path = "tests/testdata/948176.svs"
    output_h5_path = os.path.join(tmp_path, "features.h5")
    output_pt_path = os.path.join(tmp_path, "features.pt")

    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateExtractFeaturesConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=None,  # No filtering
        # Note: prefilter_model_type is NOT specified - it's not needed when there's no filtering
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


def test_tessellate_extract_features_with_slide_encoder_inference(tmp_path):
    """Test that postfilter_model_type is inferred from slide_model_type when using model aggregation."""
    
    slide_path = "tests/testdata/948176.svs"
    output_h5_path = os.path.join(tmp_path, "features.h5")
    output_pt_path = os.path.join(tmp_path, "features.pt")
    
    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateExtractFeaturesConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=None,  # No filtering
        # Note: prefilter_model_type is NOT specified - it's not needed when there's no filtering
        # Note: postfilter_model_type is also NOT specified - it will be inferred from slide_model_type
        postfilter_model_type=None,
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
        patch('mussel.cli.tessellate_extract_features.save_features') as mock_save_features,
        patch('mussel.cli.tessellate_extract_features.segment_tissue') as mock_segment
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


def test_tessellate_extract_features_without_filtering_no_prefilter_params(tmp_path):
    """Test that prefilter_model_type and prefilter_model_path are not required when classifier_pkl is None."""
    slide_path = "tests/testdata/948176.svs"
    output_h5_path = os.path.join(tmp_path, "features.h5")
    output_pt_path = os.path.join(tmp_path, "features.pt")

    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateExtractFeaturesConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=None,  # No filtering
        # Explicitly NOT setting prefilter_model_type or prefilter_model_path
        # This should work fine now since they're only needed when filtering is enabled
        seg_config=seg_config,
        num_workers=1,
        batch_size=32,
        use_gpu=False,
        keep_intermediate_files=False,
    )
    
    # Mock save_features and segment_tissue to verify the workflow runs without prefilter params
    with (
        patch('mussel.cli.tessellate_extract_features.save_features') as mock_save_features,
        patch('mussel.cli.tessellate_extract_features.segment_tissue') as mock_segment
    ):
        
        # Mock segment_tissue to return fake data
        mock_coords = [[0, 0], [256, 0], [0, 256]]
        mock_polygon = MagicMock()
        mock_grid = MagicMock()
        mock_segment.return_value = (mock_polygon, mock_grid, mock_coords, None)
        
        # Run main - this should work without errors
        main(OmegaConf.create(cfg))
        
        # Verify save_features was called once (no prefilter step since no filtering)
        assert mock_save_features.call_count == 1
        
        # Verify the model_type used is the default (CTRANSPATH) since postfilter_model_type wasn't specified
        call_args = mock_save_features.call_args
        assert call_args.kwargs['model_type'] == ModelType.CTRANSPATH, \
            f"Expected default model CTRANSPATH, got {call_args.kwargs['model_type']}"
