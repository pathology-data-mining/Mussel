import os
import pytest
from omegaconf import OmegaConf

import mussel.cli.extract_features
from mussel.cli.extract_features import ExtractFeaturesConfig
from mussel.models import ModelType, validate_slide_encoder_compatibility
from mussel.utils import extract_patch_features, aggregate_slide_features

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


@pytest.mark.slow
@pytest.mark.integration
def test_extract_features(tmp_path, test_data_path, patch_h5_path):
    slide_path = os.path.join(test_data_path, "948176.svs")
    output_h5_path = tmp_path / "test.h5"
    output_pt_path = tmp_path / "test.pt"
    cfg = ExtractFeaturesConfig(
        slide_path=slide_path,
        patch_h5_path=patch_h5_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        num_workers=1,
        model_type=ModelType.RESNET50,
        use_gpu=False,
    )
    mussel.cli.extract_features.main(OmegaConf.create(cfg))
    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)


@pytest.mark.slow
@pytest.mark.integration
def test_extract_features_two_step(tmp_path, test_data_path, patch_h5_path):
    """Test two-step feature extraction with identity aggregation."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    output_h5_path = tmp_path / "test.h5"
    output_pt_path = tmp_path / "test.pt"
    intermediate_h5_path = tmp_path / "test.patch.h5"
    
    cfg = ExtractFeaturesConfig(
        slide_path=slide_path,
        patch_h5_path=patch_h5_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        num_workers=1,
        model_type=ModelType.RESNET50,
        use_gpu=False,
        intermediate_h5_path=intermediate_h5_path,
        aggregation_method="mean",  # Two-step mode inferred from aggregation_method
    )
    mussel.cli.extract_features.main(OmegaConf.create(cfg))
    
    # Check that both intermediate and final outputs exist
    assert os.path.exists(intermediate_h5_path)
    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)


@pytest.mark.slow
@pytest.mark.integration
def test_extract_patch_features_direct(tmp_path, test_data_path, patch_h5_path):
    """Test direct call to extract_patch_features function."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    output_h5_path = tmp_path / "patch_features.h5"
    
    result = extract_patch_features(
        patch_h5_path=patch_h5_path,
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        model_type=ModelType.RESNET50,
        use_gpu=False,
        num_workers=1,
    )
    
    assert os.path.exists(result)
    assert str(result) == str(output_h5_path)


@pytest.mark.slow
@pytest.mark.integration
def test_aggregate_slide_features_identity(tmp_path, test_data_path, patch_h5_path):
    """Test aggregate_slide_features with identity aggregation."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    patch_features_h5_path = tmp_path / "patch_features.h5"
    output_h5_path = tmp_path / "slide_features.h5"
    output_pt_path = tmp_path / "slide_features.pt"
    
    # First extract patch features
    extract_patch_features(
        patch_h5_path=patch_h5_path,
        slide_path=slide_path,
        output_h5_path=patch_features_h5_path,
        model_type=ModelType.RESNET50,
        use_gpu=False,
        num_workers=1,
    )
    
    # Then aggregate them
    h5_result, pt_result = aggregate_slide_features(
        patch_features_h5_path=patch_features_h5_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        aggregation_method="identity",
    )
    
    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)
    assert str(h5_result) == str(output_h5_path)
    assert str(pt_result) == str(output_pt_path)


@pytest.mark.slow
@pytest.mark.integration
def test_aggregate_slide_features_mean(tmp_path, test_data_path, patch_h5_path):
    """Test aggregate_slide_features with mean pooling."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    patch_features_h5_path = tmp_path / "patch_features.h5"
    output_h5_path = tmp_path / "slide_features_mean.h5"
    output_pt_path = tmp_path / "slide_features_mean.pt"
    
    # First extract patch features
    extract_patch_features(
        patch_h5_path=patch_h5_path,
        slide_path=slide_path,
        output_h5_path=patch_features_h5_path,
        model_type=ModelType.RESNET50,
        use_gpu=False,
        num_workers=1,
    )
    
    # Then aggregate with mean pooling
    h5_result, pt_result = aggregate_slide_features(
        patch_features_h5_path=patch_features_h5_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        aggregation_method="mean",
    )
    
    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)
    assert str(h5_result) == str(output_h5_path)
    assert str(pt_result) == str(output_pt_path)


def test_slide_encoder_compatibility_validation():
    """Test that slide encoder compatibility validation works correctly."""
    # Valid combination: GIGAPATH with GIGAPATH_SLIDE
    validate_slide_encoder_compatibility(ModelType.GIGAPATH, ModelType.GIGAPATH_SLIDE)
    
    # Invalid combination: CLIP with GIGAPATH_SLIDE should raise ValueError
    with pytest.raises(ValueError, match="requires patch encoder"):
        validate_slide_encoder_compatibility(ModelType.CLIP, ModelType.GIGAPATH_SLIDE)


def test_auto_infer_patch_encoder():
    """Test that patch encoder is automatically inferred from slide encoder."""
    from mussel.models import get_required_patch_encoder
    
    # GIGAPATH_SLIDE should require GIGAPATH
    required = get_required_patch_encoder(ModelType.GIGAPATH_SLIDE)
    assert required == ModelType.GIGAPATH
    
    # TITAN_SLIDE should require CONCH1_5
    required = get_required_patch_encoder(ModelType.TITAN_SLIDE)
    assert required == ModelType.CONCH1_5


def test_auto_set_aggregation_method(tmp_path):
    """Test that aggregation_method is automatically set to 'model' when slide_model_type is specified."""
    import unittest.mock as mock
    from mussel.utils import save_features
    
    slide_path = "tests/testdata/948176.svs"
    patch_h5_path = "tests/testdata/948176.patch.h5"
    output_h5_path = tmp_path / "test.h5"
    
    # Mock the actual feature extraction to avoid heavy computation
    with mock.patch('mussel.utils.feature_extract.extract_patch_features') as mock_extract, \
         mock.patch('mussel.utils.feature_extract.aggregate_slide_features') as mock_aggregate:
        
        # Call with slide_model_type - aggregation_method should be auto-set to "model"
        save_features(
            patch_h5_path=patch_h5_path,
            slide_path=slide_path,
            output_h5_path=output_h5_path,
            model_type=ModelType.GIGAPATH,  # Will be auto-set
            slide_model_type=ModelType.GIGAPATH_SLIDE,
            # aggregation_method should be auto-set to "model"
            use_gpu=False,
            num_workers=1,
        )
        
        # Verify aggregate_slide_features was called with aggregation_method="model"
        mock_aggregate.assert_called_once()
        call_kwargs = mock_aggregate.call_args[1]
        assert call_kwargs['aggregation_method'] == 'model'
