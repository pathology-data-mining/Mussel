import os
import pytest
from omegaconf import OmegaConf

import mussel.cli.extract_features
from mussel.cli.extract_features import ExtractFeaturesConfig
from mussel.models import ModelType, validate_slide_encoder_compatibility
from mussel.utils import extract_patch_features, aggregate_slide_features

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def test_extract_features(tmp_path):
    slide_path = "tests/testdata/948176.svs"
    patch_h5_path = "tests/testdata/948176.patch.h5"
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


def test_extract_features_two_step(tmp_path):
    """Test two-step feature extraction with identity aggregation."""
    slide_path = "tests/testdata/948176.svs"
    patch_h5_path = "tests/testdata/948176.patch.h5"
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
        use_two_step=True,
        intermediate_h5_path=intermediate_h5_path,
        aggregation_method="identity",
    )
    mussel.cli.extract_features.main(OmegaConf.create(cfg))
    
    # Check that both intermediate and final outputs exist
    assert os.path.exists(intermediate_h5_path)
    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)


def test_extract_patch_features_direct(tmp_path):
    """Test direct call to extract_patch_features function."""
    slide_path = "tests/testdata/948176.svs"
    patch_h5_path = "tests/testdata/948176.patch.h5"
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
    assert result == str(output_h5_path)


def test_aggregate_slide_features_identity(tmp_path):
    """Test aggregate_slide_features with identity aggregation."""
    slide_path = "tests/testdata/948176.svs"
    patch_h5_path = "tests/testdata/948176.patch.h5"
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
    assert h5_result == str(output_h5_path)
    assert pt_result == str(output_pt_path)


def test_aggregate_slide_features_mean(tmp_path):
    """Test aggregate_slide_features with mean pooling."""
    slide_path = "tests/testdata/948176.svs"
    patch_h5_path = "tests/testdata/948176.patch.h5"
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
    assert h5_result == str(output_h5_path)
    assert pt_result == str(output_pt_path)


def test_slide_encoder_compatibility_validation():
    """Test that slide encoder compatibility validation works correctly."""
    # Valid combination: GIGAPATH with GIGAPATH_SLIDE
    validate_slide_encoder_compatibility(ModelType.GIGAPATH, ModelType.GIGAPATH_SLIDE)
    
    # Invalid combination: CLIP with GIGAPATH_SLIDE should raise ValueError
    with pytest.raises(ValueError, match="requires patch encoder"):
        validate_slide_encoder_compatibility(ModelType.CLIP, ModelType.GIGAPATH_SLIDE)
