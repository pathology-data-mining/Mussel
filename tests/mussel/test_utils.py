import h5py
import mussel.utils
import torch
import pickle
from mussel.models import ModelType

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def test_get_features():
    slide_path = "tests/testdata/948176.svs"
    patch_h5_path = "tests/testdata/948176.patch.h5"

    patch_h5 = h5py.File(patch_h5_path, "r")
    coords = patch_h5["coords"][:]
    attrs = patch_h5["coords"].attrs

    features, labels = mussel.utils.get_features(
        coords,
        slide_path,
        attrs,
        model_type=ModelType.RESNET50,
        use_gpu=False,
        num_workers=1,
        batch_size=16,
    )
    assert features.shape[0] == len(coords)
    assert labels.shape[0] == len(coords)


def test_segment_tissue():
    slide_path = "tests/testdata/948176.svs"
    patch_h5_path = "tests/testdata/948176.patch.h5"

    patch_h5 = h5py.File(patch_h5_path, "r")
    coords = patch_h5["coords"][:]
    attrs = patch_h5["coords"].attrs

    _, _, coords, attrs = mussel.utils.segment_tissue(
        slide_path=slide_path,
        patch_size=224,
        segment_threshold=15,
        median_blur_ksize=11,
        morphology_ex_kernel=2,
        tissue_area_threshold=2,
        hole_area_threshold=1,
        max_num_holes=2,
    )

    assert len(coords) > 0
    assert "level_dim" in attrs
    assert "hole_area_threshold" in attrs


def test_get_features_with_slide_encoder():
    """Test get_features with slide-level encoding."""
    slide_path = "tests/testdata/948176.svs"
    patch_h5_path = "tests/testdata/948176.patch.h5"

    patch_h5 = h5py.File(patch_h5_path, "r")
    coords = patch_h5["coords"][:]
    attrs = patch_h5["coords"].attrs

    # Test with mean pooling aggregation
    features, labels = mussel.utils.get_features(
        coords,
        slide_path,
        attrs,
        model_type=ModelType.RESNET50,
        use_gpu=False,
        num_workers=1,
        batch_size=16,
        use_slide_encoder=True,
        aggregation_method="mean",
    )
    # Mean pooling should produce a single feature vector
    assert features.shape[0] == 1
    assert labels.shape[0] == len(coords)


def test_get_features_auto_infer_patch_encoder():
    """Test that patch encoder is automatically inferred from slide encoder."""
    slide_path = "tests/testdata/948176.svs"
    patch_h5_path = "tests/testdata/948176.patch.h5"

    patch_h5 = h5py.File(patch_h5_path, "r")
    coords = patch_h5["coords"][:]
    attrs = patch_h5["coords"].attrs

    # Specify GIGAPATH_SLIDE without specifying the patch encoder
    # It should automatically use GIGAPATH
    features, labels = mussel.utils.get_features(
        coords,
        slide_path,
        attrs,
        model_type=ModelType.CLIP,  # This should be overridden to GIGAPATH
        use_gpu=False,
        num_workers=1,
        batch_size=16,
        use_slide_encoder=True,
        aggregation_method="mean",  # Use mean instead of model for simplicity
        slide_model_type=None,  # Not using model-based, so no auto-inference
    )
    # Should work without error
    assert features.shape[0] == 1
