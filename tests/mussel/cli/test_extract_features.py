import os
import ssl

import h5py
import ml_dtypes
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import mussel.cli.extract_features
from mussel.cli.extract_features import ExtractFeaturesConfig
from mussel.models import ModelType, validate_slide_encoder_compatibility
from mussel.utils import aggregate_slide_features, extract_patch_features

ssl._create_default_https_context = ssl._create_unverified_context


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_extract_features(
    tmp_path, test_data_path, patch_h5_path, use_gpu, num_workers
):
    slide_path = os.path.join(test_data_path, "948176.svs")
    output_h5_path = tmp_path / "test.h5"
    output_pt_path = tmp_path / "test.pt"
    cfg = ExtractFeaturesConfig(
        slide_path=slide_path,
        patch_h5_path=patch_h5_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        num_workers=num_workers,
        model_type=ModelType.RESNET50,
        use_gpu=use_gpu,
    )
    mussel.cli.extract_features.main(OmegaConf.create(cfg))
    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_extract_features_two_step(
    tmp_path, test_data_path, patch_h5_path, use_gpu, num_workers
):
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
        num_workers=num_workers,
        model_type=ModelType.RESNET50,
        use_gpu=use_gpu,
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
@pytest.mark.timeout(600)
def test_extract_patch_features_direct(
    tmp_path, test_data_path, patch_h5_path, use_gpu, num_workers
):
    """Test direct call to extract_patch_features function."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    output_h5_path = tmp_path / "patch_features.h5"

    result = extract_patch_features(
        patch_h5_path=patch_h5_path,
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        model_type=ModelType.RESNET50,
        use_gpu=use_gpu,
        num_workers=num_workers,
    )

    assert os.path.exists(result)
    assert str(result) == str(output_h5_path)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_aggregate_slide_features_identity(
    tmp_path, test_data_path, patch_h5_path, use_gpu, num_workers
):
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
        use_gpu=use_gpu,
        num_workers=num_workers,
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
@pytest.mark.timeout(600)
def test_aggregate_slide_features_mean(
    tmp_path, test_data_path, patch_h5_path, use_gpu, num_workers
):
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
        use_gpu=use_gpu,
        num_workers=num_workers,
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


def test_auto_set_aggregation_method(tmp_path, use_gpu, num_workers):
    """Test that aggregation_method is automatically set to 'model' when slide_model_type is specified."""
    import unittest.mock as mock

    from mussel.utils import save_features

    slide_path = "tests/testdata/948176.svs"
    patch_h5_path = "tests/testdata/948176.patch.h5"
    output_h5_path = tmp_path / "test.h5"

    # Mock the actual feature extraction to avoid heavy computation
    with (
        mock.patch(
            "mussel.utils.feature_extract.extract_patch_features"
        ) as mock_extract,
        mock.patch(
            "mussel.utils.feature_extract.aggregate_slide_features"
        ) as mock_aggregate,
    ):

        # Call with slide_model_type - aggregation_method should be auto-set to "model"
        save_features(
            patch_h5_path=patch_h5_path,
            slide_path=slide_path,
            output_h5_path=output_h5_path,
            model_type=ModelType.GIGAPATH,  # Will be auto-set
            slide_model_type=ModelType.GIGAPATH_SLIDE,
            # aggregation_method should be auto-set to "model"
            use_gpu=use_gpu,
            num_workers=num_workers,
        )

        # Verify aggregate_slide_features was called with aggregation_method="model"
        mock_aggregate.assert_called_once()
        call_kwargs = mock_aggregate.call_args[1]
        assert call_kwargs["aggregation_method"] == "model"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
@pytest.mark.parametrize("embedding_precision", ["float16", "bfloat16"])
def test_extract_features_two_step_intermediate_stays_float32(
    tmp_path, test_data_path, patch_h5_path, use_gpu, num_workers, embedding_precision
):
    """E2E: in two-step mode the intermediate tile H5 must be float32 regardless of
    embedding_precision; only the final slide output is cast to the requested precision."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    output_h5_path = tmp_path / f"slide_{embedding_precision}.h5"
    output_pt_path = tmp_path / f"slide_{embedding_precision}.pt"
    intermediate_h5_path = tmp_path / f"intermediate_{embedding_precision}.patch.h5"

    cfg = ExtractFeaturesConfig(
        slide_path=slide_path,
        patch_h5_path=patch_h5_path,
        output_h5_path=str(output_h5_path),
        output_pt_path=str(output_pt_path),
        intermediate_h5_path=str(intermediate_h5_path),
        num_workers=num_workers,
        model_type=ModelType.RESNET50,
        use_gpu=use_gpu,
        aggregation_method="mean",
        embedding_precision=embedding_precision,
    )
    mussel.cli.extract_features.main(OmegaConf.create(cfg))

    assert output_h5_path.exists(), "Final slide H5 must exist"
    assert intermediate_h5_path.exists(), "Intermediate tile H5 must exist"

    # Intermediate tile features must be float32 (full precision for any downstream encoder)
    with h5py.File(intermediate_h5_path, "r") as f:
        assert f["features"].dtype == np.float32, (
            f"Intermediate tile features must be float32, got {f['features'].dtype}"
        )

    # Final slide output must be at the requested reduced precision (2 bytes)
    with h5py.File(output_h5_path, "r") as f:
        assert f["features"].dtype.itemsize == 2, (
            f"Final slide features should be 2-byte {embedding_precision}, "
            f"got {f['features'].dtype}"
        )

    # PT file dtype must also match
    pt_tensor = torch.load(output_pt_path, weights_only=True)
    expected_torch_dtype = torch.float16 if embedding_precision == "float16" else torch.bfloat16
    assert pt_tensor.dtype == expected_torch_dtype, (
        f"Expected .pt dtype {expected_torch_dtype}, got {pt_tensor.dtype}"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(300)
@pytest.mark.parametrize("embedding_precision", ["float16", "bfloat16"])
def test_extract_features_embedding_precision(
    tmp_path, test_data_path, patch_h5_path, use_gpu, num_workers, embedding_precision
):
    """E2E: extract_features with float16/bfloat16 writes reduced-precision outputs."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    output_h5_path = tmp_path / f"test_{embedding_precision}.h5"
    output_pt_path = tmp_path / f"test_{embedding_precision}.pt"

    cfg = ExtractFeaturesConfig(
        slide_path=slide_path,
        patch_h5_path=patch_h5_path,
        output_h5_path=str(output_h5_path),
        output_pt_path=str(output_pt_path),
        num_workers=num_workers,
        model_type=ModelType.RESNET50,
        use_gpu=use_gpu,
        embedding_precision=embedding_precision,
    )
    mussel.cli.extract_features.main(OmegaConf.create(cfg))

    assert output_h5_path.exists()
    assert output_pt_path.exists()

    with h5py.File(output_h5_path, "r") as f:
        assert "features" in f
        assert f["features"].shape[0] > 0
        # Both float16 and bfloat16 occupy 2 bytes per element.
        assert f["features"].dtype.itemsize == 2, (
            f"Expected 2-byte dtype for {embedding_precision}, "
            f"got {f['features'].dtype}"
        )

    pt_tensor = torch.load(output_pt_path, weights_only=True)
    assert pt_tensor.shape[0] > 0
    expected_torch_dtype = (
        torch.float16 if embedding_precision == "float16" else torch.bfloat16
    )
    assert pt_tensor.dtype == expected_torch_dtype, (
        f"Expected .pt tensor dtype {expected_torch_dtype}, got {pt_tensor.dtype}"
    )
