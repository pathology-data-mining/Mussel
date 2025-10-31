import os
from pathlib import Path
from omegaconf import OmegaConf
import h5py
import torch

from mussel.cli.tessellate_extract_features import TessellateExtractFeaturesConfig, main
from mussel.cli.tessellate import SegConfig
from mussel.models import ModelType


def test_tessellate_extract_features(tmp_path):
    """Test the integrated tessellate-extract-features workflow."""
    slide_path = "tests/testdata/948176.svs"
    output_h5_path = os.path.join(tmp_path, "features.h5")
    output_pt_path = os.path.join(tmp_path, "features.pt")

    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateExtractFeaturesConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        model_type=ModelType.RESNET50,  # Use ResNet50 for testing (no model_path needed)
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


def test_tessellate_extract_features_with_intermediate_files(tmp_path):
    """Test the integrated workflow while keeping intermediate files."""
    slide_path = "tests/testdata/948176.svs"
    output_h5_path = os.path.join(tmp_path, "features.h5")
    output_pt_path = os.path.join(tmp_path, "features.pt")

    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateExtractFeaturesConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        model_type=ModelType.RESNET50,  # Use ResNet50 for testing (no model_path needed)
        seg_config=seg_config,
        num_workers=1,
        batch_size=32,
        use_gpu=False,
        keep_intermediate_files=True,  # Keep intermediate files
    )

    main(OmegaConf.create(cfg))

    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)

    # Check that intermediate tessellation file was created
    base_path = Path(output_h5_path).parent
    tessellate_h5_path = base_path / f"{Path(slide_path).stem}.tessellate.h5"
    
    assert os.path.exists(tessellate_h5_path)


def test_tessellate_extract_features_with_visualizations(tmp_path):
    """Test the workflow with optional visualization outputs."""
    slide_path = "tests/testdata/948176.svs"
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
        output_mask_path=output_mask_path,
        output_grid_mask_path=output_grid_mask_path,
        output_thumbnail_path=output_thumbnail_path,
        model_type=ModelType.RESNET50,
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
