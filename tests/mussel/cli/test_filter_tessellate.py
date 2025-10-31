import os
from pathlib import Path
from omegaconf import OmegaConf
import h5py
import torch

from mussel.cli.filter_tessellate import FilterTessellateConfig, SegConfig, main


def test_filter_tessellate(tmp_path):
    """Test the integrated filter-tessellate workflow."""
    slide_path = "tests/testdata/948176.svs"
    model_path = "tests/testdata/ctranspath.pth"  # This would need to be a real model
    classifier_pkl = "tests/testdata/simple_classifier.pkl"
    output_h5_path = os.path.join(tmp_path, "filtered.h5")
    output_pt_path = os.path.join(tmp_path, "filtered.pt")

    seg_config = SegConfig(segment_threshold=0)
    cfg = FilterTessellateConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=classifier_pkl,
        classifier_threshold=0.75,
        model_path=model_path,
        seg_config=seg_config,
        num_workers=1,
        batch_size=32,
        use_gpu=False,  # Use CPU for testing
        keep_intermediate_files=False,
        save_features_to_h5=True,
    )

    # Note: This test would require:
    # 1. A CTRANSPATH model file at model_path
    # 2. The actual model weights
    # For now, we're creating a skeleton test that would work if those were available
    
    # Uncomment the following when CTRANSPATH model is available:
    # main(OmegaConf.create(cfg))
    # 
    # assert os.path.exists(output_h5_path)
    # assert os.path.exists(output_pt_path)
    # 
    # # Check that the output files are not empty
    # with h5py.File(output_h5_path, "r") as f:
    #     assert "coords" in f
    #     assert f["coords"].shape[0] > 0
    # 
    # data = torch.load(output_pt_path, weights_only=True)
    # assert data.shape[0] > 0


def test_filter_tessellate_with_intermediate_files(tmp_path):
    """Test the integrated workflow while keeping intermediate files."""
    slide_path = "tests/testdata/948176.svs"
    model_path = "tests/testdata/ctranspath.pth"
    classifier_pkl = "tests/testdata/simple_classifier.pkl"
    output_h5_path = os.path.join(tmp_path, "filtered.h5")
    output_pt_path = os.path.join(tmp_path, "filtered.pt")

    seg_config = SegConfig(segment_threshold=0)
    cfg = FilterTessellateConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=classifier_pkl,
        classifier_threshold=0.75,
        model_path=model_path,
        seg_config=seg_config,
        num_workers=1,
        batch_size=32,
        use_gpu=False,
        keep_intermediate_files=True,  # Keep intermediate files
        save_features_to_h5=True,
    )

    # Note: This test would require:
    # 1. A CTRANSPATH model file at model_path
    # For now, we're creating a skeleton test
    
    # Uncomment when model is available:
    # main(OmegaConf.create(cfg))
    # 
    # assert os.path.exists(output_h5_path)
    # assert os.path.exists(output_pt_path)
    # 
    # # Check that intermediate files were created
    # base_path = Path(output_h5_path).parent
    # tessellate_h5_path = base_path / f"{Path(slide_path).stem}.tessellate.h5"
    # features_h5_path = base_path / f"{Path(slide_path).stem}.features.h5"
    # 
    # assert os.path.exists(tessellate_h5_path)
    # assert os.path.exists(features_h5_path)
