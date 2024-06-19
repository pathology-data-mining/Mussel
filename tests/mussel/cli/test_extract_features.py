import os
from omegaconf import OmegaConf

import mussel.cli.extract_features
from mussel.cli.extract_features import ExtractFeaturesConfig


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
    )
    mussel.cli.extract_features.main(OmegaConf.create(cfg))
    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)

