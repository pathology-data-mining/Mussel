import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import mussel.cli.cache_tiles
from mussel.cli.cache_tiles import CacheTilesConfig


@pytest.mark.slow
@pytest.mark.integration
def test_cache_tiles(tmp_path, test_data_path, num_workers):
    annotation_classes = [
        "carcinoma in situ",
        "invasive carcinoma",
        "collagenous stroma",
        "adipose",
        "vessel",
        "necrosis",
        "invasive adenocarcinoma",
        "sarcoma",
    ]
    output_indices_json = tmp_path / "test.json"
    output_pt_path = tmp_path / "test.pt"
    cfg = CacheTilesConfig(
        limit_to_class=annotation_classes,
        num_workers=num_workers,
        patch_h5_path=os.path.join(test_data_path, "948176.patch.h5"),
        slide_path=os.path.join(test_data_path, "948176.svs"),
        annotation_csv_path=os.path.join(test_data_path, "948176.annotation.csv"),
        output_indices_json_path=output_indices_json,
        output_pt_path=output_pt_path,
    )
    mussel.cli.cache_tiles.main(cfg)
    assert os.path.exists(output_indices_json)
    assert os.path.exists(output_pt_path)
