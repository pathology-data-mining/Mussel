import os
from omegaconf import OmegaConf

import mussel.cli.cache_tiles
from mussel.cli.cache_tiles import CacheTilesConfig

def test_cache_tiles(tmp_path):
    annotation_classes = [
        "carcinoma in situ",
        "invasive carcinoma",
        "collagenous stroma",
        "adipose",
        "vessel",
        "necrosis",
        "invasive adenocarcinoma",
        "sarcoma"]
    output_indices_json = tmp_path / "test.json"
    output_pt_path = tmp_path / "test.pt"
    cfg = CacheTilesConfig(
        limit_to_class=annotation_classes,
        num_workers=1,
        patch_h5_path="tests/testdata/948176.patch.h5",
        slide_path="tests/testdata/948176.svs",
        annotation_csv_path="tests/testdata/948176.annotation.csv",
        output_indices_json_path=output_indices_json,
        output_pt_path=output_pt_path,
    )
    mussel.cli.cache_tiles.main(cfg)
    assert os.path.exists(output_indices_json)
    assert os.path.exists(output_pt_path)

