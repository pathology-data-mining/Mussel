import json
import os
from pathlib import Path

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

import mussel.cli.cache_tiles
from mussel.cli.cache_tiles import CacheTilesConfig
from mussel.datasets import WholeSlideImageH5Dataset


@pytest.mark.slow
@pytest.mark.integration
def test_cache_tiles(tmp_path, test_data_path, num_workers):
    # A subset of the annotation's classes, so the class filter drops tiles: 135
    # of the 1294 annotated tiles have one of these as their most likely class.
    # (The full list kept every tile, so the filter was never exercised.)
    annotation_classes = ["necrosis", "vessel", "invasive adenocarcinoma"]
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

    # The tiles kept are those whose most likely annotated class is in the list.
    annot = pd.read_csv(cfg.annotation_csv_path)
    expected = annot.index[annot.idxmax(axis=1).isin(annotation_classes)].tolist()
    assert 0 < len(expected) < len(annot)
    assert json.loads(output_indices_json.read_text()) == expected
    tiles = torch.load(output_pt_path, weights_only=True)
    assert tiles.shape[0] == len(expected)
    # ...cached in that order: the first and last are those tiles' images.
    dataset = WholeSlideImageH5Dataset(h5_path=cfg.patch_h5_path, slide_path=cfg.slide_path,
                                       use_imagenet_rgb_dist=True, init_wsi_in_worker=False)
    for k in (0, len(expected) - 1):
        img, _ = dataset[expected[k]]
        # __getitem__ returns a (1, C, H, W) tile; collate_features concatenates them.
        torch.testing.assert_close(tiles[k : k + 1], img, msg=f"cached tile {k} is not tile {expected[k]}")
