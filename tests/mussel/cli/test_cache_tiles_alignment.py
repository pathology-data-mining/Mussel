"""cache_tiles: the indices JSON is row-aligned with the cached tensor.

Tiles that fail to load are dropped from the tensor, so they must be dropped
from the indices too; a class filter matching no tile caches nothing (it used to
cache every tile).
"""
import json
from pathlib import Path
from unittest.mock import patch

import h5py
import pandas as pd
import torch

import mussel.cli.cache_tiles
import mussel.datasets.h5 as h5mod
from mussel.cli.cache_tiles import CacheTilesConfig
from mussel.datasets import WholeSlideImageH5Dataset

_TESTDATA = Path(__file__).parent.parent.parent / "testdata"
_SLIDE = str(_TESTDATA / "948176.svs")
_H5 = str(_TESTDATA / "948176.patch.h5")


def _inputs(tmp_path, n=6):
    with h5py.File(_H5) as f:
        coords, attrs = f["coords"][:n], dict(f["coords"].attrs)
    patch_h5 = tmp_path / "s.patch.h5"
    with h5py.File(patch_h5, "w") as f:
        ds = f.create_dataset("coords", data=coords)
        for k, v in attrs.items():
            ds.attrs[k] = v
    # Tiles 1, 2, 4 and 5 are "tumor"; 0 and 3 "stroma".
    annot = pd.DataFrame({"tumor": [0.1, 0.9, 0.8, 0.2, 0.7, 0.6], "stroma": [0.9, 0.1, 0.2, 0.8, 0.3, 0.4]})
    csv = tmp_path / "annot.csv"
    annot.to_csv(csv, index=False)
    return str(patch_h5), str(csv)


def _cfg(tmp_path, classes):
    patch_h5, csv = _inputs(tmp_path)
    return CacheTilesConfig(slide_path=_SLIDE, patch_h5_path=patch_h5, annotation_csv_path=csv,
                            limit_to_class=classes, num_workers=0, batch_size=2,
                            output_pt_path=str(tmp_path / "tiles.pt"),
                            output_indices_json_path=str(tmp_path / "indices.json"))


class _FailingTiles:
    """A slide whose reads at the given level-0 x coordinates fail (corrupt tiles)."""

    def __init__(self, inner, bad_x):
        self.inner, self.bad_x = inner, set(bad_x)

    def read_region(self, location, *args):
        if int(location[0]) in self.bad_x:
            raise ValueError("imagecodecs: corrupt tile")
        return self.inner.read_region(location, *args)


def test_failed_tiles_are_dropped_from_the_indices(tmp_path):
    cfg = _cfg(tmp_path, ["tumor"])
    with h5py.File(cfg.patch_h5_path) as f:
        bad_x = [int(f["coords"][2][0])]
    real_open = h5mod.openslide.open_slide
    with patch.object(h5mod.openslide, "open_slide", side_effect=lambda p: _FailingTiles(real_open(p), bad_x)):
        mussel.cli.cache_tiles.main(cfg)

    indices = json.loads(Path(cfg.output_indices_json_path).read_text())
    tiles = torch.load(cfg.output_pt_path, weights_only=True)
    assert indices == [1, 4, 5]
    assert tiles.shape[0] == len(indices)
    ref = WholeSlideImageH5Dataset(h5_path=cfg.patch_h5_path, slide_path=_SLIDE, use_imagenet_rgb_dist=True,
                                   init_wsi_in_worker=False)
    for row, idx in enumerate(indices):
        torch.testing.assert_close(tiles[row : row + 1], ref[idx][0], msg=f"row {row} is not tile {idx}")


def test_class_filter_matching_no_tiles_caches_nothing(tmp_path):
    cfg = _cfg(tmp_path, ["necrosis"])
    mussel.cli.cache_tiles.main(cfg)
    assert json.loads(Path(cfg.output_indices_json_path).read_text()) == []
    assert torch.load(cfg.output_pt_path, weights_only=True).shape[0] == 0


def test_without_class_filter_every_tile_is_indexed(tmp_path):
    cfg = _cfg(tmp_path, None)
    mussel.cli.cache_tiles.main(cfg)
    assert json.loads(Path(cfg.output_indices_json_path).read_text()) == list(range(6))
    assert torch.load(cfg.output_pt_path, weights_only=True).shape[0] == 6
