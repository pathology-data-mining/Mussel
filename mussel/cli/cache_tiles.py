"""
Inputs: slide_file_path, patch_file_path, annot_path
Results in .pt file with N_tiles x 3 x img_size x img_size tensor
"""

import argparse
import json
import time
from dataclasses import dataclass
from typing import Optional

import openslide
import pandas as pd
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch.utils.data import DataLoader

from mussel.datasets.h5 import Whole_Slide_Bag_FP
from mussel.utils.ml import collate_features


@dataclass
class CacheTilesConfig:
    slide_path: str = MISSING
    patch_path: str = MISSING
    output_path: str = MISSING
    limit_to_class: Optional[str] = None
    annot_path: Optional[str] = None
    cache_tile_indices_path: Optional[str] = None

cs = ConfigStore.instance()
cs.store(name="cache_tiles_config", node=CacheTilesConfig)

@hydra.main(config_path=".", config_name="cache_tiles_config", version_base=None)
def main(cfg: CacheTilesConfig):
    time_start = time.time()
    if limit_to_class is not None:
        limit_to_class = limit_to_class.replace("_", " ")
        annot = pd.read_csv(annot_path)
        annot['class'] = annot.idxmax(axis=1)
        indices = annot[annot['class'] == limit_to_class].index.tolist()
        print(f"limiting to class {limit_to_class} with {len(indices)} tiles")
    
    wsi = openslide.open_slide(cfg.slide_path)
    dataset = Whole_Slide_Bag_FP(
        file_path=cfg.patch_path,
        wsi=wsi,
        use_imagenet_rgb_dist=True,
        limit_to_indices=indices if limit_to_class else None,
    )
    kwargs = {"num_workers": 8, "pin_memory": True}
    loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        **kwargs,
        collate_fn=collate_features,
        shuffle=False,
    )
    with torch.no_grad():
        batch_list = []
        for count, (batch, coords) in enumerate(loader):
            if count % 100 == 0:
                print(
                    "batch {}/{}, {} files processed".format(
                        count, len(loader), count * 32
                    )
                )
            batch_list.append(batch)
        all_tiles = torch.cat(batch_list, dim=0)
    time_elapsed = time.time() - time_start
    print("\ncaching tiles for {} took {} s".format(cfg.output_path, time_elapsed))
    print(f"all_tiles shape: {all_tiles.shape}")
    torch.save(all_tiles, cfg.output_path)
    print(f"saved to {cfg.output_path}")
    # save indices as json
    if limit_to_class:
        with open(cache_tile_indices_path, "w") as f:
            json.dump(indices, f)
    else:
        with open(cache_tile_indices_path, "w") as f:
            json.dump(list(range(len(dataset))), f)


if __name__ == "__main__":
    main()
