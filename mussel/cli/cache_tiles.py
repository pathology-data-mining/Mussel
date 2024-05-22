"""
Inputs: slide_file_path, patch_file_path, annotation_csv_path
Results in .pt file with N_tiles x 3 x img_size x img_size tensor
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

import hydra
import tiffslide as openslide
import pandas as pd
import torch
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import MISSING
from torch.utils.data import DataLoader

from mussel.datasets.h5 import Whole_Slide_Bag_FP
from mussel.utils.ml import collate_features


@dataclass
class CacheTilesConfig:
    slide_path: str = MISSING
    patch_h5_path: str = MISSING
    output_pt_path: str = MISSING
    batch_size: int = 32
    num_workers: int = 16
    limit_to_class: Optional[List[str]] = None
    annotation_csv_path: Optional[str] = None
    output_indices_json_path: Optional[str] = None

cs = ConfigStore.instance()
cs.store(name="cache_tiles_config", node=CacheTilesConfig)

@hydra.main(config_path=".", config_name="cache_tiles_config", version_base=None)
def main(cfg: CacheTilesConfig):
    time_start = time.time()
    indices = None
    if cfg.limit_to_class and cfg.annotation_csv_path:
        annot = pd.read_csv(cfg.annotation_csv_path)
        annot['class'] = annot.idxmax(axis=1)
        indices = annot[annot['class'].isin(cfg.limit_to_class)].index.tolist()
        logger.info(f"limiting to class {cfg.limit_to_class} with {len(indices)} tiles")
    
    wsi = openslide.open_slide(cfg.slide_path)
    dataset = Whole_Slide_Bag_FP(
        file_path=cfg.patch_h5_path,
        wsi=wsi,
        use_imagenet_rgb_dist=True,
        limit_to_indices=indices if cfg.limit_to_class else None,
    )
    kwargs = {"num_workers": cfg.num_workers, "pin_memory": True}
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        **kwargs,
        collate_fn=collate_features,
        shuffle=False,
    )
    with torch.no_grad():
        batch_list = []
        for count, (batch, coords) in enumerate(loader):
            if count % 100 == 0:
                logger.info(
                    "batch {}/{}, {} files processed".format(
                        count, len(loader), count * 32
                    )
                )
            batch_list.append(batch)
        all_tiles = torch.cat(batch_list, dim=0)
    time_elapsed = time.time() - time_start
    logger.info("\ncaching tiles for {} took {} s".format(cfg.output_pt_path, time_elapsed))
    logger.info(f"all_tiles shape: {all_tiles.shape}")
    torch.save(all_tiles, cfg.output_pt_path)
    logger.info(f"saved to {cfg.output_pt_path}")
    # save indices as json
    if cfg.output_indices_json_path:
        with open(cfg.output_indices_json_path, "w") as f:
            if cfg.limit_to_class:
                json.dump(indices, f)
            else:
                json.dump(list(range(len(dataset))), f)


if __name__ == "__main__":
    main()
