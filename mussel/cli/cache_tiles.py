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
import pandas as pd
import tiffslide as openslide
import torch
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import MISSING
from torch.utils.data import DataLoader

from mussel.datasets.h5 import WholeSlideImageH5Dataset
from mussel.utils.ml import collate_features


@dataclass
class CacheTilesConfig:
    """
    slide_path (str): Path to the whole slide image.
    patch_h5_path (str): Path to the HDF5 file containing patch coordinates.
    output_pt_path (str): Path to save the cached tiles in PyTorch format.
    batch_size (int): Batch size for processing patches or tiles.
    num_workers (int): Number of worker threads for data loading.
    limit_to_class (Optional[List[str]]): Optional list of classes to limit the tiles to.
    annotation_csv_path (Optional[str]): Path to a CSV file containing annotations for filtering tiles.
    output_indices_json_path (Optional[str]): Path to save the indices of the tiles that were processed.
    """

    slide_path: str = MISSING
    patch_h5_path: str = MISSING
    output_pt_path: str = MISSING
    batch_size: int = 32
    num_workers: int = 16
    limit_to_class: Optional[List[str]] = None
    annotation_csv_path: Optional[str] = None
    output_indices_json_path: Optional[str] = None


desc_doc = """== ${hydra.help.app_name} ==

Caches tiles from a whole-slide image into a PyTorch tensor.
"""

parameter_doc = f"""
== Available Parameters ==
{CacheTilesConfig.__doc__}
"""

cs = ConfigStore.instance()
cs.store(name="cache_tiles_config", node=CacheTilesConfig)
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=desc_doc, footer=parameter_doc)),
    provider="hydra",
)


@hydra.main(config_path=".", config_name="cache_tiles_config", version_base=None)
def main(cfg: CacheTilesConfig):
    """Cache tiles from a whole slide image to a PyTorch tensor file."""
    time_start = time.time()
    indices = None
    if cfg.limit_to_class and cfg.annotation_csv_path:
        annot = pd.read_csv(cfg.annotation_csv_path)
        annot["class"] = annot.idxmax(axis=1)
        indices = annot[annot["class"].isin(cfg.limit_to_class)].index.tolist()
        logger.info(f"limiting to class {cfg.limit_to_class} with {len(indices)} tiles")

    dataset = WholeSlideImageH5Dataset(
        h5_path=cfg.patch_h5_path,
        slide_path=cfg.slide_path,
        use_imagenet_rgb_dist=True,
        limit_to_indices=indices if cfg.limit_to_class else None,
    )
    kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": True,
        "persistent_workers": cfg.num_workers > 0,
        "prefetch_factor": 2 if cfg.num_workers > 0 else None,
    }
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        **kwargs,
        collate_fn=collate_features,
        worker_init_fn=dataset.worker_init,
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
    logger.info(
        "\ncaching tiles for {} took {} s".format(cfg.output_pt_path, time_elapsed)
    )
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
