"""export_tiles

CLI tool to export tiles as individual .png files from .patch.h5 file
"""

import functools
import multiprocessing as mp
import os
import time
from concurrent import futures
from dataclasses import dataclass

import h5py
import hydra
import numpy as np
import tiffslide
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import MISSING
from tqdm import tqdm

from mussel.utils import export_tiles


@dataclass
class ExportTilesConfig:
    """
    slide_path (str): Path to the whole slide image.
    patch_h5_path (str): Path to the HDF5 file containing tile coordinates.
    output_png_path (str): Path to save the exported tiles as .png files.
    patch_size (int): Size of the patches to export (in pixels).
    mpp (float): Microns per pixel of the slide.
    num_workers (int): Number of worker threads to use for exporting tiles.
    """

    slide_path: str = MISSING
    patch_h5_path: str = MISSING
    output_png_path: str = MISSING
    patch_size: int = 256
    mpp: float = 0.5
    num_workers: int = 16


desc_doc = """== ${hydra.help.app_name} ==
Exports tiles from a slide as individual .png files using a HDF5 tile coordinate manifest.
"""

parameter_doc = f"""
== Available Parameters ==
{ExportTilesConfig.__doc__}
"""

cs = ConfigStore.instance()
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=desc_doc, footer=parameter_doc)),
    provider="hydra",
)
cs.store(name="export_tiles_config", node=ExportTilesConfig)


@hydra.main(config_path=".", config_name="export_tiles_config", version_base=None)
def main(cfg: ExportTilesConfig):
    """Export tiles from a whole slide image to individual PNG files."""
    export_tiles(
        patch_h5_path=cfg.patch_h5_path,
        slide_path=cfg.slide_path,
        output_png_path=cfg.output_png_path,
        patch_size=cfg.patch_size,
        mpp=cfg.mpp,
        num_workers=cfg.num_workers,
    )


if __name__ == "__main__":
    main()
