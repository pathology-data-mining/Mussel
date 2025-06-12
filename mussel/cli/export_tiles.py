"""export_tiles

CLI tool to export tiles as individual .png files from .patch.h5 file
"""

import functools
import os
import multiprocessing as mp
import time
from dataclasses import dataclass
from concurrent import futures

import h5py
import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import MISSING
from tqdm import tqdm

from mussel.utils.wsi import is_white_patch, is_black_patch


@dataclass
class ExportTilesConfig:
    slide_path: str = MISSING
    patch_h5_path: str = MISSING
    output_png_path: str = MISSING
    patch_size: int = 256
    mpp: float = 0.5
    num_workers: int = 16


cs = ConfigStore.instance()
cs.store(name="export_tiles_config", node=ExportTilesConfig)


def export_tile(
    tile_coords: np.array,
    wsi: WholeSlideImage,
    patch_size: int,
    output_path: str,
) -> None:
    """Utility function to export tile to .png file"""
    patch = wsi.read_region(tile_coords, 0, (patch_size, patch_size)).convert("RGB")
    if is_white_patch(np.array(patch)) or is_black_patch(np.array(patch)):
        return
    file_path = os.path.join(output_path, f"{tile_coords[0]}_{tile_coords[1]}.png")
    patch.save(file_path, "png")


@hydra.main(config_path=".", config_name="export_tiles_config", version_base=None)
def main(cfg: ExportTilesConfig):

    time_start = time.time()

    logger.info(f"Loading .patches.h5 file: {cfg.patch_h5_path}")

    with h5py.File(cfg.patch_h5_path, "r") as patches_h5:
        tile_coords = np.array(patches_h5["coords"])

    # Init whole slide image
    wsi = tiffslide.TiffSlide(cfg.slide_path)
    slide_mpp = float(wsi.properties[tiffslide.PROPERTY_NAME_MPP_X])

    native_patch_size = WSI_object.get_native_size(cfg.patch_size, cfg.mpp, slide_mpp)
    logger.info(
        f"Exporting approx. {len(tile_coords)} tiles as .png files to {cfg.output_png_path}"
    )
    n_tiles = len(tile_coords)

    partial = functools.partial(
        export_tile,
        wsi_object=wsi,
        patch_size=native_patch_size,
        output_path=cfg.output_png_path,
    )

    with futures.ThreadPoolExecutor(cfg.num_workers) as executor:
        list(tqdm(executor.map(partial, tile_coords), total=n_tiles))

    time_elapsed = time.time() - time_start
    logger.info(f"Exporting tiles for {cfg.patch_h5_path} took {time_elapsed} seconds")
    logger.info(f"Estimated time per tile: {time_elapsed/n_tiles} seconds")


if __name__ == "__main__":
    main()
