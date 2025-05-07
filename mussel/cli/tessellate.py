import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import hydra
import numpy as np
import pandas as pd
import tiffslide
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import MISSING, OmegaConf

from mussel.utils.segment import draw_slide_mask, save_patches_png, segment_tissue


@dataclass
class SegConfig:
    patch_size: int = 256
    step_size: Optional[int] = None  # if None, defaults to patch_size
    mpp: float = 0.5
    seg_level: int = -1
    segment_threshold: int = 20
    segment_max_value: int = 255
    median_blur_ksize: int = 7
    morphology_ex_kernel: int = 0
    ref_patch_size: int = 512
    use_otsu: bool = False
    tissue_area_threshold: int = 100
    hole_area_threshold: int = 16
    max_num_holes: int = 8
    keep_ids: List[int] = field(default_factory=list)
    exclude_ids: List[int] = field(default_factory=list)


@dataclass
class VisConfig:
    vis_level: int = -1
    outline = "black"
    fill = (255, 0, 0, 80)
    custom_downsample: Optional[int] = None


@dataclass
class PngConfig:
    filter_black_white: bool = True
    white_threshold: int = 15
    black_threshold: int = 50


@dataclass
class TessellateConfig:
    slide_path: str = MISSING
    output_h5_path: str = MISSING
    output_png_dir: Optional[str] = None
    output_mask_path: Optional[str] = None
    output_grid_mask_path: Optional[str] = None
    output_thumbnail_path: Optional[str] = None
    thumbnail_size: tuple = (1024, 1024)
    num_workers: int = 4
    seg_config: SegConfig = field(default_factory=SegConfig)
    vis_config: VisConfig = field(default_factory=VisConfig)
    png_config: PngConfig = field(default_factory=PngConfig)


cs = ConfigStore.instance()
cs.store(name="tessellate_config", node=TessellateConfig)


@hydra.main(version_base=None, config_path=".", config_name="tessellate_config")
def main(
    cfg: TessellateConfig,
):
    # Inialize WSI
    slide_id = os.path.splitext(os.path.basename(cfg.slide_path))[0]
    wsi = tiffslide.open_slide(cfg.slide_path)

    if cfg.vis_config.vis_level < 0:
        if len(wsi.level_dimensions) == 1:
            cfg.vis_config.vis_level = 0
        else:
            cfg.vis_config.vis_level = wsi.get_best_level_for_downsample(64)

    if cfg.seg_config.seg_level < 0:
        if len(wsi.level_dimensions) == 1:
            cfg.seg_config.seg_level = 0
        else:
            cfg.seg_config.seg_level = wsi.get_best_level_for_downsample(64)

    w, h = wsi.level_dimensions[cfg.seg_config.seg_level]
    if w * h > 1e12:
        logger.error(
            "level_dim {} x {} is likely too large for successful segmentation, aborting".format(
                w, h
            )
        )
        return

    polygon, grid, coords = segment_tissue(
        wsi,
        slide_id=slide_id,
        output_h5_path=cfg.output_h5_path,
        **OmegaConf.to_container(cfg.seg_config),
    )

    if cfg.output_mask_path:
        mask = draw_slide_mask(
            wsi,
            polygon,
            **OmegaConf.to_container(cfg.vis_config),
        )
        mask.save(cfg.output_mask_path)

    if cfg.output_grid_mask_path:
        grid_mask = draw_slide_mask(
            wsi,
            grid,
            **OmegaConf.to_container(cfg.vis_config),
        )
        grid_mask.save(cfg.output_grid_mask_path)

    if cfg.output_png_dir:
        logger.info(f"saving patches to {cfg.output_png_dir}")
        save_patches_png(
            wsi,
            coords,
            save_dir=cfg.output_png_dir,
            num_workers=cfg.num_workers,
            patch_size=cfg.seg_config.patch_size,
            filter_black_white=cfg.png_config.filter_black_white,
            white_threshold=cfg.png_config.white_threshold,
            black_threshold=cfg.png_config.black_threshold,
        )

    if cfg.output_thumbnail_path:
        logger.info(f"saving thumbnail to {cfg.output_thumbnail_path}")
        thumbnail = wsi.get_thumbnail(cfg.thumbnail_size)
        with open(cfg.output_thumbnail_path, "wb") as f:
            thumbnail.save(f)

    wsi.close()


if __name__ == "__main__":
    main()
