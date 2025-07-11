import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import tiffslide
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import MISSING, OmegaConf

import hydra
from mussel.utils.segment import draw_slide_mask, save_patches_png, segment_tissue


@dataclass
class SegConfig:
    """
    patch_size (int): Patch size at specified mpp (microns per pixel).
    step_size (int): Optional step size. Defaults to the patch size.
    mpp (float): Desired microns per pixel
    seg_level (int): tessellation pyramid level. If negative, use best level for factor=64 downsample.
    segment_threshold (int): pixel threshold value . If pixel value smaller than or equal to threshold, it is set to 0, otherwise it is set to the maximum value (segment_max_value).
    segment_max_value (int): maximum pixel value.
    median_blur_ksize (int): aperture linear size. it must be odd and greater than 1. image is blurred with median filter.
    morphology_ex_kernel (int): kernel for mophological closing transformation.
    ref_patch_size (int): reference patch size to use for tissue area and hole area thresholding.
    use_otsu (bool): apply otsu thresholding
    tissue_area_threshold (int): tissue area threshold. Foreground contour area needs to exceed this threshold (scaled by reference patch size) to be included as foreground.
    hole_area_threshold (int): holea area threshold. Hole contour area needs to exceed this threshold (scaled by reference patch size) to be included as a hole.
    max_num_holes (int): maximum number of holes.
    keep_ids (List[int]): list of contour IDs to keep
    exclude_ids (List[int]): list of contour IDs to exclude
    """

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
class BiopsySegConfig(SegConfig):
    segment_threshold: int = 15
    median_blur_ksize: int = 11
    morphology_ex_kernel: int = 2
    tissue_area_threshold: int = 1
    hole_area_threshold: int = 1
    max_num_holes: int = 2


@dataclass
class ResectionSegConfig(SegConfig):
    segment_threshold: int = 15
    median_blur_ksize: int = 11
    morphology_ex_kernel: int = 4
    tissue_area_threshold: int = 100
    hole_area_threshold: int = 16
    max_num_holes: int = 8


@dataclass
class TcgaSegConfig(SegConfig):
    segment_threshold: int = 8
    median_blur_ksize: int = 7
    morphology_ex_kernel: int = 4
    tissue_area_threshold: int = 16
    hole_area_threshold: int = 4
    max_num_holes: int = 8


@dataclass
class VisConfig:
    """
    vis_level (int): Image pyramid level. If negative, use best level for factor=64 downsample.
    outline (Any): outline color
    fill (Any): fill color
    custom_downsample (Optional[int]): custom downsample
    """

    vis_level: int = -1
    outline = "black"
    fill = (255, 0, 0, 80)
    custom_downsample: Optional[int] = None


@dataclass
class PngConfig:
    """
    filter_black_white (bool): filter black/white tiles
    white_threshold (int): threshold for white tile
    black_threshold (int): threshold for black tile
    """

    filter_black_white: bool = True
    white_threshold: int = 15
    black_threshold: int = 50


defaults = ["_self_", {"seg_config": "default"}]


@dataclass
class TessellateConfig:
    """
    slide_path (str): Path to slide (.svs,.tiff)
    output_h5_path (str): Path to output h5 file
    output_png_dir (Optional[str]): Optional path to output png directory
    output_mask_path (Optional[str]): Optional path to output tissue mask
    output_grid_mask_path (Optional[str]): Optional path to output tissue grid mask
    output_thumbnail_path (Optional[str]): Optional path to output tissue thumbnail
    thumbnail_size (tuple): Thumbnail size (row, col) tuple
    num_workers (int): Number of workers for saving patches
    """

    defaults: List[Any] = field(default_factory=lambda: defaults)
    slide_path: str = MISSING
    output_h5_path: str = MISSING
    output_png_dir: Optional[str] = None
    output_mask_path: Optional[str] = None
    output_grid_mask_path: Optional[str] = None
    output_thumbnail_path: Optional[str] = None
    thumbnail_size: tuple = (1024, 1024)
    num_workers: int = 4
    seg_config: SegConfig = MISSING
    vis_config: VisConfig = field(default_factory=VisConfig)
    png_config: PngConfig = field(default_factory=PngConfig)


docs = f"""
== Available Parameters ==
{TessellateConfig.__doc__}
seg_config: {SegConfig.__doc__}
vis_config: {VisConfig.__doc__}
png_config: {PngConfig.__doc__}

== Description ==
tessellate tiles a whole-slide image.  The tile coordinates are written to an HDF5 (.h5)
file for use in downstream processing, such as feature extraction.
"""


@dataclass
class MyHelpConf(HelpConf):
    footer: str = docs


@dataclass
class MyHydraConf(HydraConf):
    help: HelpConf = field(default_factory=MyHelpConf)


cs = ConfigStore.instance()
cs.store(group="hydra", name="config", node=MyHydraConf(), provider="hydra")
cs.store(group="seg_config", name="default", node=SegConfig)
cs.store(group="seg_config", name="biopsy", node=BiopsySegConfig)
cs.store(group="seg_config", name="resection", node=ResectionSegConfig)
cs.store(group="seg_config", name="tcga", node=TcgaSegConfig)
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
