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
    seg_level (int): Tessellation pyramid level. If negative, use best level for factor=64 downsample.
    segment_threshold (int): Pixel threshold value . If pixel value smaller than or equal to threshold, it is set to 0, otherwise it is set to the maximum value (segment_max_value).
    segment_max_value (int): Maximum pixel value.
    median_blur_ksize (int): Aperture linear size. it must be odd and greater than 1. image is blurred with median filter.
    morphology_ex_kernel (int): Kernel for mophological closing transformation.
    ref_patch_size (int): Reference patch size to use for tissue area and hole area thresholding.
    use_otsu (bool): If True, apply otsu thresholding.
    tissue_area_threshold (int): Tissue area threshold. Foreground contour area needs to exceed this threshold (scaled by reference patch size) to be included as foreground.
    hole_area_threshold (int): Hole area threshold. Hole contour area needs to exceed this threshold (scaled by reference patch size) to be included as a hole.
    max_num_holes (int): Maximum number of holes.
    keep_ids (List[int]): List of contour IDs to keep.
    exclude_ids (List[int]): List of contour IDs to exclude.
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
    vis_level (int): pyramid level to visualize. If negative, use best level for factor=64 downsample.
    outline (str): color of the outline of the tissue mask.
    fill (tuple): RGBA color of the filled tissue mask.
    custom_downsample (Optional[int]): custom downsample factor for visualization. If None, use the default downsample factor.
    """

    vis_level: int = -1
    outline = "black"
    fill = (255, 0, 0, 80)
    custom_downsample: Optional[int] = None


@dataclass
class PngConfig:
    """
    filter_black_white (bool): If True, filter out black and white patches.
    white_threshold (int): Threshold for white patches.
    black_threshold (int): Threshold for black patches.
    """

    filter_black_white: bool = True
    white_threshold: int = 15
    black_threshold: int = 50


defaults = ["_self_", {"seg_config": "default"}]


@dataclass
class TessellateConfig:
    """
    slide_path (str): Path to the whole-slide image.
    output_h5_path (str): Path to save the HDF5 file with tile coordinates.
    output_png_dir (Optional[str]): Directory to save patches as PNG files.
    output_mask_path (Optional[str]): Path to save the mask image.
    output_grid_mask_path (Optional[str]): Path to save the grid mask image.
    output_thumbnail_path (Optional[str]): Path to save the thumbnail image.
    thumbnail_size (tuple): Size of the thumbnail image.
    seg_config (SegConfig): Configuration for segmentation parameters.
    vis_config (VisConfig): Configuration for visualization parameters.
    png_config (PngConfig): Configuration for PNG saving parameters.
    num_workers (int): Number of workers for saving patches.
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


desc_doc = """== ${hydra.help.app_name} ==

tessellate tiles a whole-slide image.  The tile coordinates are written to an HDF5 (.h5)
file for use in downstream processing, such as feature extraction.
"""

parameter_doc = f"""
== Available Parameters ==
{TessellateConfig.__doc__}
seg_config: {SegConfig.__doc__}
vis_config: {VisConfig.__doc__}
png_config: {PngConfig.__doc__}

"""

cs = ConfigStore.instance()
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=desc_doc, footer=parameter_doc)),
    provider="hydra",
)
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

    if values := segment_tissue(
        wsi,
        slide_id=slide_id,
        output_h5_path=cfg.output_h5_path,
        **OmegaConf.to_container(cfg.seg_config),
        ):
        polygon, grid, coords = values
    else:
        return

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
