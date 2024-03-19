import hydra
from omegaconf import DictConfig, OmegaConf
import os
import time
from loguru import logger
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from mussel.WholeSlideImage import WholeSlideImage
from mussel.utils.wsi import StitchCoords
from mussel.utils.timer import timed

@dataclass
class PatchConfig:
    use_padding: bool = True
    patch_size: int = 256 #y
    step_size: int = 256 #y
    mpp: float = 0.5 #y

@dataclass
class VisConfig:
    vis_level: int = -1
    line_thickness: int = 100

@dataclass
class SegConfig:
    seg_level: int = -1
    segment_threshold: int = 20
    segment_max_value: int = 255
    median_blur_ksize: int = 7
    morphology_ex_kernel: int = 0
    ref_patch_size: int = 512
    use_otsu: bool = False
    keep_ids: List[int] = field(default_factory=list)
    exclude_ids: List[int] = field(default_factory=list)

@dataclass
class FilterParams:
    tissue_area_threshold: int = 100 #y
    hole_area_threshold: int = 16
    max_num_holes: int = 8

@dataclass
class TessellateConfig:
    slide_path: str
    output_path: str
    stitch_path: Optional[str] = None
    mask_path: Optional[str] = None
    seg_config: SegConfig = field(default_factory=SegConfig)
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    vis_config: VisConfig = field(default_factory=VisConfig)
    patch_config: PatchConfig = field(default_factory=PatchConfig)

cs = ConfigStore.instance()
cs.store(name="base_config", node=TessellateConfig)

@hydra.main(version_base=None, config_path=".", config_name="tessellate_config")
def seg_and_patch(
    cfg: TessellateConfig,
):
    # Inialize WSI
    WSI_object = WholeSlideImage(cfg.slide_path)

    if cfg.vis_config.vis_level < 0:
        if len(WSI_object.level_dim) == 1:
            cfg.vis_config.vis_level = 0

        else:
            wsi = WSI_object.getOpenSlide()
            best_level = wsi.get_best_level_for_downsample(64)
            cfg.vis_config.vis_level = best_level

    if cfg.seg_config.seg_level < 0:
        if len(WSI_object.level_dim) == 1:
            cfg.seg_config.seg_level = 0

        else:
            wsi = WSI_object.getOpenSlide()
            best_level = wsi.get_best_level_for_downsample(64)
            cfg.seg_config.seg_level = best_level


    w, h = WSI_object.level_dim[cfg.seg_config.seg_level]
    if w * h > 1e8:
        logger.error(
            "level_dim {} x {} is likely too large for successful segmentation, aborting".format(
                w, h
            )
        )
        return

    seg_time_elapsed = -1
    WSI_object.segmentTissue(**OmegaConf.to_container(cfg.seg_config),
                                **OmegaConf.to_container(cfg.filter_config))

    if mask_path:
        mask = WSI_object.visWSI(**OmegaConf.to_container(cfg.vis_config))
        mask.save(cfg.mask_path)

    WSI_object.process_contours(
        save_path=cfg.output_path,
        **OmegaConf.to_container(cfg.patch_config))

    if cfg.stitch_path:
        heatmap = StitchCoords(
            cfg.output_path,
            WSI_object,
            downscale=64,
            bg_color=(0, 0, 0),
            alpha=-1,
            draw_grid=False,
        )
        heatmap.save(cfg.stitch_path)


if __name__ == "__main__":
    seg_and_patch()
