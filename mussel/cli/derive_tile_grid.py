"""Derive a coarser tile grid from an existing tessellation (mussel.utils.tile_grid).

Every ``patch_size`` px cell, aligned to the slide origin, that contains the centre
of a source tile, at the source tiles' resolution. For a slide encoder that needs
larger tiles than the patch encoders it is run with, e.g. TITAN's 512 px grid from
224 px tiles, so it reads the same tissue without a second segmentation.

Usage:
  derive_tile_grid source_h5_path=<id>.patch.h5 output_h5_path=<id>.512px.patch.h5 patch_size=512
  derive_tile_grid 'source_h5_paths=[a.patch.h5,b.patch.h5]' 'output_h5_paths=[a.512.h5,b.512.h5]' patch_size=512
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

import hydra
from hydra.core.config_store import ConfigStore

from mussel.utils.tile_grid import derive_patch_h5

logger = logging.getLogger(__name__)


@dataclass
class DeriveTileGridConfig:
    """
    source_h5_path (str): Patch HDF5 of the source tessellation (one slide).
    output_h5_path (str): Patch HDF5 to write the derived grid to.
    source_h5_paths (List[str]): Several slides' source patch HDF5s (instead of source_h5_path).
    output_h5_paths (List[str]): Their output paths, in the same order.
    slide_id (str): Slide ID recorded in the output (``name`` attribute); single-slide mode only.
    patch_size (int): Derived tile size in pixels at the source tiles' resolution,
        not smaller than the source tiles.
    skip_missing (bool): In batch mode, skip a slide whose source file does not exist
        (e.g. it failed tessellation) instead of failing.
    """

    source_h5_path: Optional[str] = None
    output_h5_path: Optional[str] = None
    source_h5_paths: Optional[List[str]] = None
    output_h5_paths: Optional[List[str]] = None
    slide_id: Optional[str] = None
    patch_size: int = 512
    skip_missing: bool = False


cs = ConfigStore.instance()
cs.store(name="derive_tile_grid_config", node=DeriveTileGridConfig)


@hydra.main(version_base=None, config_path=None, config_name="derive_tile_grid_config")
def main(cfg: DeriveTileGridConfig):
    """Write the derived grid for one slide or a batch of slides."""
    from pathlib import Path

    if cfg.source_h5_paths is not None or cfg.output_h5_paths is not None:
        if cfg.source_h5_paths is None or cfg.output_h5_paths is None:
            raise ValueError("Batch mode needs both source_h5_paths and output_h5_paths")
        if len(cfg.source_h5_paths) != len(cfg.output_h5_paths):
            raise ValueError(
                f"source_h5_paths ({len(cfg.source_h5_paths)}) and output_h5_paths "
                f"({len(cfg.output_h5_paths)}) must have the same length")
        for src, dst in zip(cfg.source_h5_paths, cfg.output_h5_paths):
            if cfg.skip_missing and not Path(src).exists():
                logger.warning("%s does not exist; skipping", src)
                continue
            derive_patch_h5(src, dst, cfg.patch_size)
        return
    if cfg.source_h5_path is None or cfg.output_h5_path is None:
        raise ValueError("Single-slide mode needs source_h5_path and output_h5_path")
    derive_patch_h5(cfg.source_h5_path, cfg.output_h5_path, cfg.patch_size, slide_id=cfg.slide_id)


if __name__ == "__main__":
    main()
