"""Tile grids built from an existing tessellation's patch HDF5.

derive_patch_h5: a coarser grid from finer tissue tiles. Every ``patch_size`` px
cell, aligned to the slide origin, that contains the centre of at least one
source tile, at the source tiles' resolution. A slide encoder that needs larger
tiles (TITAN: 512 px at 0.5 µm/px) then reads the same tissue as the patch
encoders tiled finer (224 px), without a second segmentation. reef v2's 512 px
TITAN embeddings were made this way from its 224 px tiles.

The output is a patch HDF5 with the source's coords attributes (updated where they
change), so it reads like a tessellate output.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from mussel.utils.segment import get_native_size

logger = logging.getLogger(__name__)

GRID_METHOD = "cells_containing_source_tile_centres"


def _attr(attrs: dict, key: str, path):
    if key not in attrs:
        raise ValueError(f"{path}: coords has no attribute {key!r}")
    value = attrs[key]
    return value.item() if hasattr(value, "item") else value


def _write_coords(path, coords: np.ndarray, attrs: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        ds = f.create_dataset("coords", data=np.asarray(coords, dtype=np.int64).reshape(-1, 2))
        for key, value in attrs.items():
            ds.attrs[key] = value
    os.replace(tmp, path)


def derive_grid(coords, source_patch_size: int, patch_size: int, mpp: float, slide_mpp: float):
    """Return ``(coords, native_patch_size)`` of the ``patch_size`` px cells containing source tile centres.

    Args:
        coords: Level-0 top-left corners of the source tiles, shape (N, 2).
        source_patch_size: Level-0 size of the source tiles.
        patch_size: Size of the derived tiles in pixels at ``mpp``.
        mpp: Resolution (µm/px) of the derived tiles, normally the source tiles'.
        slide_mpp: The slide's native resolution (µm/px).

    Output coords are level-0, unique and sorted by (x, y).
    """
    native = get_native_size(patch_size, mpp, slide_mpp)
    if native <= 0:
        raise ValueError(f"non-positive level-0 tile size {native} for slide mpp {slide_mpp}")
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.int64), native
    centres = coords + source_patch_size / 2.0
    cells = np.unique(np.floor(centres / native).astype(np.int64), axis=0)
    return cells * native, native


def derive_patch_h5(source_h5_path, output_h5_path, patch_size: int, slide_id: Optional[str] = None) -> dict:
    """Write the coarser ``patch_size`` grid of a tessellation (derive_grid); return a summary.

    The derived tiles keep the source tiles' resolution, so ``patch_size`` must not
    be smaller than the source tiles. The source must be tiled at level 0.
    """
    with h5py.File(source_h5_path, "r") as f:
        if "coords" not in f:
            raise ValueError(f"{source_h5_path}: no coords dataset")
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs)
    if int(_attr(attrs, "patch_level", source_h5_path)) != 0:
        raise ValueError(f"{source_h5_path}: patch_level {attrs['patch_level']} != 0")
    source_px = int(_attr(attrs, "patch_size_to_resize_to_for_desired_mpp", source_h5_path))
    source_native = int(_attr(attrs, "patch_size", source_h5_path))
    mpp = float(_attr(attrs, "mpp", source_h5_path))
    slide_mpp = float(_attr(attrs, "native_mpp", source_h5_path))
    if not slide_mpp > 0:
        raise ValueError(f"{source_h5_path}: invalid native_mpp {slide_mpp}")
    if patch_size < source_px:
        raise ValueError(f"derived tiles ({patch_size} px) must not be smaller than the source tiles ({source_px} px)")

    grid, native = derive_grid(coords, source_native, patch_size, mpp, slide_mpp)
    out = dict(attrs)
    out.update(
        patch_size=native,
        step_size=native,
        patch_size_to_resize_to_for_desired_mpp=patch_size,
        mpp=mpp,
        patch_level=0,
        overlap=0,
        max_tiles=-1,
        derived_grid_method=GRID_METHOD,
        derived_source_tiles=len(coords),
        derived_source_patch_size=source_native,
        derived_source_tile_px=source_px,
    )
    if slide_id is not None:
        out["name"] = str(slide_id)
    _write_coords(output_h5_path, grid, out)
    logger.info("%s: %d x %d px tiles -> %d x %d px (level-0 %d -> %d)",
                slide_id or Path(source_h5_path).name, len(coords), source_px, len(grid), patch_size,
                source_native, native)
    return {"source_tiles": int(len(coords)), "tiles": int(len(grid)), "patch_size": native}
