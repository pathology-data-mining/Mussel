import functools
import logging
import os
from concurrent import futures
from typing import Optional

import h5py
import numpy as np
import tiffslide
from tqdm import tqdm

from mussel.utils.segment import (get_native_size, get_slide_mpp,
                                  is_black_patch, is_white_patch)
from mussel.utils.timer import timed
from mussel.utils.wsi_backend import open_slide

log = logging.getLogger(__name__)


def export_tile(
    tile_coords: np.ndarray,
    wsi_object: tiffslide.TiffSlide,
    patch_size: int,
    output_path: str,
) -> None:
    """Utility function to export tile to .png file"""
    patch = wsi_object.read_region(tile_coords, 0, (patch_size, patch_size)).convert(
        "RGB"
    )
    if is_white_patch(np.array(patch)) or is_black_patch(np.array(patch)):
        return
    file_path = os.path.join(output_path, f"{tile_coords[0]}_{tile_coords[1]}.png")
    patch.save(file_path, "png")


def _level0_tile_size(attrs, wsi) -> int:
    """Level-0 tile size from a patch H5's coords attributes.

    ``patch_size`` is the tile size read at ``patch_level`` (0 for Mussel's own
    tessellation), so it is scaled by that level's downsample.
    """
    level = int(attrs.get("patch_level", 0))
    downsample = float(wsi.level_downsamples[level]) if level else 1.0
    return int(round(int(attrs["patch_size"]) * downsample))


@timed
def export_tiles(
    patch_h5_path: str,
    slide_path: str,
    output_png_path: str,
    patch_size: Optional[int] = None,
    mpp: Optional[float] = None,
    num_workers: int = 16,
    slide_mpp_override: Optional[float] = None,
) -> None:
    """Export tiles from a whole slide image to individual PNG files.

    By default each tile is exported at the size the tessellation recorded in the
    patch H5 (coords attributes ``patch_size`` / ``patch_level``), i.e. the same
    level-0 region that feature extraction reads. Passing ``patch_size`` and/or
    ``mpp`` instead derives the level-0 size from them and the slide's MPP; a value
    left as None then comes from the H5 (``patch_size_to_resize_to_for_desired_mpp``
    / ``mpp``), else 256 px / 0.5 µm/px.

    Args:
        patch_h5_path: Path to HDF5 file containing tile coordinates.
        slide_path: Path to the whole slide image.
        output_png_path: Directory to save PNG files.
        patch_size: Patch size in pixels at ``mpp`` (default: from the H5).
        mpp: Microns per pixel of ``patch_size`` (default: from the H5).
        num_workers: Number of worker threads (default: 16).
        slide_mpp_override: If provided, use this as the slide's native MPP instead of
            reading from metadata.  Useful when the slide lacks MPP tags.
    """

    log.info(f"Loading .patches.h5 file: {patch_h5_path}")

    with h5py.File(patch_h5_path, "r") as patches_h5:
        tile_coords = np.array(patches_h5["coords"])
        attrs = dict(patches_h5["coords"].attrs)

    wsi = open_slide(slide_path)
    try:
        if patch_size is None and mpp is None and "patch_size" in attrs:
            native_patch_size = _level0_tile_size(attrs, wsi)
            log.info(f"Using the tile size recorded in {patch_h5_path}: {native_patch_size} px at level 0")
        else:
            if patch_size is None:
                patch_size = int(attrs.get("patch_size_to_resize_to_for_desired_mpp", 256))
            if mpp is None:
                mpp = float(attrs.get("mpp", 0.5))
            # Get MPP with fallback handling
            slide_mpp = get_slide_mpp(wsi, slide_path, slide_mpp_override=slide_mpp_override)
            native_patch_size = get_native_size(patch_size, mpp, slide_mpp)

        log.info(
            f"Exporting approx. {len(tile_coords)} tiles as .png files to {output_png_path}"
        )
        n_tiles = len(tile_coords)

        partial = functools.partial(
            export_tile,
            wsi_object=wsi,
            patch_size=native_patch_size,
            output_path=output_png_path,
        )

        with futures.ThreadPoolExecutor(num_workers) as executor:
            list(tqdm(executor.map(partial, tile_coords), total=n_tiles))
    finally:
        wsi.close()
