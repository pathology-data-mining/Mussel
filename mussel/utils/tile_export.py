import functools
import logging
import os
from concurrent import futures

import h5py
import numpy as np
import tiffslide
from tqdm import tqdm

from mussel.utils.segment import is_black_patch, is_white_patch, get_native_size, get_slide_mpp
from mussel.utils.timer import timed

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


@timed
def export_tiles(
    patch_h5_path: str,
    slide_path: str,
    output_png_path: str,
    patch_size: int = 256,
    mpp: float = 0.5,
    num_workers: int = 16,
) -> None:
    """Export tiles from a whole slide image to individual PNG files.

    Args:
        patch_h5_path: Path to HDF5 file containing tile coordinates.
        slide_path: Path to the whole slide image.
        output_png_path: Directory to save PNG files.
        patch_size: Patch size in pixels (default: 256).
        mpp: Microns per pixel (default: 0.5).
        num_workers: Number of worker threads (default: 16).
    """

    log.info(f"Loading .patches.h5 file: {patch_h5_path}")

    with h5py.File(patch_h5_path, "r") as patches_h5:
        tile_coords = np.array(patches_h5["coords"])

    # Init whole slide image
    wsi = tiffslide.TiffSlide(slide_path)
    
    # Get MPP with fallback handling
    slide_mpp = get_slide_mpp(wsi, slide_path)

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
