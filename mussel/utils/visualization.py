"""Heatmap visualization utilities for whole-slide image attention scores."""

import logging
import os
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from scipy.stats import rankdata as _scipy_rankdata

    def _rank_normalize(scores: np.ndarray) -> np.ndarray:
        return _scipy_rankdata(scores, "average") / len(scores)

except ImportError:
    _scipy_rankdata = None  # type: ignore[assignment]

    def _rank_normalize(scores: np.ndarray) -> np.ndarray:
        n = len(scores)
        order = np.argsort(scores)
        ranks = np.empty(n, dtype=float)
        ranks[order] = (np.arange(n) + 1.0) / n
        return ranks


logger = logging.getLogger(__name__)


def create_overlay(
    scores: np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int,
    scale: np.ndarray,
    region_size: Tuple[int, int],
) -> np.ndarray:
    """Build a 2-D heatmap overlay from per-patch scores and coordinates.

    Overlapping patches are averaged.  Pixels with no patch coverage are set to
    ``NaN`` so that downstream colourmap application can treat them as
    transparent / background.

    Args:
        scores: Per-patch scores of shape ``[N]``.
        coords: Top-left coordinates of each patch at level-0 resolution,
            shape ``[N, 2]`` (x, y).
        patch_size_level0: Patch side length (pixels) at level-0 resolution.
        scale: Two-element array ``[sx, sy]`` mapping level-0 pixels to the
            visualisation resolution.
        region_size: ``(width, height)`` of the output overlay in visualisation
            pixels.

    Returns:
        Float32 overlay array of shape ``(height, width)`` with values in the
        score range plus ``NaN`` for empty areas.
    """
    patch_size = np.ceil(
        np.array([patch_size_level0, patch_size_level0]) * scale
    ).astype(int)
    scaled_coords = np.ceil(coords * scale).astype(int)

    overlay = np.zeros(tuple(np.flip(region_size)), dtype=float)
    counter = np.zeros_like(overlay, dtype=np.uint16)

    for idx, coord in enumerate(scaled_coords):
        x, y = coord[0], coord[1]
        overlay[y : y + patch_size[1], x : x + patch_size[0]] += scores[idx]
        counter[y : y + patch_size[1], x : x + patch_size[0]] += 1

    zero_mask = counter == 0
    overlay[~zero_mask] /= counter[~zero_mask]
    overlay[zero_mask] = np.nan

    return overlay


def apply_colormap(overlay: np.ndarray, cmap_name: str) -> np.ndarray:
    """Map scalar overlay values to RGB colours using a matplotlib colormap.

    ``NaN`` pixels (empty background) are mapped to black ``(0, 0, 0)``.

    Args:
        overlay: 2-D float array, possibly containing ``NaN`` values.
        cmap_name: Name of any matplotlib colormap (e.g. ``"coolwarm"``).

    Returns:
        RGB uint8 array of shape ``(H, W, 3)``.
    """
    cmap = plt.get_cmap(cmap_name)
    overlay_colored = np.zeros((*overlay.shape, 3), dtype=np.uint8)
    valid_mask = ~np.isnan(overlay)
    colored_valid = (cmap(overlay[valid_mask]) * 255).astype(np.uint8)[:, :3]
    overlay_colored[valid_mask] = colored_valid
    return overlay_colored


def visualize_heatmap(
    slide_path: str,
    scores: np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int,
    vis_level: int = 2,
    cmap: str = "coolwarm",
    normalize: bool = True,
    num_top_patches: int = -1,
    output_path: str = "output/heatmap.png",
    output_patch_dir: Optional[str] = None,
    vis_mag: Optional[int] = None,
    overlay_only: bool = False,
) -> str:
    """Generate and save a heatmap overlaid on a whole-slide image.

    Args:
        slide_path: Filesystem path to the WSI file (any format supported by
            tiffslide).
        scores: Per-patch attention/score values, shape ``[N]``.
        coords: Level-0 top-left coordinates of each patch, shape ``[N, 2]``.
        patch_size_level0: Patch side length at level-0 resolution (pixels).
        vis_level: WSI pyramid level used for visualisation (ignored when
            ``vis_mag`` is set).  Defaults to 2.
        cmap: Matplotlib colormap name.  Defaults to ``"coolwarm"``.
        normalize: If ``True`` the scores are rank-normalised to ``[0, 1]``
            before colourmap application.  Defaults to ``True``.
        num_top_patches: Number of highest-scoring patches to save as
            individual PNG files.  ``-1`` disables this feature.  Defaults to
            ``-1``.
        output_path: Full filesystem path for the saved heatmap PNG (parent
            directory is created automatically).  Defaults to
            ``"output/heatmap.png"``.
        output_patch_dir: Directory where top-k patch tiles are written when
            ``num_top_patches > 0``.  If ``None`` (default), tiles are saved
            to a ``topk_patches/`` subdirectory next to ``output_path``::

                <output_path parent>/
                  heatmap.png          ← output_path
                  topk_patches/        ← default output_patch_dir
                    top_0_score_0.9234.png
                    top_1_score_0.8912.png
                    ...

            When an explicit path is provided, the directory is created
            automatically and patches are written directly there.
        vis_mag: Target magnification for visualisation.  When provided,
            ``vis_level`` is ignored and the best matching pyramid level is
            selected automatically.
        overlay_only: If ``True``, save only the colourised overlay without
            blending it onto the slide thumbnail.  Defaults to ``False``.

    Returns:
        Absolute path to the saved heatmap image.
    """
    import tiffslide as openslide

    wsi = openslide.open_slide(slide_path)

    if normalize:
        scores = _rank_normalize(scores)

    if vis_mag is not None:
        src_downsample = wsi.level_downsamples[0]
        # level_downsamples[0] is always 1.0; we need the slide's native mag.
        # tiffslide exposes mpp; convert to approximate downsample ratio.
        downsample = wsi.level_downsamples[vis_level]  # fallback value
        try:
            mpp = wsi.properties.get("tiffslide.mpp-x") or wsi.properties.get(
                "openslide.mpp-x"
            )
            if mpp is not None:
                native_mag = 10.0 / float(mpp)
                downsample = native_mag / vis_mag
                vis_level = wsi.get_best_level_for_downsample(downsample)
            else:
                logger.warning(
                    "MPP not found in slide properties; vis_mag ignored, "
                    "falling back to vis_level=%d.",
                    vis_level,
                )
        except Exception as exc:
            logger.warning(
                "Could not determine vis_level from vis_mag=%d (%s); "
                "falling back to vis_level=%d.",
                vis_mag,
                exc,
                vis_level,
            )
        downsample = wsi.level_downsamples[vis_level]
    else:
        downsample = wsi.level_downsamples[vis_level]

    scale = np.array([1.0 / downsample, 1.0 / downsample])
    region_size = tuple(
        (np.array(wsi.level_dimensions[0]) * scale).astype(int)
    )  # (W, H)

    overlay = create_overlay(scores, coords, patch_size_level0, scale, region_size)
    overlay_colored = apply_colormap(overlay, cmap)

    if overlay_only:
        blended_img = overlay_colored
    else:
        img = wsi.read_region(
            (0, 0), vis_level, wsi.level_dimensions[vis_level]
        ).convert("RGB")
        img = img.resize(region_size, resample=Image.Resampling.BICUBIC)
        img_arr = np.array(img)
        blended_img = cv2.addWeighted(img_arr, 0.6, overlay_colored, 0.4, 0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    heatmap_path = output_path
    Image.fromarray(blended_img).save(heatmap_path)
    logger.info("Saved heatmap to %s", heatmap_path)

    if num_top_patches > 0:
        if output_patch_dir is None:
            output_patch_dir = os.path.join(
                os.path.dirname(os.path.abspath(output_path)), "topk_patches"
            )
        os.makedirs(output_patch_dir, exist_ok=True)
        topk_indices = np.argsort(scores)[-num_top_patches:]
        for rank, idx in enumerate(topk_indices):
            x, y = int(coords[idx][0]), int(coords[idx][1])
            patch = wsi.read_region(
                (x, y), 0, (patch_size_level0, patch_size_level0)
            )
            patch_path = os.path.join(
                output_patch_dir, f"top_{rank}_score_{scores[idx]:.4f}.png"
            )
            patch.save(patch_path)
        logger.info("Saved %d top patches to %s", num_top_patches, output_patch_dir)

    return heatmap_path
