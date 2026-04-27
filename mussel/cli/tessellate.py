import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, List, Optional

import hydra
import numpy as np
import pandas as pd
import tiffslide
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import MISSING, OmegaConf

logger = logging.getLogger(__name__)

from mussel.utils.segment import (draw_slide_mask, save_patches_png,
                                  segment_tissue)


@dataclass
class SegConfig:
    """
    patch_size (int): Tile size in pixels at the resolution set by ``mpp``.
    step_size (int): Step size in pixels between tile origins. Defaults to ``patch_size`` (no overlap).
        Overridden by ``overlap`` if set.
    overlap (int): Tile overlap in absolute pixels (0 = no overlap).
        Sets ``step_size = patch_size - overlap``.
    mpp (float): Target resolution for tile extraction in microns per pixel (µm/px).
        0.5 ≈ 20× magnification; 0.25 ≈ 40×.
    seg_level (int): Slide pyramid level used for tissue segmentation. If negative, the level
        closest to a 64× downsample is chosen automatically.
    segment_threshold (int): HSV-value threshold for the ``"classic"`` and ``"otsu"`` backends.
        Pixels with value ≤ threshold are set to 0 (background); all others are set to
        ``segment_max_value`` (foreground). Ignored by the ``"neural"`` backend.
    segment_max_value (int): Foreground pixel value assigned in the binary tissue mask (default 255).
        Ignored by the ``"neural"`` backend.
    median_blur_ksize (int): Kernel size for the median blur applied to the tissue mask before
        thresholding. Must be an odd integer greater than 1. Larger values smooth small noise.
        Ignored by the ``"neural"`` backend.
    morphology_ex_kernel (int): Kernel size for morphological closing applied to the tissue mask
        (0 to disable). Applied regardless of ``seg_model``.
    ref_patch_size (int): Reference patch size (in pixels) used to scale the ``tissue_area_threshold``
        and ``hole_area_threshold`` values.
    use_otsu (bool): **Deprecated** — use ``seg_model="otsu"`` instead.
    tissue_area_threshold (int): Minimum size of a tissue contour, expressed as the number of
        tiles (at the configured ``patch_size`` and ``mpp``) the region must span. Contours
        smaller than this are discarded as debris. Default 100 (≈ a ~3.2 mm² region at
        256 px / 0.5 µm/px). Set to 1 to keep all contours (useful for biopsies).
    hole_area_threshold (int): Minimum size of a hole inside a tissue contour, expressed as the
        number of tiles it spans. Holes smaller than this are filled in (treated as tissue).
        Default 16.
    max_num_holes (int): Maximum number of holes retained per tissue contour.
    keep_ids (List[int]): Contour IDs to keep; all others are discarded. Empty list keeps all.
    exclude_ids (List[int]): Contour IDs to discard. Empty list excludes none.
    min_tissue_proportion (float): Minimum fraction of tile area that must be tissue (0.0–1.0).
        Tiles below this threshold are discarded.
    remove_artifacts (bool): If True, apply artifact removal to the tissue mask before patching
        (requires ``artifact_remover_fn`` to be provided).
    remove_penmarks (bool): If True, apply pen-mark removal to the tissue mask before patching
        (requires ``artifact_remover_fn`` to be provided).
    seg_model (str): Segmentation backend:
        ``"classic"`` (default) — HSV color space + fixed intensity threshold;
        ``"otsu"`` — HSV color space + Otsu's automatic threshold;
        ``"neural"`` — DeepLabV3 deep-learning model (requires torch; weights downloaded
        automatically from HuggingFace on first use).
        ``segment_threshold`` and ``median_blur_ksize`` are ignored for ``"neural"``;
        ``morphology_ex_kernel`` applies to all three backends.
    slide_mpp_override (float): If set, use this value (µm/px) as the slide's native MPP instead of
        reading it from slide metadata. Useful when MPP tags are missing or incorrect.
    artifact_remover_fn: Optional callable ``(img, mask, mpp) -> mask`` where ``img`` is the RGB
        thumbnail, ``mask`` is the binary tissue mask, and ``mpp`` is the thumbnail's µm/px.
        Returns a corrected binary mask. Use :class:`~mussel.utils.artifact_removal.GrandQCArtifactRemover`
        for a ready-made GrandQC-based implementation.
    """

    # Default patch size constant - used to detect when automatic patch size selection should apply
    DEFAULT_PATCH_SIZE: ClassVar[int] = 256

    patch_size: int = 256
    step_size: Optional[int] = None  # if None, defaults to patch_size
    mpp: float = 0.5
    seg_level: int = -1
    segment_threshold: int = 20
    segment_max_value: int = 255
    median_blur_ksize: int = 7
    morphology_ex_kernel: int = 0
    ref_patch_size: int = 512
    use_otsu: bool = False  # Deprecated: use seg_model="otsu" instead.
    tissue_area_threshold: int = 100
    hole_area_threshold: int = 16
    max_num_holes: int = 8
    keep_ids: List[int] = field(default_factory=list)
    exclude_ids: List[int] = field(default_factory=list)
    overlap: int = (
        0  # Patch overlap in absolute pixels (0 = no overlap). step_size = patch_size - overlap
    )
    min_tissue_proportion: float = (
        0.0  # Minimum fraction of patch that must be tissue (0.0-1.0). Patches below this are discarded.
    )
    remove_artifacts: bool = (
        False  # If True, apply artifact removal to the tissue mask before patching.
    )
    remove_penmarks: bool = (
        False  # If True, apply pen mark removal to the tissue mask before patching.
    )
    seg_model: str = (
        "classic"  # "classic" (HSV + fixed threshold), "otsu" (HSV + Otsu), or "neural" (DeepLabV3).
    )
    slide_mpp_override: Optional[float] = (
        None  # If set, use this as the slide's native MPP instead of reading from metadata.
    )


@dataclass
class BiopsySegConfig(SegConfig):
    """Preset tuned for needle-core and punch biopsies.

    Biopsies are small, elongated tissue fragments that occupy a narrow strip of the slide.
    This preset uses a lower tissue-area threshold (1) and fewer holes (2) so that small,
    disconnected tissue cores are retained rather than filtered out.

    When used with ``seg_model="neural"``, ``segment_threshold`` and ``median_blur_ksize``
    are ignored (a warning is emitted); ``morphology_ex_kernel``, ``tissue_area_threshold``,
    ``hole_area_threshold``, and ``max_num_holes`` still take effect.
    """

    segment_threshold: int = 15
    median_blur_ksize: int = 11
    morphology_ex_kernel: int = 2
    tissue_area_threshold: int = 1
    hole_area_threshold: int = 1
    max_num_holes: int = 2


@dataclass
class ResectionSegConfig(SegConfig):
    """Preset tuned for surgical resection specimens.

    Resections are large tissue sections that typically contain substantial background and
    cavities (e.g. necrosis, fat). This preset uses stronger morphological closing (kernel 4)
    to bridge gaps, while keeping the default area thresholds to discard small debris.

    When used with ``seg_model="neural"``, ``segment_threshold`` and ``median_blur_ksize``
    are ignored (a warning is emitted). Only ``morphology_ex_kernel=4`` differs from the
    ``default`` preset in that case, so consider using ``default seg_config.seg_model=neural``
    instead to avoid the warning.
    """

    segment_threshold: int = 15
    median_blur_ksize: int = 11
    morphology_ex_kernel: int = 4
    tissue_area_threshold: int = 100
    hole_area_threshold: int = 16
    max_num_holes: int = 8


@dataclass
class TcgaSegConfig(SegConfig):
    """Preset tuned for TCGA (The Cancer Genome Atlas) whole-slide images.

    TCGA slides often have lightly stained or faded tissue. This preset uses a lower
    ``segment_threshold`` (8) so that pale tissue regions are not discarded as background,
    with stronger morphological closing (kernel 4) to unify fragmented foreground regions.

    When used with ``seg_model="neural"``, ``segment_threshold`` is ignored (its non-default
    value triggers a warning); ``morphology_ex_kernel=4``, ``tissue_area_threshold=16``, and
    ``hole_area_threshold=4`` still take effect and remain beneficial.
    """

    segment_threshold: int = 8
    median_blur_ksize: int = 7
    morphology_ex_kernel: int = 4
    tissue_area_threshold: int = 16
    hole_area_threshold: int = 4
    max_num_holes: int = 8


@dataclass
class VisConfig:
    """
    Controls the appearance of tissue-mask and grid-mask overlay images.

    vis_level (int): Slide pyramid level to use as the visualization base image.
        If negative, the level closest to a 64× downsample is chosen automatically.
    outline (str): PIL color string for the tissue contour outline (e.g. ``"black"``, ``"red"``).
    fill (tuple): RGBA color (0–255 each) for the filled tissue region (e.g. ``(255, 0, 0, 80)``
        for semi-transparent red).
    custom_downsample (Optional[int]): Additional integer downsample applied on top of the
        pyramid level. If None, no extra downsampling is applied.
    """

    vis_level: int = -1
    outline = "black"
    fill = (255, 0, 0, 80)
    custom_downsample: Optional[int] = None


@dataclass
class PngConfig:
    """
    Controls filtering applied when saving tiles as PNG files (``output_png_dir``).

    filter_black_white (bool): If True, discard tiles that appear predominantly black or white
        before saving.
    white_threshold (int): Maximum mean HSV saturation (0–255) for a tile to be considered white.
        Tiles with mean saturation below this value are discarded as background.
    black_threshold (int): Maximum mean RGB value (0–255) for a tile to be considered black.
        Tiles where all three RGB channels average below this value are discarded.
    """

    filter_black_white: bool = True
    white_threshold: int = 15
    black_threshold: int = 50


defaults = ["_self_", {"seg_config": "default"}]


@dataclass
class TessellateConfig:
    """
    slide_path (str): Path to the whole-slide image (SVS, TIFF, NDPI, etc.).
    slide_id (Optional[str]): Identifier written into the HDF5 output. Defaults to the slide
        filename without extension.
    output_h5_path (str): Path for the HDF5 output file containing tile coordinates and metadata.
    output_png_dir (Optional[str]): Directory to save each tile as an individual PNG file.
        Filtered by ``png_config`` settings when set.
    output_mask_path (Optional[str]): Path to save a PNG overlay showing the segmented tissue
        polygons on a slide thumbnail.
    output_grid_mask_path (Optional[str]): Path to save a PNG overlay showing the selected tile
        grid on a slide thumbnail.
    output_thumbnail_path (Optional[str]): Path to save a plain slide thumbnail (no overlay).
    thumbnail_size (tuple): Width × height in pixels for the saved thumbnail (default 1024×1024).
    seg_config (SegConfig): Segmentation and tiling parameters (see below).
    vis_config (VisConfig): Visualization appearance parameters for mask overlays (see below).
    png_config (PngConfig): Filtering parameters applied when saving PNG tiles (see below).
    num_workers (int): Number of parallel workers used when saving PNG tiles.
    """

    defaults: List[Any] = field(default_factory=lambda: defaults)
    slide_path: str = MISSING
    slide_id: Optional[str] = None
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

tessellate tiles a whole-slide image and detects foreground tissue.  Tile coordinates and
slide metadata are written to an HDF5 (.h5) file for use in downstream steps such as
extract_features and tessellate_extract_features.

Key options (use Hydra override syntax, e.g. seg_config.mpp=0.25):
  slide_path          Path to the slide file (required)
  output_h5_path      Path for the output HDF5 file (required)
  seg_config          Preset segmentation profile: default | biopsy | resection | tcga
  seg_config.mpp      Target resolution in µm/px (default 0.5 ≈ 20×; 0.25 ≈ 40×)
  seg_config.patch_size  Tile size in pixels at the target MPP (default 256)
  seg_config.seg_model   Segmentation backend: classic | otsu | neural

Example:
  tessellate slide_path=slide.svs output_h5_path=out.h5 seg_config=biopsy
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
    """Tile a whole slide image and perform tissue segmentation."""
    if values := segment_tissue(
        slide_path=cfg.slide_path,
        slide_id=cfg.slide_id,
        output_h5_path=cfg.output_h5_path,
        **OmegaConf.to_container(cfg.seg_config),
    ):
        polygon, grid, coords, _ = values
    else:
        return

    if cfg.output_mask_path:
        mask = draw_slide_mask(
            cfg.slide_path,
            polygon,
            **OmegaConf.to_container(cfg.vis_config),
        )
        mask.save(cfg.output_mask_path)

    if cfg.output_grid_mask_path:
        grid_mask = draw_slide_mask(
            cfg.slide_path,
            grid,
            **OmegaConf.to_container(cfg.vis_config),
        )
        grid_mask.save(cfg.output_grid_mask_path)

    if cfg.output_png_dir:
        logger.info(f"saving patches to {cfg.output_png_dir}")
        save_patches_png(
            cfg.slide_path,
            coords,
            save_dir=cfg.output_png_dir,
            num_workers=cfg.num_workers,
            mpp=cfg.seg_config.mpp,
            patch_size=cfg.seg_config.patch_size,
            filter_black_white=cfg.png_config.filter_black_white,
            white_threshold=cfg.png_config.white_threshold,
            black_threshold=cfg.png_config.black_threshold,
            slide_mpp_override=cfg.seg_config.slide_mpp_override,
        )

    if cfg.output_thumbnail_path:
        with tiffslide.TiffSlide(cfg.slide_path) as wsi:
            logger.info(f"saving thumbnail to {cfg.output_thumbnail_path}")
            thumbnail = wsi.get_thumbnail(cfg.thumbnail_size)
            with open(cfg.output_thumbnail_path, "wb") as f:
                thumbnail.save(f)


if __name__ == "__main__":
    main()
