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

from mussel.utils.artifact_removal import (
    EXCLUDE_ALL_ARTIFACTS,
    EXCLUDE_PENMARKS_ONLY,
    GrandQCArtifactRemover,
)
from mussel.utils.segment import draw_slide_mask, save_patches_png, segment_tissue


@dataclass
class NeuralSegConfig:
    """Runtime controls for the ``seg_model=neural`` backend.

    These parameters configure model loading and inference, rather than the
    tissue-mask morphology controls in :class:`SegConfig`.
    """

    weights_path: Optional[str] = None
    device: str = "auto"
    batch_size: int = 8
    confidence_thresh: float = 0.5
    # None preserves the segmenter's environment fallback (4096 when unset).
    max_inference_tiles: Optional[int] = None


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
    max_tiles (Optional[int]): Maximum number of output tiles to retain after tissue filtering.
        ``None`` keeps all tiles.
    max_tiles_strategy (str): Sampling strategy when ``max_tiles`` is reached: ``"random"``
        (seeded, default) or ``"first"``.
    max_tiles_seed (int): Random seed used by the ``"random"`` max-tile strategy.
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
    artifact_exclude_classes: Optional[List[int]] = field(default=None)
    # Optional list of GrandQC class indices to exclude (overrides remove_artifacts /
    # remove_penmarks preset logic when set).  Use the CLASS_* constants from
    # mussel.utils.artifact_removal, e.g. [4, 7] for pen marks + background only.
    # Common presets: EXCLUDE_PENMARKS_ONLY=[4,7], EXCLUDE_FOLDS_AND_PENMARKS=[4,5,6,7],
    # EXCLUDE_ALL_ARTIFACTS=[2,3,4,5,6,7].  When None, the preset is derived from the
    # remove_artifacts / remove_penmarks flags.
    seg_model: str = (
        "classic"  # "classic" (HSV + fixed threshold), "otsu" (HSV + Otsu), or "neural" (DeepLabV3).
    )
    slide_mpp_override: Optional[float] = (
        None  # If set, use this as the slide's native MPP instead of reading from metadata.
    )
    max_tiles: Optional[int] = None
    max_tiles_strategy: str = "random"
    max_tiles_seed: int = 42


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
    slide_paths (Optional[List[str]]): Batch mode input slide paths. Mutually exclusive with
        ``slide_path``.
    slide_ids (Optional[List[str]]): Optional batch mode slide IDs. Defaults to each slide
        filename without extension.
    output_h5_paths (Optional[List[str]]): Batch mode output HDF5 paths. Mutually exclusive with
        ``output_dir``.
    output_dir (Optional[str]): Batch mode output directory. When set, writes
        ``{slide_id}.patch.h5`` for each input slide.
    continue_on_error (bool): In batch mode, continue processing remaining slides after a
        per-slide failure and exit successfully after writing a failures TSV if
        ``failures_tsv_path`` is set. By default any failure causes a non-zero exit after the
        first failed slide.
    failures_tsv_path (Optional[str]): Optional TSV path receiving ``slide_id, slide_path,
        output_h5_path, error`` rows for batch failures.
    output_png_dir (Optional[str]): Directory to save each tile as an individual PNG file.
        Filtered by ``png_config`` settings when set.
    output_mask_path (Optional[str]): Path to save a PNG overlay showing the segmented tissue
        polygons on a slide thumbnail.
    output_grid_mask_path (Optional[str]): Path to save a PNG overlay showing the selected tile
        grid on a slide thumbnail.
    output_thumbnail_path (Optional[str]): Path to save a plain slide thumbnail (no overlay).
    thumbnail_size (tuple): Width × height in pixels for the saved thumbnail (default 1024×1024).
    seg_config (SegConfig): Segmentation and tiling parameters (see below).
    neural_config (NeuralSegConfig): Model and inference parameters used when
        ``seg_config.seg_model="neural"``.
    vis_config (VisConfig): Visualization appearance parameters for mask overlays (see below).
    png_config (PngConfig): Filtering parameters applied when saving PNG tiles (see below).
    num_workers (int): Number of parallel workers used when saving PNG tiles.
    """

    defaults: List[Any] = field(default_factory=lambda: defaults)
    slide_path: Optional[str] = None
    slide_id: Optional[str] = None
    output_h5_path: Optional[str] = None
    slide_paths: Optional[List[str]] = None
    slide_ids: Optional[List[str]] = None
    output_h5_paths: Optional[List[str]] = None
    output_dir: Optional[str] = None
    continue_on_error: bool = False
    failures_tsv_path: Optional[str] = None
    output_png_dir: Optional[str] = None
    output_mask_path: Optional[str] = None
    output_grid_mask_path: Optional[str] = None
    output_thumbnail_path: Optional[str] = None
    thumbnail_size: tuple = (1024, 1024)
    num_workers: int = 4
    seg_config: SegConfig = MISSING
    neural_config: NeuralSegConfig = field(default_factory=NeuralSegConfig)
    vis_config: VisConfig = field(default_factory=VisConfig)
    png_config: PngConfig = field(default_factory=PngConfig)


desc_doc = """== ${hydra.help.app_name} ==

tessellate tiles a whole-slide image and detects foreground tissue.  Tile coordinates and
slide metadata are written to an HDF5 (.h5) file for use in downstream steps such as
extract_features and tessellate_extract_features.

Key options (use Hydra override syntax, e.g. seg_config.mpp=0.25):
  slide_path          Path to the slide file (required)
  output_h5_path      Path for the output HDF5 file (required)
  slide_paths         Batch mode slide paths; use output_h5_paths or output_dir for outputs
  seg_config          Preset segmentation profile: default | biopsy | resection | tcga
  seg_config.mpp      Target resolution in µm/px (default 0.5 ≈ 20×; 0.25 ≈ 40×)
  seg_config.patch_size  Tile size in pixels at the target MPP (default 256)
  seg_config.seg_model   Segmentation backend: classic | otsu | neural
  seg_config.max_tiles   Optional cap on output tiles after tissue filtering
  neural_config.*        Neural model/runtime controls (used with seg_model=neural)

Example:
  tessellate slide_path=slide.svs output_h5_path=out.h5 seg_config=biopsy
  tessellate 'slide_paths=[a.svs,b.svs]' output_dir=tiles seg_config=biopsy
"""

parameter_doc = f"""
== Available Parameters ==
{TessellateConfig.__doc__}
seg_config: {SegConfig.__doc__}
neural_config: {NeuralSegConfig.__doc__}
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


def _build_artifact_remover(
    seg_cfg: dict,
) -> "Optional[GrandQCArtifactRemover]":
    """Instantiate a :class:`~mussel.utils.artifact_removal.GrandQCArtifactRemover`
    from a plain ``seg_cfg`` dict (as returned by ``OmegaConf.to_container``).

    Returns ``None`` when artifact removal is not requested.  The returned
    instance is safe to share across multiple slides.

    Exclusion set resolution order:

    1. ``artifact_exclude_classes`` list — full per-run control.
    2. Flags ``remove_artifacts`` / ``remove_penmarks``:

       * ``remove_artifacts=True``             → :data:`~mussel.utils.artifact_removal.EXCLUDE_ALL_ARTIFACTS`
       * ``remove_penmarks=True`` only         → :data:`~mussel.utils.artifact_removal.EXCLUDE_PENMARKS_ONLY`
    """
    if not (seg_cfg.get("remove_artifacts") or seg_cfg.get("remove_penmarks")):
        return None

    explicit = seg_cfg.get("artifact_exclude_classes")
    if explicit:
        exclude_classes = frozenset(int(c) for c in explicit)
        mode = f"custom {sorted(exclude_classes)}"
    elif bool(seg_cfg.get("remove_artifacts")):
        exclude_classes = EXCLUDE_ALL_ARTIFACTS
        mode = "EXCLUDE_ALL_ARTIFACTS (aggressive)"
    else:
        exclude_classes = EXCLUDE_PENMARKS_ONLY
        mode = "EXCLUDE_PENMARKS_ONLY (conservative)"

    remover = GrandQCArtifactRemover(exclude_classes=exclude_classes)
    logger.info(
        "Artifact removal enabled: remove_artifacts=%s remove_penmarks=%s "
        "exclude_classes=%s",
        seg_cfg.get("remove_artifacts"),
        seg_cfg.get("remove_penmarks"),
        mode,
    )
    return remover


def _slide_id_for_path(slide_path: str, slide_id: Optional[str] = None) -> str:
    return slide_id if slide_id else Path(slide_path).stem


def _run_tessellation(
    *,
    slide_path: str,
    output_h5_path: str,
    seg_cfg: dict,
    artifact_remover_fn: "Optional[GrandQCArtifactRemover]",
    slide_id: Optional[str] = None,
    neural_segmenter: Optional[Any] = None,
) -> tuple[Any, Any, np.ndarray] | None:
    # Strip config-only keys that are not segment_tissue() parameters.
    call_seg_cfg = dict(seg_cfg)
    call_seg_cfg.pop("artifact_exclude_classes", None)
    if neural_segmenter is not None:
        call_seg_cfg["neural_segmenter"] = neural_segmenter
    values = segment_tissue(
        slide_path=slide_path,
        slide_id=slide_id,
        output_h5_path=output_h5_path,
        artifact_remover_fn=artifact_remover_fn,
        **call_seg_cfg,
    )
    if not values:
        return None
    polygon, grid, coords, _ = values
    return polygon, grid, coords


def _resolve_batch_outputs(cfg: TessellateConfig) -> list[tuple[str, str, str]]:
    slide_paths = list(cfg.slide_paths or [])
    if not slide_paths:
        raise ValueError(
            "Batch mode requires slide_paths to contain at least one slide."
        )

    slide_ids = (
        list(cfg.slide_ids)
        if cfg.slide_ids is not None
        else [_slide_id_for_path(p) for p in slide_paths]
    )
    if len(slide_ids) != len(slide_paths):
        raise ValueError(
            f"slide_ids length ({len(slide_ids)}) must match slide_paths length ({len(slide_paths)})."
        )

    has_output_h5_paths = cfg.output_h5_paths is not None
    has_output_dir = cfg.output_dir is not None
    if has_output_h5_paths == has_output_dir:
        raise ValueError(
            "Batch mode requires exactly one of output_h5_paths or output_dir."
        )

    if has_output_h5_paths:
        output_h5_paths = list(cfg.output_h5_paths or [])
        if len(output_h5_paths) != len(slide_paths):
            raise ValueError(
                "output_h5_paths length "
                f"({len(output_h5_paths)}) must match slide_paths length ({len(slide_paths)})."
            )
    else:
        output_dir = Path(cfg.output_dir)
        output_h5_paths = [
            str(output_dir / f"{slide_id}.patch.h5") for slide_id in slide_ids
        ]

    duplicate_outputs = sorted(
        output_path
        for output_path in set(output_h5_paths)
        if output_h5_paths.count(output_path) > 1
    )
    if duplicate_outputs:
        raise ValueError(
            "Batch mode output paths must be unique; duplicate output_h5_path(s): "
            + ", ".join(duplicate_outputs)
        )

    return list(zip(slide_paths, slide_ids, output_h5_paths))


def _build_neural_segmenter(
    seg_cfg: dict,
    use_gpu: Optional[bool] = None,
    gpu_device_id: Optional[int | List[int]] = None,
    gpu_device_ids: Optional[List[int]] = None,
    neural_config: Optional[dict] = None,
) -> Optional[Any]:
    if str(seg_cfg.get("seg_model", "classic")).strip().lower() != "neural":
        return None

    from mussel.utils.neural_seg import NeuralTissueSegmenter

    neural_config = dict(neural_config or {})
    neural_config.pop("_target_", None)
    configured_device = neural_config.get("device", "auto")
    device_is_auto = str(configured_device).strip().lower() == "auto"

    # An explicit neural device is independent from feature extraction's
    # use_gpu setting. Only resolve GPU options when the neural device is auto.
    if device_is_auto and use_gpu is not None:
        if not use_gpu:
            neural_config["device"] = "cpu"
        else:
            import torch

            from mussel.utils.gpu import first_gpu_device_id, resolve_gpu_device_id

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "seg_config.seg_model='neural' requested with use_gpu=True, "
                    "but CUDA is not available. Set use_gpu=False or "
                    "neural_config.device=cpu to run neural segmentation on CPU."
                )
            device_id = first_gpu_device_id(
                resolve_gpu_device_id(gpu_device_id, gpu_device_ids)
            )
            neural_config["device"] = (
                "cuda" if device_id is None else f"cuda:{device_id}"
            )
    elif not device_is_auto and str(configured_device).strip().lower().startswith(
        "cuda"
    ):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"neural_config.device={configured_device!r} requested, "
                "but CUDA is not available. Set neural_config.device=cpu."
            )

    logger.info("Loading neural tissue segmenter.")
    return NeuralTissueSegmenter(**neural_config)


def _run_batch(
    cfg: TessellateConfig, seg_cfg: dict, neural_config: Optional[dict] = None
) -> None:
    if any(
        value is not None
        for value in (
            cfg.output_png_dir,
            cfg.output_mask_path,
            cfg.output_grid_mask_path,
            cfg.output_thumbnail_path,
        )
    ):
        raise ValueError(
            "Batch tessellate mode only writes patch H5 outputs; PNG, mask, grid-mask, "
            "and thumbnail outputs are not supported."
        )

    artifact_remover_fn = _build_artifact_remover(seg_cfg)
    neural_segmenter = _build_neural_segmenter(seg_cfg, neural_config=neural_config)
    failures: list[tuple[str, str, str, str]] = []
    items = _resolve_batch_outputs(cfg)
    logger.info("Batch tessellating %d slide(s)", len(items))

    for i, (slide_path, slide_id, output_h5_path) in enumerate(items, start=1):
        try:
            Path(output_h5_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info("Tessellating slide %d/%d: %s", i, len(items), slide_id)
            result = _run_tessellation(
                slide_path=slide_path,
                slide_id=slide_id,
                output_h5_path=output_h5_path,
                seg_cfg=seg_cfg,
                artifact_remover_fn=artifact_remover_fn,
                neural_segmenter=neural_segmenter,
            )
            if result is None or not Path(output_h5_path).exists():
                raise RuntimeError(f"tessellation produced no patch H5 for {slide_id}")
        except Exception as exc:
            failures.append((slide_id, slide_path, output_h5_path, str(exc)))
            try:
                Path(output_h5_path).unlink(missing_ok=True)
            except Exception:
                pass
            logger.exception("Failed to tessellate %s", slide_id)
            if not cfg.continue_on_error:
                break

    if failures and cfg.failures_tsv_path:
        failure_path = Path(cfg.failures_tsv_path)
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("w") as f:
            f.write("slide_id\tslide_path\toutput_h5_path\terror\n")
            for slide_id, slide_path, output_h5_path, error in failures:
                f.write(f"{slide_id}\t{slide_path}\t{output_h5_path}\t{error}\n")

    all_slides_failed = failures and len(failures) == len(items)
    if failures and (not cfg.continue_on_error or all_slides_failed):
        raise RuntimeError(
            f"Tessellation failed for {len(failures)} of {len(items)} slide(s)."
        )
    if failures:
        logger.warning(
            "Tessellation failed for %d of %d slide(s).", len(failures), len(items)
        )


@hydra.main(version_base=None, config_path=".", config_name="tessellate_config")
def main(
    cfg: TessellateConfig,
):
    """Tile a whole slide image and perform tissue segmentation."""
    seg_cfg = OmegaConf.to_container(cfg.seg_config, resolve=True)
    neural_cfg_obj = getattr(cfg, "neural_config", NeuralSegConfig())
    neural_cfg = (
        OmegaConf.to_container(neural_cfg_obj, resolve=True)
        if OmegaConf.is_config(neural_cfg_obj)
        else vars(neural_cfg_obj)
    )
    if cfg.slide_paths is not None:
        if cfg.slide_path is not None or cfg.output_h5_path is not None:
            raise ValueError(
                "Batch mode is mutually exclusive with slide_path and output_h5_path."
            )
        _run_batch(cfg, seg_cfg, neural_config=neural_cfg)
        return

    if cfg.slide_path is None or cfg.output_h5_path is None:
        raise ValueError("Single-slide mode requires slide_path and output_h5_path.")

    artifact_remover_fn = _build_artifact_remover(seg_cfg)
    neural_segmenter = _build_neural_segmenter(
        seg_cfg, neural_config=neural_cfg
    )
    if values := _run_tessellation(
        slide_path=cfg.slide_path,
        slide_id=cfg.slide_id,
        output_h5_path=cfg.output_h5_path,
        seg_cfg=seg_cfg,
        artifact_remover_fn=artifact_remover_fn,
        neural_segmenter=neural_segmenter,
    ):
        polygon, grid, coords = values
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
