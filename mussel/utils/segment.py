import functools
import logging
import math
import multiprocessing as mp
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import shapely
import tiffslide
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform
from shapely.prepared import prep

from mussel.utils.file import save_hdf5
from mussel.utils.timer import timed
from mussel.utils.wsi_backend import open_slide as _wsi_open_slide

Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)

_SEGMENT_THRESHOLD_DEFAULT = 20
_MEDIAN_BLUR_DEFAULT = 7
_NEURAL_TARGET_MPP = 1.0
_NEURAL_MAX_AUTO_UPSCALE = 2.25
_NEURAL_MAX_UPSCALE_ENV = "MUSSEL_NEURAL_SEG_MAX_UPSCALE"


def get_slide_mpp(
    wsi,
    slide_path: Optional[str] = None,
    default_mpp: float = 0.5,
    slide_mpp_override: Optional[float] = None,
) -> float:
    """Get MPP (microns per pixel) from slide metadata with fallback handling.

    Fallback chain (first successful value wins):

    1. ``slide_mpp_override`` — explicit override; skips all metadata reading.
    2. ``tiffslide.mpp-x`` — standard tiffslide property (populated for SVS, SCN,
       and most TIFF-based formats when resolution tags are present).
    3. ``aperio.MPP`` / ``openslide.mpp-x`` — legacy vendor property names.
    4. ``tiff.XResolution`` + ``tiff.ResolutionUnit`` — raw TIFF resolution tags,
       converted to µm/px.  Tiffslide exposes these for partially-supported formats
       (NDPI, BIF, MRXS, QPTIFF, CZI) even when it cannot derive mpp-x itself.
    5. Objective-power magnification estimate — ``10.0 / magnification`` using
       ``aperio.AppMag``, ``openslide.objective-power``, or
       ``tiffslide.objective-power``.
    6. ``default_mpp`` (0.5 µm/px, typical for 20x TCGA slides).

    Args:
        wsi: TiffSlide object.
        slide_path: Optional path to slide for logging.
        default_mpp: Fallback MPP if all metadata reads fail (default: 0.5).
        slide_mpp_override: If provided, return this value directly without
            reading any metadata.

    Returns:
        MPP value as float.
    """
    if slide_mpp_override is not None:
        slide_mpp_override = float(slide_mpp_override)
        if slide_mpp_override <= 0:
            raise ValueError(
                f"slide_mpp_override must be a positive value, got {slide_mpp_override}"
            )
        logger.info(f"Using slide_mpp_override: {slide_mpp_override}")
        return slide_mpp_override

    try:
        slide_name = slide_path if slide_path else "slide"

        # Step 2: standard tiffslide property
        slide_mpp_value = wsi.properties.get(tiffslide.PROPERTY_NAME_MPP_X)

        # Step 3: vendor-specific legacy keys
        if slide_mpp_value is None:
            for key in ["aperio.MPP", "openslide.mpp-x"]:
                slide_mpp_value = wsi.properties.get(key)
                if slide_mpp_value is not None:
                    logger.info(f"Found MPP in alternate property: {key}")
                    break

        if slide_mpp_value is not None:
            slide_mpp = float(slide_mpp_value)
            logger.info(f"slide_mpp: {slide_mpp}")
            return slide_mpp

        # Step 4: raw TIFF resolution tags.
        # Tiffslide exposes tiff.XResolution / tiff.ResolutionUnit for partially-
        # supported formats (NDPI, BIF, MRXS, QPTIFF, CZI) even when it cannot
        # complete the conversion to tiffslide.mpp-x (e.g. unrecognised unit).
        x_resolution = wsi.properties.get("tiff.XResolution")
        resolution_unit = wsi.properties.get("tiff.ResolutionUnit")
        if x_resolution is not None and resolution_unit is not None:
            unit = str(resolution_unit).upper()
            scale = {
                "INCH": 25400.0,
                "CENTIMETER": 10000.0,
                "MILLIMETER": 1000.0,
                "MICROMETER": 1.0,
            }.get(unit)
            if scale is not None:
                try:
                    x_res_float = float(x_resolution)
                    if x_res_float <= 0:
                        raise ValueError(
                            f"tiff.XResolution is non-positive: {x_resolution}"
                        )
                    slide_mpp = round(scale / x_res_float, 4)
                    logger.warning(
                        f"MPP not in standard metadata for {slide_name}; "
                        f"derived from tiff.XResolution ({x_resolution} px/{unit.lower()}): "
                        f"{slide_mpp:.4f} µm/px"
                    )
                    return slide_mpp
                except (ValueError, ArithmeticError):
                    pass

        # Step 5: estimate from objective-power magnification
        magnification = None
        for key in [
            "aperio.AppMag",
            "openslide.objective-power",
            tiffslide.PROPERTY_NAME_OBJECTIVE_POWER,
        ]:
            mag_value = wsi.properties.get(key)
            if mag_value is not None:
                try:
                    magnification = float(mag_value)
                    logger.info(f"Found magnification: {magnification}x from {key}")
                    break
                except (ValueError, TypeError):
                    continue

        if magnification is not None and magnification > 0:
            slide_mpp = 10.0 / magnification
            logger.warning(
                f"MPP metadata not found for {slide_name}, "
                f"estimated from magnification ({magnification}x): {slide_mpp:.3f}"
            )
            return slide_mpp

        # Step 6: default fallback
        logger.warning(
            f"MPP metadata not found for {slide_name}, using default MPP: {default_mpp}"
        )
        return default_mpp

    except (KeyError, TypeError, ValueError) as e:
        slide_name = slide_path if slide_path else "slide"
        logger.warning(
            f"Failed to read MPP metadata for {slide_name}: {e}, using default MPP: {default_mpp}"
        )
        return default_mpp


def get_level_for_magnification(wsi, target_mag: float, fallback_level: int = 2) -> int:
    """Return the best pyramid level for a target magnification.

    Reads the slide's native MPP via :func:`get_slide_mpp`, computes the
    required downsample factor, and delegates to
    ``wsi.get_best_level_for_downsample``.

    Args:
        wsi: TiffSlide-compatible slide object.
        target_mag: Desired magnification (e.g. ``20.0`` for 20×).
        fallback_level: Pyramid level to return when MPP cannot be determined
            (default: 2).

    Returns:
        Integer pyramid level index closest to ``target_mag``.
    """
    try:
        mpp = get_slide_mpp(wsi)
        native_mag = 10.0 / mpp
        downsample = native_mag / target_mag
        return wsi.get_best_level_for_downsample(downsample)
    except Exception:
        return fallback_level


def _get_neural_seg_level(wsi, slide_mpp: float, level_downsamples) -> int:
    """Choose a pyramid level near the neural model's native resolution.

    Prefer levels at 1 µm/px or up to ~2× coarser. This avoids reading a huge
    full-resolution image when a near-target pyramid level exists, while
    preventing the very coarse 16×-style thumbnail upsampling that breaks the
    neural model's input semantics.
    """
    level_mpps = [slide_mpp * ds[0] for ds in level_downsamples]
    acceptable_coarser = [
        (idx, level_mpp)
        for idx, level_mpp in enumerate(level_mpps)
        if _NEURAL_TARGET_MPP
        <= level_mpp
        <= _NEURAL_TARGET_MPP * _NEURAL_MAX_AUTO_UPSCALE
    ]
    if acceptable_coarser:
        return min(acceptable_coarser, key=lambda item: item[1])[0]

    finer_or_equal = [
        (idx, level_mpp)
        for idx, level_mpp in enumerate(level_mpps)
        if level_mpp <= _NEURAL_TARGET_MPP
    ]
    if finer_or_equal:
        return max(finer_or_equal, key=lambda item: item[1])[0]

    return min(
        range(len(level_mpps)),
        key=lambda idx: abs(math.log(level_mpps[idx] / _NEURAL_TARGET_MPP)),
    )


def _get_neural_max_upscale() -> Optional[float]:
    value = os.environ.get(_NEURAL_MAX_UPSCALE_ENV)
    if value is None:
        return _NEURAL_MAX_AUTO_UPSCALE
    try:
        parsed = float(value)
    except ValueError:
        warnings.warn(
            f"Invalid {_NEURAL_MAX_UPSCALE_ENV} value; using default "
            f"{_NEURAL_MAX_AUTO_UPSCALE}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _NEURAL_MAX_AUTO_UPSCALE
    if parsed <= 0:
        return None
    return parsed


def _validate_neural_seg_mpp(seg_level_mpp: float, seg_level: int) -> None:
    max_upscale = _get_neural_max_upscale()
    if max_upscale is None:
        return
    max_mpp = _NEURAL_TARGET_MPP * max_upscale
    if seg_level_mpp <= max_mpp:
        return
    raise ValueError(
        f"seg_model='neural' cannot use seg_level={seg_level} at "
        f"{seg_level_mpp:.3f} µm/px because that would require "
        f"{seg_level_mpp / _NEURAL_TARGET_MPP:.1f}x upsampling to the "
        f"{_NEURAL_TARGET_MPP:.1f} µm/px neural model resolution. "
        f"Use a finer seg_level or set {_NEURAL_MAX_UPSCALE_ENV}=0 to disable this guard."
    )


def is_white_patch(patch, saturation_threshold=5):
    """
    Determine if patch is white based on HSV saturation threshold.

    Args:
        patch: RGB patch array
        saturation_threshold: Saturation threshold for white detection

    Returns:
        True if patch is white, False otherwise
    """
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    mean_saturation = np.mean(patch_hsv[:, :, 1])
    return mean_saturation < saturation_threshold


def is_black_patch(patch, rgb_threshold=40):
    """
    Determine if patch is black based on RGB threshold.

    Args:
        patch: RGB patch array
        rgb_threshold: RGB threshold for black detection

    Returns:
        True if patch is black, False otherwise
    """
    mean_rgb = np.mean(patch, axis=(0, 1))
    return np.all(mean_rgb < rgb_threshold)


def is_black_patch_S(patch, rgb_threshold=20, percentage=0.05):
    """
    Determine if percentage of patch is black.

    Args:
        patch: PIL Image patch
        rgb_threshold: RGB threshold for black detection
        percentage: Minimum percentage of black pixels required

    Returns:
        True if percentage of patch is black, False otherwise
    """
    num_pixels = patch.size[0] * patch.size[1]
    patch_array = np.array(patch)
    black_pixels = np.all(patch_array < rgb_threshold, axis=2).sum()
    return black_pixels > num_pixels * percentage


def is_white_patch_S(patch, rgb_threshold=220, percentage=0.2):
    """
    Determine if percentage of patch is white.

    Args:
        patch: PIL Image patch
        rgb_threshold: RGB threshold for white detection
        percentage: Minimum percentage of white pixels required

    Returns:
        True if percentage of patch is white, False otherwise
    """
    num_pixels = patch.size[0] * patch.size[1]
    patch_array = np.array(patch)
    white_pixels = np.all(patch_array > rgb_threshold, axis=2).sum()
    return white_pixels > num_pixels * percentage


def scale_geometry(geometry: shapely.Geometry, scale_factor: float):
    """
    scale geometry by scale factor
    """

    def scale_coords(x, y):
        """Apply scaling to coordinates.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            Tuple of scaled (x, y) coordinates.
        """
        return x * scale_factor, y * scale_factor

    return transform(scale_coords, geometry)


def contours_to_polygon(foreground_contours, hole_contours=None) -> MultiPolygon:
    """
    Merge individual contours into one MultiPolygon
    """
    polygon = MultiPolygon()

    def create_polygon(contour):
        """Create a valid shapely polygon from a contour.

        Args:
            contour: Contour array.

        Returns:
            Valid shapely Polygon or None if contour is too small.
        """
        contour = np.squeeze(contour)
        if len(contour) < 4:  # Need at least 4 coordinates
            return None
        # Convert contour to shapely polygon
        new_poly = Polygon(contour)

        # Not all polygons are shapely-valid (self intersection, etc.)
        if not new_poly.is_valid:
            # Convert invalid polygon to valid
            new_poly = new_poly.buffer(0)
        return new_poly

    for idx, contour in enumerate(foreground_contours):
        try:
            new_poly = create_polygon(contour)
        except Exception:
            logger.warning(f"Unable to create polygon from foreground contour {idx}")
            new_poly = None
        if new_poly is not None:
            polygon = polygon.union(new_poly)

    if hole_contours:
        for idx, contours in enumerate(hole_contours):
            for contour in contours:
                try:
                    new_poly = create_polygon(contour)
                except Exception:
                    logger.warning(f"Unable to create polygon from hole contour {idx}")
                    new_poly = None
                if new_poly is not None:
                    polygon = polygon.difference(new_poly)

    return polygon


def grid_bounds(geometry: shapely.Geometry, step_size: int, patch_size: int):
    """
    Create grid encompassing geometry
    """
    minx, miny, maxx, maxy = geometry.bounds
    grid_x_coords = np.arange(minx, maxx, step=step_size)
    grid_y_coords = np.arange(miny, maxy, step=step_size)
    grid = []
    for i in range(len(grid_x_coords) - 1):
        for j in range(len(grid_y_coords) - 1):
            poly_ij = Polygon(
                [
                    [grid_x_coords[i], grid_y_coords[j]],
                    [grid_x_coords[i], grid_y_coords[j] + patch_size],
                    [grid_x_coords[i] + patch_size, grid_y_coords[j] + patch_size],
                    [grid_x_coords[i] + patch_size, grid_y_coords[j]],
                ]
            )
            grid.append(poly_ij)
    return grid


def partition(geometry: shapely.Geometry, step_size: int, patch_size: int):
    """
    Partition geometry into a grid
    """
    prepared_geom = prep(geometry)
    grid = list(
        filter(prepared_geom.intersects, grid_bounds(geometry, step_size, patch_size))
    )
    return grid


def scale_contour_dim(contours, scale):
    """Scale contour dimensions by a scale factor.

    Args:
        contours: List of contour arrays.
        scale: Scale factor to apply.

    Returns:
        List of scaled contour arrays.
    """
    return [np.array(cont * scale, dtype="int32") for cont in contours]


def scale_holes_dim(contours, scale):
    """Scale hole contour dimensions by a scale factor.

    Args:
        contours: List of hole contour lists.
        scale: Scale factor to apply.

    Returns:
        List of scaled hole contour lists.
    """
    return [
        [np.array(hole * scale, dtype="int32") for hole in holes] for holes in contours
    ]


def _assert_level_downsamples(wsi):
    """Calculate level downsamples from WSI dimensions.

    Args:
        wsi: Whole slide image object.

    Returns:
        List of downsampling factors as tuples for each level.
    """
    level_downsamples = []
    dim_0 = wsi.level_dimensions[0]

    for downsample, dim in zip(wsi.level_downsamples, wsi.level_dimensions):
        estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
        (
            level_downsamples.append(estimated_downsample)
            if estimated_downsample
            != (
                downsample,
                downsample,
            )
            else level_downsamples.append((downsample, downsample))
        )

    return level_downsamples


def get_native_size(size, mpp, slide_mpp):
    """Calculate native pixel size for a desired microns-per-pixel resolution.

    Args:
        size: Desired size in pixels.
        mpp: Desired microns per pixel.
        slide_mpp: Native slide microns per pixel.

    Returns:
        Native pixel size as integer.
    """
    assert mpp >= slide_mpp - 0.01, "mpp must be greater than or equal to mpp_wsi"
    scale_factor = mpp / slide_mpp
    logger.debug(
        f"desired_mpp: {mpp:.3f}, slide_mpp: {slide_mpp:.3f}, mpp scale: {scale_factor:.3f}"
    )
    return int(round(size * scale_factor))


def _filter_contours(
    contours,
    hierarchy,
    tissue_area_threshold: int,
    hole_area_threshold: int,
    max_num_holes: int,
):
    """
    Filter contours by area.
    """
    filtered = []

    # find indices of foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    all_holes = []

    # loop through foreground contour indices
    for cont_idx in hierarchy_1:
        # actual contour
        contour = contours[cont_idx]
        # indices of holes contained in this contour (children of parent contour)
        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
        # take contour area (includes holes)
        contour_area = cv2.contourArea(contour)
        # calculate the contour area of each hole
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
        # actual area of foreground contour region
        contour_area = contour_area - np.array(hole_areas).sum()
        if contour_area == 0:
            continue
        if tuple((tissue_area_threshold,)) < tuple((contour_area,)):
            filtered.append(cont_idx)
            all_holes.append(holes)

    foreground_contours = [contours[cont_idx] for cont_idx in filtered]

    hole_contours = []

    for hole_ids in all_holes:
        unfiltered_holes = [contours[idx] for idx in hole_ids]
        unfiltered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
        # take max_n_holes largest holes by area
        unfiltered_holes = unfiltered_holes[:max_num_holes]
        filtered_holes = []

        # filter these holes
        for hole in unfiltered_holes:
            if cv2.contourArea(hole) > hole_area_threshold:
                filtered_holes.append(hole)

        hole_contours.append(filtered_holes)

    return foreground_contours, hole_contours


def _segment_tissue_neural(
    img: np.ndarray, slide_mpp: float, segmenter=None
) -> np.ndarray:
    """Generate a binary tissue mask using Mussel's native neural segmentor.

    Uses a DeepLabV3-ResNet50 model (pre-trained on histopathology slides) to
    segment tissue from background.  The model and inference pipeline are
    implemented directly in Mussel — no HEST package is required.
    PyTorch and torchvision must be installed (``uv sync --extra torch-gpu``).

    Weights are downloaded automatically from ``MahmoodLab/hest-tissue-seg``
    on HuggingFace on first use and cached in the HuggingFace cache directory.

    Args:
        img: RGB uint8 image array at the segmentation pyramid level, shape (H, W, 3).
        slide_mpp: Microns per pixel of ``img``.  Used to rescale the image to
            the model's native 1 µm/px operating resolution.

    Returns:
        Binary uint8 mask, shape (H, W), values 0 (background) or 255 (tissue).
    """
    from mussel.utils.neural_seg import NeuralTissueSegmenter

    if segmenter is None:
        segmenter = NeuralTissueSegmenter()
    return segmenter.segment(img, slide_mpp=slide_mpp)


@timed
def segment_tissue(
    slide_path: str,
    slide_id: Optional[str] = None,
    seg_level: int = -1,
    segment_threshold: int = _SEGMENT_THRESHOLD_DEFAULT,
    segment_max_value: int = 255,
    median_blur_ksize: int = _MEDIAN_BLUR_DEFAULT,
    morphology_ex_kernel: int = 0,
    use_otsu: bool = False,
    tissue_area_threshold: int = 100,
    hole_area_threshold: int = 16,
    max_num_holes: int = 10,
    patch_size: int = 256,
    mpp: float = 0.5,
    step_size: Optional[int] = None,
    ref_patch_size: int = 512,
    exclude_ids: Optional[List[int]] = None,
    keep_ids: Optional[List[int]] = None,
    output_h5_path: Optional[str] = None,
    overlap: int = 0,
    min_tissue_proportion: float = 0.0,
    remove_artifacts: bool = False,
    remove_penmarks: bool = False,
    artifact_remover_fn=None,  # Optional callable: (img, mask, mpp) -> mask
    seg_model: str = "classic",  # "classic" (HSV + manual threshold), "otsu" (HSV + Otsu threshold), or "neural" (DeepLabV3)
    slide_mpp_override: Optional[float] = None,
    neural_segmenter=None,
):
    """Segment tissue regions in a whole-slide image and generate tissue patches.

    Performs tissue segmentation using HSV color space, median filtering, and binary
    thresholding to identify tissue regions. Then partitions tissue into a grid of
    patches for downstream processing.

    Args:
        slide_path: Path to the whole-slide image file.
        slide_id: Optional identifier for the slide (defaults to filename stem).
        seg_level: Pyramid level to use for segmentation (default: -1 for auto-select).
        segment_threshold: Binary threshold value for tissue detection (default: 20).
            Only used when ``seg_model`` is ``"classic"``.
        segment_max_value: Maximum value for binary thresholding (default: 255).
            Only used when ``seg_model`` is ``"classic"`` or ``"otsu"``.
        median_blur_ksize: Kernel size for median blur filter (default: 7).
            Only used when ``seg_model`` is ``"classic"`` or ``"otsu"``.
        morphology_ex_kernel: Kernel size for morphological closing applied to the tissue
            mask (0 to disable). Applied after segmentation for all ``seg_model`` values,
            including ``"neural"``.
        use_otsu: **Deprecated** — use ``seg_model="otsu"`` instead. If ``True``,
            overrides ``seg_model`` to ``"otsu"`` with a deprecation warning.
        tissue_area_threshold: Minimum tissue contour area in requested patches (default: 100).
            A contour must cover at least this many patches (at ``patch_size`` and ``mpp``)
            to be retained.  Because the threshold is expressed in terms of the actual
            requested patch size and resolution, the value is scale-independent — it
            produces the same minimum tissue size in µm² regardless of which pyramid
            level is used for segmentation.
        hole_area_threshold: Maximum hole area in requested patches (default: 16).
        max_num_holes: Maximum number of holes allowed per tissue contour (default: 10).
        patch_size: Target patch size in pixels at desired MPP (default: 256).
        mpp: Target microns per pixel for patches (default: 0.5).
        step_size: Step size between patches in pixels at desired MPP (defaults to patch_size).
        ref_patch_size: Deprecated; no longer used for area calculations.
        exclude_ids: List of contour indices to exclude from processing.
        keep_ids: List of contour indices to keep (if empty, keeps all except excluded).
        output_h5_path: Optional path to save coordinates and attributes as HDF5.
        overlap: Patch overlap in absolute pixels (0 = no overlap). step_size = patch_size - overlap.
        min_tissue_proportion: Minimum fraction of patch area that must be tissue (0.0–1.0).
            Patches below this threshold are discarded. Defaults to 0.0 (no filtering).
        remove_artifacts: If True, apply artifact removal before patching (requires artifact_remover_fn).
        remove_penmarks: If True, apply pen mark removal before patching (requires artifact_remover_fn).
        artifact_remover_fn: Optional callable
            ``(img: np.ndarray, mask: np.ndarray, mpp: float) -> np.ndarray``
            called after morphological closing. ``img`` is the RGB thumbnail at
            ``seg_level``, ``mask`` is the binary tissue mask (uint8, 0/1), and
            ``mpp`` is the microns-per-pixel of ``img``. The callable should
            return a corrected binary mask of the same shape. Used for
            artifact/pen-mark removal.  See
            :class:`~mussel.utils.artifact_removal.GrandQCArtifactRemover` for
            a ready-made implementation.
        seg_model: Segmentation backend (default: ``"classic"``).

            - ``"classic"``: HSV colour space + median blur + fixed threshold
              (``segment_threshold``).
            - ``"otsu"``: HSV colour space + median blur + Otsu's automatic threshold.
              Ignores ``segment_threshold``.
            - ``"neural"``: DeepLabV3-ResNet50 trained on histopathology (HEST).
              Ignores ``segment_threshold`` and ``median_blur_ksize``; raises a warning
              if either is set to a non-default value. Requires PyTorch; weights are
              auto-downloaded from ``MahmoodLab/hest-tissue-seg``.

            ``morphology_ex_kernel`` applies to all three modes.
        slide_mpp_override: If set, use this value (µm/px) as the slide's native MPP
            instead of reading it from slide metadata.

    Returns:
        tuple: A 4-tuple containing:
            - polygon (shapely.geometry.MultiPolygon): Tissue regions as a multipolygon
            - grid (list): List of shapely.geometry.box objects representing patches
            - coords (list): List of (x, y) coordinates for top-left corner of each patch
            - attrs (dict): Dictionary of segmentation parameters and metadata including:
                - seg_level: Segmentation pyramid level used
                - patch_size: Native patch size in pixels
                - step_size: Native step size in pixels
                - mpp: Target microns per pixel
                - native_mpp: Slide's native MPP
                - level_dim: Dimensions at level 0
                - name: Slide identifier
                - (and other segmentation parameters)

        Returns None if no tissue contours are found or if slide dimensions are too large.
    """
    wsi = _wsi_open_slide(slide_path)

    try:
        if slide_id is None:
            slide_id = Path(slide_path).stem

        if not (0.0 <= min_tissue_proportion <= 1.0):
            raise ValueError(
                f"min_tissue_proportion must be in [0.0, 1.0], got {min_tissue_proportion}"
            )

        if exclude_ids is None:
            exclude_ids = []
        if keep_ids is None:
            keep_ids = []

        # Validate and normalise seg_model before choosing the segmentation level.
        # Neural segmentation needs a materially finer source image than the
        # classic HSV thumbnail path.
        supported_seg_models = {"classic", "otsu", "neural"}
        if seg_model is None:
            seg_model = "classic"
        elif not isinstance(seg_model, str):
            raise ValueError(f"seg_model must be a string, got {type(seg_model)!r}")
        else:
            seg_model = seg_model.strip().lower()
        if seg_model not in supported_seg_models:
            raise ValueError(
                f"Unsupported seg_model {seg_model!r}. "
                f"Supported values: {sorted(supported_seg_models)}"
            )

        # Handle deprecated use_otsu flag.
        if use_otsu:
            warnings.warn(
                "use_otsu=True is deprecated; use seg_model='otsu' instead. "
                "Overriding seg_model to 'otsu'.",
                DeprecationWarning,
                stacklevel=2,
            )
            seg_model = "otsu"

        # Get MPP with fallback handling.
        # Probe without a default first to detect whether real metadata exists.
        _mpp_probe = get_slide_mpp(
            wsi, slide_path, default_mpp=None, slide_mpp_override=slide_mpp_override
        )
        slide_mpp = (
            _mpp_probe
            if _mpp_probe is not None
            else get_slide_mpp(wsi, slide_path, slide_mpp_override=slide_mpp_override)
        )
        # True when no MPP metadata was found and the 0.5 µm/px default was used.
        mpp_is_fallback = _mpp_probe is None and slide_mpp_override is None

        level_downsamples = _assert_level_downsamples(wsi)

        if seg_level < 0:
            if len(wsi.level_dimensions) == 1:
                seg_level = 0
            elif seg_model == "neural":
                seg_level = _get_neural_seg_level(wsi, slide_mpp, level_downsamples)
            else:
                seg_level = wsi.get_best_level_for_downsample(64)

        logger.info(f"Using level {seg_level} for segmentation")
        width, height = wsi.level_dimensions[seg_level]
        if width * height > 1e12:
            logger.error(
                "level_dim {} x {} is likely too large for successful segmentation, aborting".format(
                    width, height
                )
            )
            return None

        if seg_model == "neural":
            seg_level_ds = level_downsamples[seg_level][0]
            seg_level_mpp = slide_mpp * seg_level_ds
            _validate_neural_seg_mpp(seg_level_mpp, seg_level)

        if step_size is None:
            if overlap < 0:
                raise ValueError(f"overlap must be non-negative, got {overlap}")
            if overlap > 0:
                step_size = patch_size - overlap
                if step_size <= 0:
                    raise ValueError(
                        f"overlap ({overlap}) must be less than patch_size ({patch_size})"
                    )
            else:
                step_size = patch_size
        elif overlap > 0:
            raise ValueError(
                f"Cannot specify both step_size ({step_size}) and overlap ({overlap}). "
                "Use overlap to derive step_size automatically, or pass step_size directly."
            )

        native_step_size = get_native_size(step_size, mpp, slide_mpp)
        native_patch_size = get_native_size(patch_size, mpp, slide_mpp)
        logger.info(f"native_step_size: {native_step_size}")
        logger.info(f"native_patch_size: {native_patch_size}")

        img = np.array(
            wsi.read_region((0, 0), seg_level, wsi.level_dimensions[seg_level])
        )
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]

        # Warn about classic-only parameters that are ignored by the neural backend.
        if seg_model == "neural":
            ignored = []
            if segment_threshold != _SEGMENT_THRESHOLD_DEFAULT:
                ignored.append(f"segment_threshold={segment_threshold!r}")
            if median_blur_ksize != _MEDIAN_BLUR_DEFAULT:
                ignored.append(f"median_blur_ksize={median_blur_ksize!r}")
            if ignored:
                logger.warning(
                    "seg_model='neural' ignores classic-only parameters: %s. "
                    "These values have no effect.",
                    ", ".join(ignored),
                )

        if seg_model == "neural":
            # The img is read at seg_level. Compute its actual MPP so that
            # NeuralTissueSegmenter can rescale to the model's 1 µm/px target.
            tissue_mask = _segment_tissue_neural(
                img, seg_level_mpp, segmenter=neural_segmenter
            )
        else:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
            img_med = cv2.medianBlur(
                img_hsv[:, :, 1], median_blur_ksize
            )  # Apply median blurring

            # Thresholding
            if seg_model == "otsu":
                _, tissue_mask = cv2.threshold(
                    img_med, 0, segment_max_value, cv2.THRESH_OTSU + cv2.THRESH_BINARY
                )
            else:
                _, tissue_mask = cv2.threshold(
                    img_med, segment_threshold, segment_max_value, cv2.THRESH_BINARY
                )

        # Morphological closing — applied regardless of seg_model.
        if morphology_ex_kernel > 0:
            kernel = np.ones((morphology_ex_kernel, morphology_ex_kernel), np.uint8)
            tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

        # Optional artifact/pen mark removal via pluggable callable.
        # Compute the thumbnail's MPP so the remover can rescale to its model's
        # native resolution (e.g. GrandQC expects 10× / 1 µm-per-pixel input).
        if artifact_remover_fn is not None:
            if remove_artifacts or remove_penmarks:
                img_mpp = slide_mpp * level_downsamples[seg_level][0]
                remover_img = img
                remover_mask = tissue_mask
                remover_mpp = img_mpp

                # If the seg-level thumbnail is too coarse for the remover (e.g.
                # GrandQC requires ≤ 8 µm/px but the lowest pyramid level is
                # 8–16 µm/px), read a separate thumbnail at a finer pyramid level.
                max_mpp = getattr(artifact_remover_fn, "max_input_mpp", None)
                if max_mpp is not None and img_mpp >= max_mpp:
                    target_ds = max_mpp / slide_mpp
                    artifact_level = wsi.get_best_level_for_downsample(target_ds)
                    artifact_level_mpp = (
                        slide_mpp * level_downsamples[artifact_level][0]
                    )
                    if artifact_level_mpp <= max_mpp:
                        remover_img = np.array(
                            wsi.read_region(
                                (0, 0),
                                artifact_level,
                                wsi.level_dimensions[artifact_level],
                            )
                        )
                        remover_mpp = artifact_level_mpp
                        ah, aw = remover_img.shape[:2]
                        remover_mask = cv2.resize(
                            tissue_mask, (aw, ah), interpolation=cv2.INTER_NEAREST
                        )
                        logger.info(
                            "Artifact removal: using level %d (%.2f µm/px) instead of "
                            "seg level %d (%.2f µm/px)",
                            artifact_level,
                            remover_mpp,
                            seg_level,
                            img_mpp,
                        )

                pre_removal_mask = tissue_mask.copy()
                result_mask = artifact_remover_fn(
                    remover_img, remover_mask, remover_mpp
                )

                # Resize result back to seg-level dimensions if needed.
                if result_mask.shape != tissue_mask.shape:
                    sh, sw = tissue_mask.shape[:2]
                    result_mask = cv2.resize(
                        result_mask.astype(np.uint8),
                        (sw, sh),
                        interpolation=cv2.INTER_NEAREST,
                    )

                # Fallback: if artifact removal eliminated most or all tissue,
                # revert to the pre-removal mask.  Over-aggressive removal can
                # occur when the GrandQC model is out-of-distribution for a
                # particular tissue type (e.g. necrotic CNS tumours).
                _MIN_TISSUE_SURVIVAL = (
                    0.10  # at least 10% of original tissue must remain
                )
                pre_pixels = int(np.count_nonzero(pre_removal_mask))
                post_pixels = int(np.count_nonzero(result_mask))
                removal_fraction = (
                    1.0 - post_pixels / pre_pixels if pre_pixels > 0 else 1.0
                )
                if removal_fraction >= 1.0 - _MIN_TISSUE_SURVIVAL:
                    logger.warning(
                        "Artifact removal eliminated %.0f%% of tissue for slide %s "
                        "(threshold: revert if >=%.0f%% removed). "
                        "Falling back to pre-removal mask. "
                        "Consider using artifact_exclude_classes=[4, 7] (pen marks only) "
                        "or disabling remove_artifacts for this slide type.",
                        removal_fraction * 100,
                        slide_id,
                        (1.0 - _MIN_TISSUE_SURVIVAL) * 100,
                    )
                    tissue_mask = pre_removal_mask
                else:
                    logger.info(
                        "Artifact removal: %.0f%% of tissue retained for slide %s.",
                        (1.0 - removal_fraction) * 100,
                        slide_id,
                    )
                    tissue_mask = result_mask
            else:
                logger.warning(
                    "artifact_remover_fn was provided but neither remove_artifacts nor "
                    "remove_penmarks is True. The function will not be called. Set "
                    "remove_artifacts=True or remove_penmarks=True to enable removal."
                )
        elif remove_artifacts or remove_penmarks:
            logger.warning(
                "remove_artifacts/remove_penmarks flags are set but no artifact_remover_fn "
                "was provided. Flags will have no effect. Pass artifact_remover_fn to use "
                "artifact removal."
            )

        scale = level_downsamples[seg_level]
        # tissue_area_threshold / hole_area_threshold are in units of requested patches.
        # Convert to seg-level pixel area using the native patch size (derived from
        # patch_size and mpp) rather than the legacy ref_patch_size.  This makes the
        # threshold truly scale-independent: the same threshold value produces the same
        # minimum-tissue-size in µm² regardless of which pyramid level is used for
        # segmentation.
        native_patch_area = native_patch_size**2
        seg_patch_area = int(native_patch_area / (scale[0] * scale[1]))
        tissue_area_threshold *= seg_patch_area
        hole_area_threshold *= seg_patch_area

        # Find and filter contours
        contours, hierarchy = cv2.findContours(
            tissue_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )  # Find contours
        if contours is None or hierarchy is None:
            logger.warning(
                "No contours found for slide %s (tissue mask may be empty after artifact removal or segmentation).",
                slide_id,
            )
            return None
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        foreground_contours, hole_contours = _filter_contours(
            contours,
            hierarchy,
            tissue_area_threshold,
            hole_area_threshold,
            max_num_holes,
        )  # Necessary for filtering out artifacts

        contours_tissue = scale_contour_dim(foreground_contours, scale)
        holes_tissue = scale_holes_dim(hole_contours, scale)

        # exclude_ids = [0,7,9]
        if len(keep_ids) > 0:
            contour_ids = set(keep_ids) - set(exclude_ids)
        else:
            contour_ids = set(np.arange(len(contours_tissue))) - set(exclude_ids)

        contours_tissue = [contours_tissue[i] for i in contour_ids]
        holes_tissue = [holes_tissue[i] for i in contour_ids]

        logger.info(f"Creating patches for: {slide_id} ...")
        if contours_tissue is None or len(contours_tissue) == 0:
            logger.info("0 contours found")
            return None

        n_contours = len(contours_tissue)
        logger.info(f"Total number of contours: {n_contours}")

        polygon = contours_to_polygon(contours_tissue, holes_tissue)
        grid = partition(polygon, native_step_size, native_patch_size)
        coords = [g.exterior.coords[0] for g in grid]
        logger.info(f"Total number of patches: {len(coords)}")

        if min_tissue_proportion > 0.0:
            prepared_polygon = prep(polygon)
            filtered = [
                (g, c)
                for g, c in zip(grid, coords)
                if prepared_polygon.intersects(g)
                and prepared_polygon.intersection(g).area / g.area
                >= min_tissue_proportion
            ]
            if filtered:
                grid, coords = zip(*filtered)
                grid, coords = list(grid), list(coords)
            else:
                grid, coords = [], []
            logger.info(
                f"After min_tissue_proportion={min_tissue_proportion:.2f} filter: "
                f"{len(coords)} patches remaining"
            )

        attrs = {
            "seg_level": seg_level,
            "segment_threshold": segment_threshold,
            "segment_max_value": segment_max_value,
            "median_blur_ksize": median_blur_ksize,
            "morphology_ex_kernel": morphology_ex_kernel,
            "tissue_area_threshold": tissue_area_threshold,
            "hole_area_threshold": hole_area_threshold,
            "max_num_holes": max_num_holes,
            "ref_patch_size": ref_patch_size,
            "patch_size": native_patch_size,
            "step_size": native_step_size,
            "patch_size_to_resize_to_for_desired_mpp": patch_size,
            "patch_level": 0,
            "mpp": mpp,
            "native_mpp": slide_mpp,
            "mpp_is_fallback": mpp_is_fallback,
            "level_dim": wsi.level_dimensions[0],
            "name": slide_id,
            "overlap": overlap,
            "min_tissue_proportion": min_tissue_proportion,
            "seg_model": seg_model,
        }
        if output_h5_path:
            asset_dict = {"coords": np.array(coords, dtype=np.int64)}
            attr_dict = {"coords": attrs}
            save_hdf5(output_h5_path, asset_dict, attr_dict, mode="w")
            logger.info(f"Writing to {output_h5_path}")

        return polygon, grid, coords, attrs
    finally:
        wsi.close()


def draw_slide_mask(
    slide_path: str,
    polygons: shapely.Geometry | List[shapely.Geometry],
    vis_level=0,
    outline="black",
    outline_width=1,
    fill=(255, 0, 0, 80),
    max_size=None,
    custom_downsample=None,
):
    """
    Draw slide mask with polygon contours or list of grid polygons
    """
    wsi = _wsi_open_slide(slide_path)

    try:
        if vis_level < 0:
            if len(wsi.level_dimensions) == 1:
                vis_level = 0
            else:
                vis_level = wsi.get_best_level_for_downsample(64)

        if type(polygons) != list:
            polygons = [polygons]

        level_downsamples = _assert_level_downsamples(wsi)
        downsample = level_downsamples[vis_level]
        scale = [1 / downsample[0], 1 / downsample[1]]
        region_size = wsi.level_dimensions[vis_level]

        img = np.array(wsi.read_region((0, 0), vis_level, region_size).convert("RGB"))

        img = Image.fromarray(img)

        draw = ImageDraw.Draw(img, "RGBA")

        for polygon in polygons:
            scaled_polygon = scale_geometry(polygon, scale[0])
            if isinstance(polygon, MultiPolygon):
                for geom in scaled_polygon.geoms:
                    draw.polygon(
                        geom.exterior.coords,
                        outline=outline,
                        fill=fill,
                        width=outline_width,
                    )
            else:
                draw.polygon(
                    scaled_polygon.exterior.coords,
                    outline=outline,
                    fill=fill,
                    width=outline_width,
                )

        image_width, image_height = img.size
        if custom_downsample and custom_downsample > 1:
            img = img.resize(
                (
                    int(image_width / custom_downsample),
                    int(image_height / custom_downsample),
                )
            )

        if max_size is not None and (image_width > max_size or image_height > max_size):
            resize_factor = (
                max_size / image_width
                if image_width > image_height
                else max_size / image_height
            )
            img = img.resize(
                (int(image_width * resize_factor), int(image_height * resize_factor))
            )

        return img
    finally:
        # Always close WSI file handle, even if exception occurs
        wsi.close()


def _save_patch_png(coord, img, save_dir):
    """
    Save patch as png
    """
    file_path = save_dir / f"{int(coord[0])}_{int(coord[1])}.png"
    img.save(file_path, "png")


def get_patch_generator(
    wsi,
    coords,
    patch_level,
    patch_size,
    filter_black_white=True,
    white_threshold=15,
    black_threshold=50,
):
    """
    Generate patches at specified coordinates
    """
    for coord in coords:
        img = wsi.read_region(coord, patch_level, (patch_size, patch_size)).convert(
            "RGB"
        )
        img_arr = np.array(img)
        if filter_black_white and (
            is_black_patch(img_arr, black_threshold)
            or is_white_patch(img_arr, white_threshold)
        ):
            continue
        yield coord, img


def save_patches_png(
    slide_path: str,
    coords: list,
    save_dir: str,
    mpp=0.5,
    patch_size=256,
    filter_black_white=True,
    white_threshold=15,
    black_threshold=50,
    num_workers=4,
    slide_mpp_override: Optional[float] = None,
):
    """
    Save patches as png
    """
    wsi = _wsi_open_slide(slide_path)
    pool = None

    try:
        # Get MPP with fallback handling
        slide_mpp = get_slide_mpp(
            wsi, slide_path, slide_mpp_override=slide_mpp_override
        )

        native_patch_size = get_native_size(patch_size, mpp, slide_mpp)

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        pool = mp.Pool(num_workers)
        patch_gen = get_patch_generator(
            wsi,
            coords,
            patch_level=0,
            patch_size=native_patch_size,
            filter_black_white=filter_black_white,
            black_threshold=black_threshold,
            white_threshold=white_threshold,
        )
        pool.starmap(
            functools.partial(
                _save_patch_png,
                save_dir=save_dir,
            ),
            patch_gen,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        wsi.close()
