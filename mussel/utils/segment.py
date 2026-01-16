import functools
import logging
import multiprocessing as mp
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

Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)


def get_slide_mpp(wsi, slide_path: Optional[str] = None, default_mpp: float = 0.5) -> float:
    """
    Get MPP (microns per pixel) from slide metadata with fallback handling.
    
    Args:
        wsi: TiffSlide object
        slide_path: Optional path to slide for logging
        default_mpp: Default MPP to use if metadata not found (default: 0.5 for 20x TCGA slides)
        
    Returns:
        MPP value as float
    """
    try:
        # Try standard tiffslide property first
        slide_mpp_value = wsi.properties.get(tiffslide.PROPERTY_NAME_MPP_X)
        
        # If not found, try alternative property names
        if slide_mpp_value is None:
            # Try common alternative property names
            for key in ['tiffslide.mpp-x', 'aperio.MPP', 'openslide.mpp-x']:
                slide_mpp_value = wsi.properties.get(key)
                if slide_mpp_value is not None:
                    logger.info(f"Found MPP in alternate property: {key}")
                    break
        
        if slide_mpp_value is None:
            # Try to estimate MPP from magnification if available
            magnification = None
            for key in ['aperio.AppMag', 'openslide.objective-power', 'tiffslide.objective-power']:
                mag_value = wsi.properties.get(key)
                if mag_value is not None:
                    try:
                        magnification = float(mag_value)
                        logger.info(f"Found magnification: {magnification}x from {key}")
                        break
                    except (ValueError, TypeError):
                        continue
            
            if magnification is not None:
                # Estimate MPP from magnification using standard conversion
                # Typical values: 40x -> 0.25 MPP, 20x -> 0.5 MPP, 10x -> 1.0 MPP
                slide_mpp = 10.0 / magnification
                slide_name = slide_path if slide_path else "slide"
                logger.warning(f"MPP metadata not found for {slide_name}, estimated from magnification ({magnification}x): {slide_mpp:.3f}")
            else:
                # Use default MPP (common for TCGA slides at 20x magnification)
                slide_mpp = default_mpp
                slide_name = slide_path if slide_path else "slide"
                logger.warning(f"MPP metadata not found for {slide_name}, using default MPP: {slide_mpp}")
        else:
            slide_mpp = float(slide_mpp_value)
            logger.info(f"slide_mpp: {slide_mpp}")
            
        return slide_mpp
        
    except (KeyError, TypeError, ValueError) as e:
        # Fallback to default MPP if property is missing or invalid
        slide_name = slide_path if slide_path else "slide"
        logger.warning(f"Failed to read MPP metadata for {slide_name}: {e}, using default MPP: {default_mpp}")
        return default_mpp


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


@timed
def segment_tissue(
    slide_path: str,
    slide_id: Optional[str] = None,
    seg_level: int = -1,
    segment_threshold: int = 20,
    segment_max_value: int = 255,
    median_blur_ksize: int = 7,
    morphology_ex_kernel: int = 0,
    use_otsu: bool = False,
    tissue_area_threshold: int = 100,
    hole_area_threshold: int = 16,
    max_num_holes: int = 10,
    patch_size: int = 256,
    mpp: float = 0.5,
    step_size: Optional[int] = None,
    ref_patch_size: int = 512,
    exclude_ids: List[int] = [],
    keep_ids: List[int] = [],
    output_h5_path: Optional[str] = None,
):
    """
    Segment the tissue via HSV -> Median thresholding -> Binary threshold
    """
    wsi = tiffslide.open_slide(slide_path)
    if slide_id is None:
        slide_id = Path(slide_path).stem

    if seg_level < 0:
        if len(wsi.level_dimensions) == 1:
            seg_level = 0
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
        return

    if step_size is None:
        step_size = patch_size

    # Get MPP with fallback handling
    slide_mpp = get_slide_mpp(wsi, slide_path)

    native_step_size = get_native_size(step_size, mpp, slide_mpp)
    native_patch_size = get_native_size(patch_size, mpp, slide_mpp)
    logger.info(f"native_step_size: {native_step_size}")
    logger.info(f"native_patch_size: {native_patch_size}")

    img = np.array(wsi.read_region((0, 0), seg_level, wsi.level_dimensions[seg_level]))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
    img_med = cv2.medianBlur(
        img_hsv[:, :, 1], median_blur_ksize
    )  # Apply median blurring

    # Thresholding
    if use_otsu:
        _, img_otsu = cv2.threshold(
            img_med, 0, segment_max_value, cv2.THRESH_OTSU + cv2.THRESH_BINARY
        )
    else:
        _, img_otsu = cv2.threshold(
            img_med, segment_threshold, segment_max_value, cv2.THRESH_BINARY
        )

    # Morphological closing
    if morphology_ex_kernel > 0:
        kernel = np.ones((morphology_ex_kernel, morphology_ex_kernel), np.uint8)
        img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

    level_downsamples = _assert_level_downsamples(wsi)
    scale = level_downsamples[seg_level]
    scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
    tissue_area_threshold *= scaled_ref_patch_area
    hole_area_threshold *= scaled_ref_patch_area

    # Find and filter contours
    contours, hierarchy = cv2.findContours(
        img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )  # Find contours
    if contours is None or hierarchy is None:
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

    attrs = {
        "seg_level": seg_level,
        "segment_threshold": segment_threshold,
        "segment_max_value": segment_max_value,
        "median_blur_ksize": median_blur_ksize,
        "morphology_ex_kernel": morphology_ex_kernel,
        "use_otsu": use_otsu,
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
        "level_dim": wsi.level_dimensions[0],
        "name": slide_id,
    }
    if output_h5_path:
        asset_dict = {"coords": np.array(coords)}
        attr_dict = {"coords": attrs}
        save_hdf5(output_h5_path, asset_dict, attr_dict, mode="w")
        logger.info(f"Writing to {output_h5_path}")

    wsi.close()

    return polygon, grid, coords, attrs


def draw_slide_mask(
    slide_path: str,
    polygons: shapely.Geometry | List[shapely.Geometry],
    vis_level=0,
    outline="black",
    fill=(255, 0, 0, 80),
    max_size=None,
    custom_downsample=None,
):
    """
    Draw slide mask with polygon contours or list of grid polygons
    """
    wsi = tiffslide.open_slide(slide_path)

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
                draw.polygon(geom.exterior.coords, outline=outline, fill=fill)
        else:
            draw.polygon(scaled_polygon.exterior.coords, outline=outline, fill=fill)

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

    wsi.close()

    return img


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
):
    """
    Save patches as png
    """
    wsi = tiffslide.open_slide(slide_path)

    # Get MPP with fallback handling
    slide_mpp = get_slide_mpp(wsi, slide_path)
    
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
    pool.close()
    wsi.close()
