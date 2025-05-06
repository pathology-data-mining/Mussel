import functools
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import shapely
import tiffslide
from loguru import logger
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform
from shapely.prepared import prep

from mussel.utils.timer import timed
from mussel.utils.file import save_hdf5

Image.MAX_IMAGE_PIXELS = None

def is_white_patch(patch, satThresh=5):
    """
    Determine if patch is white
    """
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return True if np.mean(patch_hsv[:, :, 1]) < satThresh else False


def is_black_patch(patch, rgbThresh=40):
    """
    Determine if patch is black
    """
    return True if np.all(np.mean(patch, axis=(0, 1)) < rgbThresh) else False


def is_black_patch_S(patch, rgbThresh=20, percentage=0.05):
    """
    Determine if percentage of patch is black
    """
    num_pixels = patch.size[0] * patch.size[1]
    return (
        True
        if np.all(np.array(patch) < rgbThresh, axis=(2)).sum() > num_pixels * percentage
        else False
    )


def is_white_patch_S(patch, rgbThresh=220, percentage=0.2):
    """
    Determine if percentage of patch is white
    """
    num_pixels = patch.size[0] * patch.size[1]
    return (
        True
        if np.all(np.array(patch) > rgbThresh, axis=(2)).sum() > num_pixels * percentage
        else False
    )


def scale_geometry(geometry: shapely.Geometry, scale_factor: float):
    """
    scale geometry by scale factor
    """

    def scale_coords(x, y):
        return x * scale_factor, y * scale_factor

    return transform(scale_coords, geometry)


def contours_to_polygon(foreground_contours, hole_contours) -> MultiPolygon:
    """
    Merge individual contours into one MultiPolygon
    """
    polygon = MultiPolygon()

    def create_polygon(contour):
        contour = np.squeeze(contour)
        if len(contour) < 2:
            return None
        # Convert contour to shapely polygon
        new_poly = Polygon(contour)

        # Not all polygons are shapely-valid (self intersection, etc.)
        if not new_poly.is_valid:
            # Convert invalid polygon to valid
            new_poly = new_poly.buffer(0)
        return new_poly

    for contour in foreground_contours:
        new_poly = create_polygon(contour)
        if new_poly is not None:
            polygon = polygon.union(new_poly)

    for contours in hole_contours:
        for contour in contours:
            new_poly = create_polygon(contour)
            if new_poly is not None:
                polygon = polygon.difference(new_poly)

    return polygon


def grid_bounds(geometry: shapely.geometry, step_size: int, patch_size: int):
    """
    Create grid encompassing geometry
    """
    minx, miny, maxx, maxy = geometry.bounds
    gx = np.arange(minx, maxx, step=step_size)
    gy = np.arange(miny, maxy, step=step_size)
    # x_coords, y_coords = np.meshgrid(x_range, y_range, indexing="ij")
    # gx, gy = np.linspace(minx,maxx,nx), np.linspace(miny,maxy,ny)
    grid = []
    for i in range(len(gx) - 1):
        for j in range(len(gy) - 1):
            poly_ij = Polygon(
                [
                    [gx[i], gy[j]],
                    [gx[i], gy[j] + patch_size],
                    [gx[i] + patch_size, gy[j] + patch_size],
                    [gx[i] + patch_size, gy[j]],
                ]
            )
            grid.append(poly_ij)
    return grid


def partition(geometry: shapely.geometry, step_size: int, patch_size: int):
    """
    Partition geometry into a grid
    """
    prepared_geom = prep(geometry)
    grid = list(
        filter(prepared_geom.intersects, grid_bounds(geometry, step_size, patch_size))
    )
    return grid


def scale_contour_dim(contours, scale):
    return [np.array(cont * scale, dtype="int32") for cont in contours]


def scale_holes_dim(contours, scale):
    return [
        [np.array(hole * scale, dtype="int32") for hole in holes] for holes in contours
    ]


def _assert_level_downsamples(wsi):
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
    Filter contours by: area.
    """
    filtered = []

    # find indices of foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    all_holes = []

    # loop through foreground contour indices
    for cont_idx in hierarchy_1:
        # actual contour
        cont = contours[cont_idx]
        # indices of holes contained in this contour (children of parent contour)
        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
        # take contour area (includes holes)
        a = cv2.contourArea(cont)
        # calculate the contour area of each hole
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
        # actual area of foreground contour region
        a = a - np.array(hole_areas).sum()
        if a == 0:
            continue
        if tuple((tissue_area_threshold,)) < tuple((a,)):
            filtered.append(cont_idx)
            all_holes.append(holes)

    foreground_contours = [contours[cont_idx] for cont_idx in filtered]

    hole_contours = []

    for hole_ids in all_holes:
        unfiltered_holes = [contours[idx] for idx in hole_ids]
        unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
        # take max_n_holes largest holes by area
        unfilered_holes = unfilered_holes[:max_num_holes]
        filtered_holes = []

        # filter these holes
        for hole in unfilered_holes:
            if cv2.contourArea(hole) > hole_area_threshold:
                filtered_holes.append(hole)

        hole_contours.append(filtered_holes)

    return foreground_contours, hole_contours


@timed
def segment_tissue(
    wsi: tiffslide.TiffSlide,
    slide_id: str,
    seg_level: int = 0,
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

    if step_size is None:
        step_size = patch_size

    # get mpp of WSI
    slide_mpp = float(wsi.properties[tiffslide.PROPERTY_NAME_MPP_X])
    logger.info(f"slide_mpp: {slide_mpp}")

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

    if output_h5_path:
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

        asset_dict = {"coords": np.array(coords)}
        attr_dict = {"coords": attrs}
        save_hdf5(output_h5_path, asset_dict, attr_dict, mode="w")
        logger.info(f"Writing to {output_h5_path}")

    return polygon, grid, coords


def draw_slide_mask(
    wsi,
    polygons: shapely.geometry | List[shapely.geometry],
    vis_level=0,
    outline="black",
    fill=(255, 0, 0, 80),
    max_size=None,
    custom_downsample=None,
):
    """
    Draw slide mask with polygon contours or list of grid polygons
    """

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
                draw.polygon(geom.exterior.coords, outline="black", fill=fill)
        else:
            draw.polygon(scaled_polygon.exterior.coords, outline="black", fill=fill)

    w, h = img.size
    if custom_downsample and custom_downsample > 1:
        img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

    if max_size is not None and (w > max_size or h > max_size):
        resizeFactor = max_size / w if w > h else max_size / h
        img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

    return img


def _save_patch_png(coord, img, save_dir):
    """
    Save patch as png
    """
    file_path = save_dir / f"{int(coord[0])}_{int(coord[1])}.png"
    img.save(file_path, "png")


def get_patch_generator(wsi, coords, patch_level, patch_size, filter_black_white=True, white_threshold=15, black_threshold=50):
    """
    Generate patches at specified coordinates
    """
    for coord in coords:
        img = wsi.read_region(coord, patch_level, (patch_size, patch_size)).convert(
            "RGB"
        )
        img_arr = np.array(img)
        if filter_black_white and (is_black_patch(img_arr, black_threshold) or is_white_patch(img_arr, white_threshold)):
            continue
        yield coord, img


def save_patches_png(
    wsi: tiffslide.TiffSlide,
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

    slide_mpp = float(wsi.properties[tiffslide.PROPERTY_NAME_MPP_X])
    native_patch_size = get_native_size(patch_size, mpp, slide_mpp)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pool = mp.Pool(num_workers)
    patch_gen = get_patch_generator(
        wsi, coords, patch_level=0, patch_size=native_patch_size,
        filter_black_white=filter_black_white, black_threshold=black_threshold, white_threshold=white_threshold
    )
    pool.starmap(
        functools.partial(
            _save_patch_png,
            save_dir=save_dir,
        ),
        patch_gen,
    )
    pool.close()
