import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import geopandas as gpd
import h5py
import hydra
import numpy as np
import pandas as pd
import yaml
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

logger = logging.getLogger(__name__)
from PIL import Image
from shapely.geometry import MultiPolygon
from shapely import box as shapely_box

from mussel.utils.segment import contours_to_polygon

Image.MAX_IMAGE_PIXELS = None


@dataclass
class MergeAnnotationFeaturesConfig:
    """
    features_h5_path (str): Path to the HDF5 file containing tile features.
    annotation_bmp_path (str): Path to the BMP file containing annotations.
    output_parquet_path (str): Path to save the merged results in Parquet format.
    slide_id (Optional[str]): Optional slide identifier to include in the output.
    class_mapping_yaml_path (Optional[str]): Optional path to a YAML file for class mapping.
    """

    features_h5_path: str = MISSING
    annotation_bmp_path: str = MISSING
    output_parquet_path: str = MISSING
    slide_id: Optional[str] = None
    class_mapping_yaml_path: Optional[str] = None


desc_doc = """== ${hydra.help.app_name} ==
Merges tile features with annotations from a BMP file. It reads features from an HDF5 file, processes annotations, and saves the merged results in Parquet format.
"""

parameter_doc = f"""
== Available Parameters ==
{MergeAnnotationFeaturesConfig.__doc__}
"""

cs = ConfigStore.instance()
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=desc_doc, footer=parameter_doc)),
    provider="hydra",
)
cs.store(name="merge_annotation_features_config", node=MergeAnnotationFeaturesConfig)


@hydra.main(
    version_base=None, config_path=".", config_name="merge_annotation_features_config"
)
def main(cfg: MergeAnnotationFeaturesConfig):
    """Merge tile features with annotations from a BMP file."""
    with open(cfg.features_h5_path, "rb") as f:
        logger.info(f"Reading features from {cfg.features_h5_path}...")
        tiles_h5 = h5py.File(f, "r")
        tile_size = tiles_h5["coords"].attrs.get("patch_size")

        coords = np.array(tiles_h5["coords"])  # shape (N, 2): [i, j]
        i_arr, j_arr = coords[:, 0], coords[:, 1]
        tiles = shapely_box(i_arr, j_arr, i_arr + tile_size, j_arr + tile_size)
        tiles_gdf = gpd.GeoDataFrame(
            pd.DataFrame(tiles_h5["features"]).add_prefix("feature_", axis=1),
            geometry=tiles,
        )
        tiles_gdf["tile_size"] = tile_size
        tiles_gdf["tile_area"] = tiles_gdf.area
        logger.info(f"Loaded {len(tiles_gdf)} tiles")

    logger.info(f"Reading annotations from {cfg.annotation_bmp_path}...")
    img_arr = np.array(Image.open(cfg.annotation_bmp_path))

    class_mapping = None
    if cfg.class_mapping_yaml_path is not None:
        with open(cfg.class_mapping_yaml_path, "r") as f:
            class_mapping = yaml.safe_load(f)

    # When class_mapping is used, iterate over raw (original) classes so each
    # raw annotation region gets its own polygon, then store the mapped label.
    # Without class_mapping, iterate over pixel values directly.
    raw_classes = np.unique(img_arr)

    gdfs = []
    for clss in raw_classes:
        if clss == 0:
            # Pixel value 0 is the unannotated background; always skip.
            continue
        class_img_arr = (img_arr == clss).astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            class_img_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        class_polygon = contours_to_polygon(contours)
        if isinstance(class_polygon, MultiPolygon):
            class_gdf = gpd.GeoDataFrame(geometry=list(class_polygon.geoms))
        else:
            class_gdf = gpd.GeoDataFrame(geometry=[class_polygon])
        if class_mapping is not None:
            annotation_val = class_mapping.get(int(clss))
            if annotation_val is None:
                logger.debug(f"Skipping pixel value {clss} not found in class_mapping")
                continue
        else:
            annotation_val = clss
        class_gdf = class_gdf.assign(annotation=annotation_val)
        class_gdf = class_gdf.assign(annotation_area=class_gdf.area)
        gdfs.append(class_gdf)
    gdf = pd.concat(gdfs)
    if len(gdf) == 0:
        logger.info("No annotated tiles found")
        return

    gdf = tiles_gdf.overlay(gdf, how="intersection")
    gdf = gdf.assign(overlap_area=gdf.area)

    logger.info(f"Writing merged results to {cfg.output_parquet_path}")
    if cfg.slide_id is not None:
        gdf = gdf.assign(slide_id=cfg.slide_id)

    gdf.to_parquet(cfg.output_parquet_path)


if __name__ == "__main__":
    main()
