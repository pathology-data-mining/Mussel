from dataclasses import dataclass
from typing import Optional

import cv2
import geopandas as gpd
import h5py
import hydra
import numpy as np
import pandas as pd
import yaml
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import MISSING, OmegaConf
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon

from mussel.utils.segment import contours_to_polygon

Image.MAX_IMAGE_PIXELS = None


@dataclass
class MergeAnnotationFeaturesConfig:
    features_h5_path: str = MISSING
    annotation_bmp_path: str = MISSING
    output_parquet_path: str = MISSING
    slide_id: Optional[str] = None
    class_mapping_yaml_path: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="merge_annotation_features_config", node=MergeAnnotationFeaturesConfig)


@hydra.main(
    version_base=None, config_path=".", config_name="merge_annotation_features_config"
)
def main(cfg: MergeAnnotationFeaturesConfig):
    with open(cfg.features_h5_path, "rb") as f:
        logger.info(f"Reading features from {cfg.features_h5_path}...")
        tiles_h5 = h5py.File(f, "r")
        tile_size = tiles_h5["coords"].attrs.get("patch_size")

        tiles_df = pd.DataFrame(tiles_h5["coords"], columns=["i", "j"])
        tiles = [
            Polygon(
                [
                    [row.i, row.j],
                    [row.i, row.j + tile_size],
                    [row.i + tile_size, row.j + tile_size],
                    [row.i + tile_size, row.j],
                ]
            )
            for row in tiles_df.itertuples(index=False)
        ]
        tiles_gdf = gpd.GeoDataFrame(
            pd.DataFrame(tiles_h5["features"]).add_prefix("feature_", axis=1),
            geometry=tiles,
        )
        tiles_gdf["tile_size"] = tile_size
        tiles_gdf["tile_area"] = tiles_gdf.area
        logger.info(f"Loaded {len(tiles_gdf)} tiles")

    logger.info(f"Reading annotations from {cfg.annotation_bmp_path}...")
    img_arr = np.array(Image.open(cfg.annotation_bmp_path))
    if cfg.class_mapping_yaml_path is not None:
        with open(cfg.class_mapping_yaml_path, "r") as f:
            class_mapping = yaml.safe_load(f)
        classes, inv = np.unique(img_arr, return_inverse=True)
        img_arr = np.array([class_mapping[int(x)] for x in classes])[inv].reshape(
            img_arr.shape
        )
    else:
        classes = np.unique(img_arr)

    gdfs = []
    for clss in classes:
        class_img_arr = (img_arr == clss).astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            class_img_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        class_polygon = contours_to_polygon(contours)
        if isinstance(class_polygon, MultiPolygon):
            class_gdf = gpd.GeoDataFrame(geometry=list(class_polygon.geoms))
        else:
            class_gdf = gpd.GeoDataFrame(geometry=[class_polygon])
        class_gdf = class_gdf.assign(annotation=clss)
        class_gdf = class_gdf.assign(annotation_area=class_gdf.area)
        gdfs.append(class_gdf)
    gdf = pd.concat(gdfs)

    gdf = gdf[gdf.annotation != 0]
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
