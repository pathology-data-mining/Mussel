from dataclasses import dataclass
from typing import Optional

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
from shapely.geometry import box

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

@hydra.main(version_base=None, config_path=".", config_name="merge_annotation_features_config")
def main(cfg: MergeAnnotationFeaturesConfig):
    with open(cfg.features_h5_path, 'rb') as f:
        logger.info(f"Reading features from {cfg.features_h5_path}...")
        tiles_h5 = h5py.File(f, 'r')
        patch_size = tiles_h5['coords'].attrs.get("patch_size")
        df = pd.DataFrame(tiles_h5['coords'], columns=['i', 'j'])

        tiles = [box(row.i, row.j, row.i + patch_size, row.j + patch_size) for row in df.itertuples(index=False)]
        tiles_gdf = gpd.GeoDataFrame(
            pd.DataFrame(tiles_h5['features']).add_prefix('feature_', axis=1),
            geometry=tiles)
        tiles_gdf["patch_size"] = patch_size
        logger.info(f"Loaded {len(tiles_gdf)} tiles")

    logger.info(f"Reading annotations from {cfg.annotation_bmp_path}...")
    img_arr = np.array(Image.open(cfg.annotation_bmp_path))
    ind = np.nonzero(img_arr)
    df = pd.DataFrame(np.transpose(ind), columns = ['i', 'j'])
    df = df.assign(annotation=img_arr[ind])
    if cfg.class_mapping_yaml_path is not None:
        with open(cfg.class_mapping_yaml_path, 'r') as f:
            class_mapping = yaml.safe_load(f)
        df.annotation = df.annotation.replace(class_mapping)
    #df.annotation = pd.Categorical(df.annotation).rename_categories(CATEGORIES)
    ann_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['i'], df['j']))
    logger.info(f"Loaded {len(ann_gdf)} annotations pixels")

    logger.info("Joining tiles and annotations...")
    mgdf = tiles_gdf.sjoin(ann_gdf, how='inner')
    gb = mgdf.reset_index().groupby('index')
    ann_count = gb.size()
    ann_mean = gb.annotation.mean()
    logger.info(f"Found {len(ann_mean)} tiles with annotations")

    ann_tiles_gdf = tiles_gdf.iloc[ann_mean.index,:]
    ann_tiles_gdf = ann_tiles_gdf.assign(annotation_count=ann_count, annotation_mean=ann_mean)

    logger.info(f"Writing merged results to {cfg.output_parquet_path}")
    if cfg.slide_id is not None:
        ann_tiles_gdf = ann_tiles_gdf.assign(slide_id=cfg.slide_id)

    ann_tiles_gdf.to_parquet(cfg.output_parquet_path)

if __name__ == "__main__":
    main()

