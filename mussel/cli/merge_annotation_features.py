import logging
from dataclasses import dataclass
from typing import Optional

import h5py
import hydra
import numpy as np
import pandas as pd
import yaml
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from PIL import Image

logger = logging.getLogger(__name__)

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


def _raster_merge(coords, features_arr, img_arr, tile_size, class_mapping):
    """
    Pixel-space tile/annotation merge.

    For each tile, counts how many pixels of each annotation class fall inside
    it using np.bincount on the uint8 BMP slice.  Avoids shapely polygon
    construction and geopandas overlay entirely.

    Returns a plain DataFrame with columns:
        feature_0 ... feature_N, annotation, overlap_area, tile_area, tile_size
    """
    H, W = img_arr.shape
    n_tiles = len(coords)
    n_features = features_arr.shape[1]
    feature_col_names = [f"feature_{k}" for k in range(n_features)]

    i_arr = coords[:, 0].astype(np.int64)
    j_arr = coords[:, 1].astype(np.int64)
    i_end = np.minimum(i_arr + tile_size, W)
    j_end = np.minimum(j_arr + tile_size, H)
    actual_tile_areas = (j_end - j_arr) * (i_end - i_arr)

    # Build a flat uint8->int64 lookup for the class mapping (-1 = skip).
    lookup = np.full(256, -1, dtype=np.int64)
    if class_mapping is not None:
        for raw_cls, mapped_val in class_mapping.items():
            raw_cls = int(raw_cls)
            if 0 <= raw_cls <= 255:
                lookup[raw_cls] = int(mapped_val)
    else:
        for v in range(1, 256):
            lookup[v] = v

    # Single pass over tiles: np.bincount(uint8) is O(n+256) -- no sort needed.
    per_class: dict = {}  # mapped_val -> [(tile_idx, overlap_count, tile_area)]
    for k in range(n_tiles):
        tile = img_arr[j_arr[k] : j_end[k], i_arr[k] : i_end[k]]
        bin_cnt = np.bincount(tile.ravel(), minlength=256)
        tile_area = int(actual_tile_areas[k])
        for raw_cls in np.nonzero(bin_cnt)[0]:
            if raw_cls == 0:
                continue
            mapped_val = int(lookup[raw_cls])
            if mapped_val == -1:
                logger.debug("Skipping pixel value %d not in class_mapping", raw_cls)
                continue
            per_class.setdefault(mapped_val, []).append(
                (k, int(bin_cnt[raw_cls]), tile_area)
            )

    if not per_class:
        logger.info("No annotated tiles found")
        return pd.DataFrame()

    dfs = []
    for mapped_val, records in per_class.items():
        tile_idxs = [r[0] for r in records]
        df = pd.DataFrame(features_arr[tile_idxs], columns=feature_col_names)
        df["annotation"] = mapped_val
        df["overlap_area"] = [r[1] for r in records]
        df["tile_area"] = [r[2] for r in records]
        df["tile_size"] = tile_size
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


@hydra.main(
    version_base=None, config_path=".", config_name="merge_annotation_features_config"
)
def main(cfg: MergeAnnotationFeaturesConfig):
    """Merge tile features with annotations from a BMP file."""
    with open(cfg.features_h5_path, "rb") as f:
        logger.info("Reading features from %s...", cfg.features_h5_path)
        tiles_h5 = h5py.File(f, "r")
        tile_size = int(tiles_h5["coords"].attrs.get("patch_size"))
        coords = np.array(tiles_h5["coords"])        # (N, 2): [x, y]
        features_arr = np.array(tiles_h5["features"])  # (N, D)
        logger.info("Loaded %d tiles  feature_dim=%d", len(coords), features_arr.shape[1])

    logger.info("Reading annotations from %s...", cfg.annotation_bmp_path)
    img_arr = np.array(Image.open(cfg.annotation_bmp_path))

    class_mapping = None
    if cfg.class_mapping_yaml_path is not None:
        with open(cfg.class_mapping_yaml_path, "r") as f:
            class_mapping = yaml.safe_load(f)

    result = _raster_merge(coords, features_arr, img_arr, tile_size, class_mapping)

    if result.empty:
        return

    if cfg.slide_id is not None:
        result["slide_id"] = cfg.slide_id

    logger.info("Writing %d rows to %s", len(result), cfg.output_parquet_path)
    result.to_parquet(cfg.output_parquet_path)


if __name__ == "__main__":
    main()
