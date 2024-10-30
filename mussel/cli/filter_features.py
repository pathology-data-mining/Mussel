import os
import pickle
from dataclasses import dataclass, field

import h5py
import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import MISSING, OmegaConf

from mussel.utils.file import save_hdf5


@dataclass
class FilterFeaturesConfig:
    features_h5_path: str = MISSING
    output_h5_path: str = MISSING
    classifier_pkl: str = MISSING
    classifier_threshold: float = 0.75


cs = ConfigStore.instance()
cs.store(name="filter_features_config", node=FilterFeaturesConfig)

@hydra.main(version_base=None, config_path=".", config_name="filter_features_config")
def main(
    cfg: FilterFeaturesConfig,
):
    logger.info(f"loading model pkl {cfg.classifier_pkl}")
    with open(cfg.classifier_pkl, 'rb') as f:
        clf = pickle.load(f)

    with h5py.File(cfg.features_h5_path, 'r') as features_h5:
        logger.info("Predicting probabilities...")
        inclusion_mask = clf.predict_proba(features_h5['features'])[:, 1] > cfg.classifier_threshold
        logger.info(f"{sum(inclusion_mask)} tiles above {cfg.classifier_threshold} threshold")
        features = features_h5['features'][inclusion_mask]
        coords = features_h5['coords'][inclusion_mask]

    logger.info(f"Saving to {cfg.output_h5_path}")
    save_hdf5(cfg.output_h5_path, {"features": features, "coords": coords}, attr_h5_path=cfg.features_h5_path, mode='w')

if __name__ == "__main__":
    main()
