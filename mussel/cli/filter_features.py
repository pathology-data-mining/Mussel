import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import h5py
import hydra
import numpy as np
import torch
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import MISSING, OmegaConf

from mussel.utils import save_hdf5, filter_features


@dataclass
class FilterFeaturesConfig:
    """
    features_h5_path (str): Path to the HDF5 file containing features.
    features_pt_path (Optional[str]): Path to the PyTorch file containing features.
    output_h5_path (str): Path to save the filtered features in HDF5 format.
    output_pt_path (str): Path to save the filtered features in PyTorch format.
    classifier_pkl (str): Path to the classifier model in pickle format.
    classifier_threshold (float): Threshold for the classifier to filter features.
    save_features_to_h5 (bool): Whether to save the filtered features to HDF5.
    """

    features_h5_path: str = MISSING
    features_pt_path: Optional[str] = None
    output_h5_path: str = MISSING
    output_pt_path: str = MISSING
    classifier_pkl: str = MISSING
    classifier_threshold: float = 0.75
    save_features_to_h5: bool = False


desc_doc = """== ${hydra.help.app_name} ==
Filter tiles using a classifier model. Features are loaded from an HDF5 or PyTorch
file and assigned a score by the classifier.  Tiles that meet the threshold value are
written to the output file(s).  """

parameter_doc = f"""
== Available Parameters ==
{FilterFeaturesConfig.__doc__}
"""

cs = ConfigStore.instance()
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=desc_doc, footer=parameter_doc)),
    provider="hydra",
)
cs.store(name="filter_features_config", node=FilterFeaturesConfig)


@hydra.main(version_base=None, config_path=".", config_name="filter_features_config")
def main(
    cfg: FilterFeaturesConfig,
):
    """Filter features using a classifier model."""
    logger.info(f"loading model pkl {cfg.classifier_pkl}")
    with open(cfg.classifier_pkl, "rb") as f:
        classifier = pickle.load(f)

    with h5py.File(cfg.features_h5_path, "r") as features_h5:
        if cfg.features_pt_path:
            features = torch.load(cfg.features_pt_path, weights_only=True)
        else:
            features = np.array(features_h5["features"])
            features = torch.Tensor(features)
        logger.info(
            f"Loaded {features.shape[0]} features of dimension {features.shape[1]}"
        )
        features, coords = filter_features(
            features,
            features_h5["coords"][:],
            classifier,
            cfg.classifier_threshold,
        )

        logger.info(f"Saving to {cfg.output_h5_path}")
        asset_dict = {"coords": coords}
        if cfg.save_features_to_h5:
            asset_dict["features"] = features.numpy()
        save_hdf5(
            cfg.output_h5_path,
            asset_dict,
            attr_h5_path=cfg.features_h5_path,
            mode="w",
        )

        torch.save(features, cfg.output_pt_path)


if __name__ == "__main__":
    main()
