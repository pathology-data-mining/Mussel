import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import hydra
import torch
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

logger = logging.getLogger(__name__)

from mussel.utils import (filter_features, load_classifier,
                          load_features_from_h5, save_hdf5)


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
    classifier = load_classifier(cfg.classifier_pkl)

    features, coords_all = load_features_from_h5(
        cfg.features_h5_path, cfg.features_pt_path
    )
    logger.info(f"Loaded {features.shape[0]} features of dimension {features.shape[1]}")
    features, coords = filter_features(
        features,
        coords_all,
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
