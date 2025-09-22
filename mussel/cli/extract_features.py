import os
import ssl
from dataclasses import dataclass
from typing import List, Optional

import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import MISSING

from mussel.models import ModelType
from mussel.utils import save_features

ssl._create_default_https_context = ssl._create_unverified_context


@dataclass
class ExtractFeaturesConfig:
    """
    patch_h5_path (str): Path to the HDF5 file containing patches.
    slide_path (str): Path to the whole slide image.
    output_h5_path (str): Path to save the computed features in HDF5 format.
    output_pt_path (Optional[str]): Path to save the computed features in PyTorch format.
    model_type (ModelType): Type of model to use for feature extraction.
    model_path (Optional[str]): Path to the model weights, if applicable.
    patch_path (Optional[str]): Directory containing pre-tiled images, if applicable.
    batch_size (int): Batch size for processing patches or tiles.
    use_gpu (bool): Whether to use GPU for computation.
    gpu_device_id (Optional[int]): Specific GPU device ID to use, if applicable.
    gpu_device_ids (Optional[List[int]]): List of GPU device IDs to use, if applicable.
    num_workers (int): Number of worker threads for data loading.
    """

    patch_h5_path: str = MISSING
    slide_path: str = MISSING
    output_h5_path: str = MISSING
    output_pt_path: Optional[str] = None
    model_type: ModelType = ModelType.CLIP
    model_path: Optional[str] = None
    model_save_path: Optional[str] = None
    patch_path: Optional[str] = None
    batch_size: int = 64
    use_gpu: bool = True
    gpu_device_id: Optional[int] = None
    gpu_device_ids: Optional[List[int]] = None
    num_workers: int = 16


desc_doc = """== ${hydra.help.app_name} ==

Extract features (embeddings) from whole slide images (WSI) or patches using a 
pathology foundation model.  The embeddings are written to a PyTorch tensor file (.pt)
and an HDF5 (.h5) file.
"""

parameter_doc = f"""== Available Parameters ==
{ExtractFeaturesConfig.__doc__}
"""

cs = ConfigStore.instance()
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=desc_doc, footer=parameter_doc)),
    provider="hydra",
)
cs.store(name="extract_features_config", node=ExtractFeaturesConfig)


@hydra.main(version_base=None, config_path=".", config_name="extract_features_config")
def main(cfg: ExtractFeaturesConfig):
    save_features(
        slide_path=cfg.slide_path,
        gpu_device_id=cfg.gpu_device_id,
        model_type=cfg.model_type,
        model_path=cfg.model_path,
        use_gpu=cfg.use_gpu,
        output_h5_path=cfg.output_h5_path,
        output_pt_path=cfg.output_pt_path,
        patch_h5_path=cfg.patch_h5_path,
        patch_path=cfg.patch_path,
        model_save_path=cfg.model_save_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        gpu_device_ids=cfg.gpu_device_ids,
    )


if __name__ == "__main__":
    main()
