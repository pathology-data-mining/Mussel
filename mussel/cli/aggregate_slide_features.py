import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig

from mussel.models import ModelType
from mussel.utils import aggregate_slide_features, resolve_aggregation_method

logger = logging.getLogger(__name__)


@dataclass
class AggregateSlideFeaturesConfig:
    """
    Configuration for aggregating patch features to slide-level features.

    Args:
        patch_features_h5_path (str): Path to HDF5 file containing patch-level feature embeddings.
        output_h5_path (str): Path to save the aggregated slide-level features in HDF5 format.
        aggregation_method (str): Method for aggregation: 'identity' (no aggregation), 'mean', 'max', or 'model'.
        slide_model_type (Optional[ModelType]): Type of slide encoder model (when aggregation_method="model").
        slide_model_path (Optional[str]): Path to slide encoder model weights.
        slide_model_kwargs (Dict[str, Any]): Extra keyword arguments forwarded to the slide model constructor.
            TITAN_SLIDE applies its OOM patch by default via patch_oom=True; pass
            slide_model_kwargs={patch_oom:false} to disable that patch and revision pin.
        use_gpu (bool): Whether to use GPU for computation.
        gpu_device_id (Optional[int]): Specific GPU device ID to use.
        gpu_device_ids (Optional[List[int]]): List of GPU device IDs for multi-GPU inference.
        ssl_verify (bool): Whether to verify SSL certificates when downloading models or accessing remote resources (default: True).
        embedding_precision (str): Numeric precision for saved embeddings.
            Options: "float32" (default), "float16", "bfloat16".
    """

    patch_features_h5_path: str = MISSING
    output_h5_path: str = MISSING
    aggregation_method: str = "identity"
    slide_model_type: Optional[ModelType] = None
    slide_model_path: Optional[str] = None
    slide_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    use_gpu: bool = True
    gpu_device_id: Optional[int] = None
    gpu_device_ids: Optional[List[int]] = None
    ssl_verify: bool = True  # Whether to verify SSL certificates for remote operations
    embedding_precision: str = "float32"

    def __post_init__(self):
        if isinstance(self.slide_model_kwargs, DictConfig):
            self.slide_model_kwargs = dict(self.slide_model_kwargs)


desc_doc = """== ${hydra.help.app_name} ==

Aggregate patch-level feature embeddings to slide-level features using various aggregation methods.

This tool takes an HDF5 file containing patch-level features (as produced by extract_features 
in two-step mode) and aggregates them to slide-level representations. It supports both simple 
pooling methods and learned slide encoder models.

Aggregation methods:
  - identity: No aggregation, keeps all patch features (default)
  - mean: Mean pooling across patches
  - max: Max pooling across patches  
  - model: Use a slide encoder model (e.g., GIGAPATH_SLIDE, TITAN_SLIDE)
"""

parameter_doc = f"""== Available Parameters ==
{AggregateSlideFeaturesConfig.__doc__}
"""

cs = ConfigStore.instance()
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=desc_doc, footer=parameter_doc)),
    provider="hydra",
)
cs.store(name="aggregate_slide_features_config", node=AggregateSlideFeaturesConfig)


@hydra.main(
    version_base=None,
    config_path=None,
    config_name="aggregate_slide_features_config",
)
def main(cfg: AggregateSlideFeaturesConfig):
    """
    CLI command to aggregate patch-level features to slide-level features.
    
    This command takes an HDF5 file containing patch-level feature embeddings
    (as produced by extract_features with two-step mode) and aggregates them
    to slide-level features using various methods including trained slide encoder models.
    
    Examples:
        # Mean pooling aggregation
        aggregate_slide_features \\
            patch_features_h5_path=patch_features.h5 \\
            output_h5_path=slide_features.h5 \\
            aggregation_method=mean
        
        # Using GigaPath slide encoder
        aggregate_slide_features \\
            patch_features_h5_path=patch_features.h5 \\
            output_h5_path=slide_features.h5 \\
            slide_model_type=GIGAPATH_SLIDE
        
        # Using TITAN slide encoder
        aggregate_slide_features \\
            patch_features_h5_path=patch_features.h5 \\
            output_h5_path=slide_features.h5 \\
            slide_model_type=TITAN_SLIDE

        # Disable TITAN's default OOM monkey-patch/revision pin, e.g. to test upstream
        aggregate_slide_features \\
            patch_features_h5_path=patch_features.h5 \\
            output_h5_path=slide_features.h5 \\
            slide_model_type=TITAN_SLIDE \\
            slide_model_kwargs={patch_oom:false}
    """
    logger.info("Starting slide feature aggregation")
    logger.info(f"Input: {cfg.patch_features_h5_path}")
    logger.info(f"Output: {cfg.output_h5_path}")
    logger.info(f"Aggregation method: {cfg.aggregation_method}")

    # Auto-set aggregation_method to "model" if slide_model_type is specified
    aggregation_method = resolve_aggregation_method(
        cfg.aggregation_method, cfg.slide_model_type
    )

    if cfg.slide_model_type is not None:
        logger.info(f"Slide model type: {cfg.slide_model_type}")
        if cfg.slide_model_path:
            logger.info(f"Slide model path: {cfg.slide_model_path}")

    # Perform aggregation
    aggregate_slide_features(
        patch_features_h5_path=cfg.patch_features_h5_path,
        output_h5_path=cfg.output_h5_path,
        aggregation_method=aggregation_method,
        model_type=cfg.slide_model_type,
        model_path=cfg.slide_model_path,
        use_gpu=cfg.use_gpu,
        gpu_device_id=cfg.gpu_device_id,
        gpu_device_ids=cfg.gpu_device_ids,
        embedding_precision=cfg.embedding_precision,
        slide_model_kwargs=cfg.slide_model_kwargs,
    )

    logger.info(f"Slide features saved to: {cfg.output_h5_path}")
    logger.info("Aggregation complete!")


if __name__ == "__main__":
    main()
