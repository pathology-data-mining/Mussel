import os
import ssl
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import MISSING, OmegaConf

from mussel.cli.tessellate import (
    SegConfig,
    BiopsySegConfig,
    ResectionSegConfig,
    TcgaSegConfig,
    VisConfig,
    PngConfig,
)
from mussel.cli.tessellate_extract_features_common import (
    process_slide_tessellation_and_filtering,
    create_visualizations,
)
from mussel.models import ModelType, get_required_patch_encoder

ssl._create_default_https_context = ssl._create_unverified_context


defaults = ["_self_", {"seg_config": "default"}]


@dataclass
class TessellateExtractFeaturesConfig:
    """
    Configuration for tessellate-extract-features workflow.
    
    Core Parameters:
        slide_path (str): Path to the whole-slide image.
        slide_id (Optional[str]): Optional slide ID. If None, uses slide filename without extension.
        output_h5_path (str): Path to save final HDF5 file with coordinates and features (post-filtering).
        output_pt_path (str): Path to save final features in PyTorch format (post-filtering).
    
    Filtering Parameters (Optional):
        classifier_pkl (Optional[str]): Path to the classifier model in pickle format for filtering. If None, filtering is skipped.
        classifier_threshold (float): Threshold for the classifier to filter features.
    
    Model Parameters (Pre-Filter Extraction):
        prefilter_model_type (ModelType): Type of model for pre-filtering feature extraction.
        prefilter_model_path (Optional[str]): Path to pre-filtering model weights, if applicable.
    
    Model Parameters (Post-Filter Extraction):
        postfilter_model_type (Optional[ModelType]): Type of model for post-filtering extraction. If None and aggregation_method="model" with slide_model_type specified, automatically infers the required patch encoder from slide_model_type. Otherwise, uses prefilter_model_type.
        postfilter_model_path (Optional[str]): Path to post-filtering model weights, if applicable.
        intermediate_h5_path (Optional[str]): Path for intermediate patch features (two-step mode for post-filtering).
        aggregation_method (str): Aggregation method for post-filtering: identity (single-step), mean/max/model (two-step).
        slide_model_type (Optional[ModelType]): Type of slide encoder model for post-filtering (when aggregation_method="model").
        slide_model_path (Optional[str]): Path to slide encoder model weights for post-filtering.
    
    Visualization Parameters:
        output_png_dir (Optional[str]): Directory to save patches as PNG files (post-filtering).
        output_mask_path (Optional[str]): Path to save the mask image.
        output_grid_mask_path (Optional[str]): Path to save grid mask image (post-filtering).
        output_thumbnail_path (Optional[str]): Path to save thumbnail image.
        thumbnail_size (tuple): Size of the thumbnail image.
    
    Segmentation & Processing Parameters:
        seg_config (SegConfig): Configuration for segmentation parameters.
        vis_config (VisConfig): Configuration for visualization parameters.
        png_config (PngConfig): Configuration for PNG saving parameters.
        num_workers (int): Number of workers for saving patches and feature extraction.
        batch_size (int): Batch size for feature extraction.
        use_gpu (bool): Whether to use GPU for feature extraction.
        gpu_device_id (Optional[int]): Specific GPU device ID to use, if applicable.
        gpu_device_ids (Optional[List[int]]): List of GPU device IDs to use, if applicable.
        keep_intermediate_files (bool): Whether to keep intermediate files (tessellation and pre-filter features).
        save_features_to_h5 (bool): Whether to save the post-filtering features to HDF5.
    """

    defaults: List[Any] = field(default_factory=lambda: defaults)
    slide_path: str = MISSING
    slide_id: Optional[str] = None
    output_h5_path: str = MISSING
    output_pt_path: str = MISSING
    classifier_pkl: Optional[str] = None
    classifier_threshold: float = 0.75
    prefilter_model_type: ModelType = ModelType.CTRANSPATH
    prefilter_model_path: Optional[str] = None
    postfilter_model_type: Optional[ModelType] = None
    postfilter_model_path: Optional[str] = None
    output_png_dir: Optional[str] = None
    output_mask_path: Optional[str] = None
    output_grid_mask_path: Optional[str] = None
    output_thumbnail_path: Optional[str] = None
    thumbnail_size: tuple = (1024, 1024)
    num_workers: int = 4
    batch_size: int = 64
    use_gpu: bool = True
    gpu_device_id: Optional[int] = None
    gpu_device_ids: Optional[List[int]] = None
    keep_intermediate_files: bool = False
    save_features_to_h5: bool = False
    seg_config: SegConfig = MISSING
    vis_config: VisConfig = field(default_factory=VisConfig)
    png_config: PngConfig = field(default_factory=PngConfig)
    intermediate_h5_path: Optional[str] = None
    aggregation_method: str = "identity"
    slide_model_type: Optional[ModelType] = None
    slide_model_path: Optional[str] = None


desc_doc = """== ${hydra.help.app_name} ==

tessellate-extract-features performs an integrated workflow that tessellates a whole-slide image 
and extracts features from the tiles using a foundation model. Optionally, it can filter tiles 
using a classifier and extract features again from the filtered tiles (dual extraction).

Workflow modes:
1. Without filtering (classifier_pkl=None): tessellate → extract features (2 steps)
2. With filtering, same model: tessellate → extract → filter (3 steps, optimized)
3. With filtering, different models: tessellate → extract → filter → re-extract (4 steps)
"""

parameter_doc = f"""
== Available Parameters ==
{TessellateExtractFeaturesConfig.__doc__}
seg_config: {SegConfig.__doc__}
vis_config: {VisConfig.__doc__}
png_config: {PngConfig.__doc__}

"""

cs = ConfigStore.instance()
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=desc_doc, footer=parameter_doc)),
    provider="hydra",
)
cs.store(group="seg_config", name="default", node=SegConfig)
cs.store(group="seg_config", name="biopsy", node=BiopsySegConfig)
cs.store(group="seg_config", name="resection", node=ResectionSegConfig)
cs.store(group="seg_config", name="tcga", node=TcgaSegConfig)
cs.store(name="tessellate_extract_features_config", node=TessellateExtractFeaturesConfig)


@hydra.main(version_base=None, config_path=".", config_name="tessellate_extract_features_config")
def main(
    cfg: TessellateExtractFeaturesConfig,
):
    """Tessellate and extract features, optionally with filtering in between."""
    # Create temporary directory for intermediate files if not keeping them
    temp_dir = None
    base_path = Path(cfg.output_h5_path).parent
    
    if not cfg.keep_intermediate_files:
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Using temporary directory for intermediate files: {temp_dir}")
    
    # Determine if filtering is enabled
    use_filtering = cfg.classifier_pkl is not None
    
    # Determine models for each extraction step
    postfilter_model_type = cfg.postfilter_model_type
    if postfilter_model_type is None:
        if cfg.aggregation_method == "model" and cfg.slide_model_type is not None:
            postfilter_model_type = get_required_patch_encoder(cfg.slide_model_type)
            logger.info(
                f"Auto-inferring postfilter_model_type={postfilter_model_type.name} "
                f"from slide_model_type={cfg.slide_model_type.name}"
            )
        else:
            postfilter_model_type = cfg.prefilter_model_type
    
    postfilter_model_path = cfg.postfilter_model_path if cfg.postfilter_model_path is not None else cfg.prefilter_model_path
    
    # Optimization: If filtering is enabled and models are the same, skip second extraction
    models_are_same = (postfilter_model_type == cfg.prefilter_model_type and 
                       postfilter_model_path == cfg.prefilter_model_path)
    skip_second_extraction = use_filtering and models_are_same
    
    # Process the slide using shared logic
    result = process_slide_tessellation_and_filtering(
        slide_path=cfg.slide_path,
        slide_id=cfg.slide_id,
        output_h5_path=cfg.output_h5_path,
        output_pt_path=cfg.output_pt_path,
        cfg=cfg,
        temp_dir=temp_dir,
        base_path=base_path,
        use_filtering=use_filtering,
        prefilter_model_type=cfg.prefilter_model_type,
        prefilter_model_path=cfg.prefilter_model_path,
        postfilter_model_type=postfilter_model_type,
        postfilter_model_path=postfilter_model_path,
        skip_second_extraction=skip_second_extraction,
        output_mask_path=cfg.output_mask_path,
        two_step_mode=False,  # Single-slide mode doesn't use two-step
    )
    
    if result is None:
        # Processing failed or was completed early
        if temp_dir:
            import shutil
            shutil.rmtree(temp_dir)
        return
    
    # Create visualizations
    create_visualizations(
        slide_path=cfg.slide_path,
        final_coords=result['final_coords'],
        tessellate_h5_path=result['tessellate_h5_path'],
        cfg=cfg,
        output_grid_mask_path=cfg.output_grid_mask_path,
        output_png_dir=cfg.output_png_dir,
        output_thumbnail_path=cfg.output_thumbnail_path,
    )
    
    # Clean up temporary directory if not keeping intermediate files
    if temp_dir:
        import shutil
        shutil.rmtree(temp_dir)
        logger.info("Cleaned up temporary files.")


if __name__ == "__main__":
    main()
