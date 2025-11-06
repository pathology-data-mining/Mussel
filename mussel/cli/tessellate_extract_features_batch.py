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
from mussel.utils import aggregate_slide_features_batch

ssl._create_default_https_context = ssl._create_unverified_context


defaults = ["_self_", {"seg_config": "default"}]


@dataclass
class TessellateExtractFeaturesBatchConfig:
    """
    Configuration for batch tessellate-extract-features workflow.
    
    Core Parameters:
        slide_paths (List[str]): Paths to the whole-slide images to process.
        slide_ids (Optional[List[str]]): Optional slide IDs. If None, uses slide filenames without extension.
        output_dir (str): Directory to save output files. Each slide will have separate output files.
        output_h5_suffix (str): Suffix for output HDF5 files (default: "features.h5").
        output_pt_suffix (str): Suffix for output PyTorch files (default: "features.pt").
    
    Filtering Parameters (Optional):
        classifier_pkl (Optional[str]): Path to the classifier model in pickle format for filtering. If None, filtering is skipped.
        classifier_threshold (float): Threshold for the classifier to filter features.
    
    Model Parameters (Pre-Filter Extraction):
        prefilter_model_type (ModelType): Type of model for pre-filtering feature extraction.
        prefilter_model_path (Optional[str]): Path to pre-filtering model weights, if applicable.
    
    Model Parameters (Post-Filter Extraction):
        postfilter_model_type (Optional[ModelType]): Type of model for post-filtering extraction.
        postfilter_model_path (Optional[str]): Path to post-filtering model weights, if applicable.
        intermediate_h5_suffix (Optional[str]): Suffix for intermediate patch features files.
        aggregation_method (str): Aggregation method for post-filtering: identity (single-step), mean/max/model (two-step).
        slide_model_type (Optional[ModelType]): Type of slide encoder model for post-filtering (when aggregation_method="model").
        slide_model_path (Optional[str]): Path to slide encoder model weights for post-filtering.
        slide_batch_size (int): Number of slides to process in a single batch during slide-level aggregation (default: 8).
    
    Visualization Parameters:
        output_png_dir_suffix (Optional[str]): Suffix for PNG output directories (e.g., "_patches").
        output_mask_suffix (Optional[str]): Suffix for mask image files (e.g., "_mask.png").
        output_grid_mask_suffix (Optional[str]): Suffix for grid mask files (e.g., "_grid.png").
        output_thumbnail_suffix (Optional[str]): Suffix for thumbnail files (e.g., "_thumbnail.png").
        thumbnail_size (tuple): Size of the thumbnail image.
    
    Segmentation & Processing Parameters:
        seg_config (SegConfig): Configuration for segmentation parameters.
        vis_config (VisConfig): Configuration for visualization parameters.
        png_config (PngConfig): Configuration for PNG saving parameters.
        num_workers (int): Number of workers for saving patches and feature extraction.
        batch_size (int): Batch size for tile-level feature extraction.
        use_gpu (bool): Whether to use GPU for feature extraction.
        gpu_device_id (Optional[int]): Specific GPU device ID to use, if applicable.
        gpu_device_ids (Optional[List[int]]): List of GPU device IDs to use, if applicable.
        keep_intermediate_files (bool): Whether to keep intermediate files (tessellation and pre-filter features).
        save_features_to_h5 (bool): Whether to save the post-filtering features to HDF5.
    """

    defaults: List[Any] = field(default_factory=lambda: defaults)
    slide_paths: List[str] = MISSING
    slide_ids: Optional[List[str]] = None
    output_dir: str = MISSING
    output_h5_suffix: str = "features.h5"
    output_pt_suffix: str = "features.pt"
    classifier_pkl: Optional[str] = None
    classifier_threshold: float = 0.75
    prefilter_model_type: ModelType = ModelType.CTRANSPATH
    prefilter_model_path: Optional[str] = None
    postfilter_model_type: Optional[ModelType] = None
    postfilter_model_path: Optional[str] = None
    output_png_dir_suffix: Optional[str] = None
    output_mask_suffix: Optional[str] = None
    output_grid_mask_suffix: Optional[str] = None
    output_thumbnail_suffix: Optional[str] = None
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
    intermediate_h5_suffix: Optional[str] = None
    aggregation_method: str = "identity"
    slide_model_type: Optional[ModelType] = None
    slide_model_path: Optional[str] = None
    slide_batch_size: int = 8


desc_doc = """== ${hydra.help.app_name} ==

tessellate-extract-features-batch performs batch processing of multiple whole-slide images,
tessellating and extracting features from tiles using a foundation model. This is optimized
for processing multiple slides efficiently, especially when using slide-level aggregation models.

Key benefits of batch processing:
1. Model loaded only once for all slides
2. Better GPU utilization when processing slide-level features
3. Reduced overhead from model initialization

Workflow modes:
1. Without filtering (classifier_pkl=None): tessellate → extract features (2 steps per slide)
2. With filtering, same model: tessellate → extract → filter (3 steps per slide, optimized)
3. With filtering, different models: tessellate → extract → filter → re-extract (4 steps per slide)

When using aggregation_method="model", slides are batch processed during slide-level aggregation
to maximize GPU efficiency.
"""

parameter_doc = f"""
== Available Parameters ==
{TessellateExtractFeaturesBatchConfig.__doc__}
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
cs.store(name="tessellate_extract_features_batch_config", node=TessellateExtractFeaturesBatchConfig)


@hydra.main(version_base=None, config_path=".", config_name="tessellate_extract_features_batch_config")
def main(
    cfg: TessellateExtractFeaturesBatchConfig,
):
    """Batch tessellate and extract features from multiple slides."""
    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for intermediate files if not keeping them
    temp_dir = None
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
    
    # Generate slide IDs if not provided
    slide_ids = cfg.slide_ids
    if slide_ids is None:
        slide_ids = [Path(sp).stem for sp in cfg.slide_paths]
    
    # Process each slide
    logger.info(f"Processing {len(cfg.slide_paths)} slides in batch mode")
    
    slides_for_batch_aggregation = []
    
    for i, (slide_path, slide_id) in enumerate(zip(cfg.slide_paths, slide_ids)):
        logger.info(f"\n=== Processing slide {i+1}/{len(cfg.slide_paths)}: {slide_id} ===")
        
        # Determine output paths
        base_path = output_dir
        output_h5_path = str(output_dir / f"{slide_id}.{cfg.output_h5_suffix}")
        output_pt_path = str(output_dir / f"{slide_id}.{cfg.output_pt_suffix}")
        
        # Determine output mask path
        output_mask_path = None
        if cfg.output_mask_suffix:
            output_mask_path = str(base_path / f"{slide_id}{cfg.output_mask_suffix}")
        
        # Use two-step mode for batch aggregation
        two_step_mode = cfg.aggregation_method != "identity"
        
        # Process slide using shared logic
        result = process_slide_tessellation_and_filtering(
            slide_path=slide_path,
            slide_id=slide_id,
            output_h5_path=output_h5_path,
            output_pt_path=output_pt_path,
            cfg=cfg,
            temp_dir=temp_dir,
            base_path=base_path,
            use_filtering=use_filtering,
            prefilter_model_type=cfg.prefilter_model_type,
            prefilter_model_path=cfg.prefilter_model_path,
            postfilter_model_type=postfilter_model_type,
            postfilter_model_path=postfilter_model_path,
            skip_second_extraction=skip_second_extraction,
            output_mask_path=output_mask_path,
            two_step_mode=two_step_mode,
        )
        
        if result is None:
            # Processing failed or was completed early
            continue
        
        # Check if we need batch aggregation
        if 'intermediate_h5_path' in result:
            slides_for_batch_aggregation.append(result)
        else:
            # Create visualizations for completed slides
            output_grid_mask_path = None
            if cfg.output_grid_mask_suffix:
                output_grid_mask_path = str(base_path / f"{slide_id}{cfg.output_grid_mask_suffix}")
            
            output_png_dir = None
            if cfg.output_png_dir_suffix:
                output_png_dir = str(base_path / f"{slide_id}{cfg.output_png_dir_suffix}")
            
            output_thumbnail_path = None
            if cfg.output_thumbnail_suffix:
                output_thumbnail_path = str(base_path / f"{slide_id}{cfg.output_thumbnail_suffix}")
            
            create_visualizations(
                slide_path=slide_path,
                final_coords=result['final_coords'],
                tessellate_h5_path=result['tessellate_h5_path'],
                cfg=cfg,
                output_grid_mask_path=output_grid_mask_path,
                output_png_dir=output_png_dir,
                output_thumbnail_path=output_thumbnail_path,
            )
    
    # Perform batch aggregation if needed
    if slides_for_batch_aggregation and cfg.aggregation_method == "model":
        logger.info(f"\n=== Batch aggregating {len(slides_for_batch_aggregation)} slides ===")
        
        patch_h5_paths = [s['intermediate_h5_path'] for s in slides_for_batch_aggregation]
        output_h5_paths = [s['output_h5_path'] for s in slides_for_batch_aggregation]
        output_pt_paths = [s['output_pt_path'] for s in slides_for_batch_aggregation]
        
        aggregate_slide_features_batch(
            patch_features_h5_paths=patch_h5_paths,
            output_h5_paths=output_h5_paths,
            output_pt_paths=output_pt_paths,
            aggregation_method=cfg.aggregation_method,
            model_type=cfg.slide_model_type,
            model_path=cfg.slide_model_path,
            use_gpu=cfg.use_gpu,
            gpu_device_id=cfg.gpu_device_id,
            gpu_device_ids=cfg.gpu_device_ids,
            slide_batch_size=cfg.slide_batch_size,
        )
        
        # Create visualizations for batch-processed slides
        for result in slides_for_batch_aggregation:
            slide_id = Path(result['slide_path']).stem
            base_path = output_dir
            
            output_grid_mask_path = None
            if cfg.output_grid_mask_suffix:
                output_grid_mask_path = str(base_path / f"{slide_id}{cfg.output_grid_mask_suffix}")
            
            output_png_dir = None
            if cfg.output_png_dir_suffix:
                output_png_dir = str(base_path / f"{slide_id}{cfg.output_png_dir_suffix}")
            
            output_thumbnail_path = None
            if cfg.output_thumbnail_suffix:
                output_thumbnail_path = str(base_path / f"{slide_id}{cfg.output_thumbnail_suffix}")
            
            create_visualizations(
                slide_path=result['slide_path'],
                final_coords=result['final_coords'],
                tessellate_h5_path=result['tessellate_h5_path'],
                cfg=cfg,
                output_grid_mask_path=output_grid_mask_path,
                output_png_dir=output_png_dir,
                output_thumbnail_path=output_thumbnail_path,
            )
    
    # Clean up temporary directory if not keeping intermediate files
    if temp_dir:
        import shutil
        shutil.rmtree(temp_dir)
        logger.info("Cleaned up temporary files.")
    
    logger.info(f"\n=== Batch processing complete! ===")
    logger.info(f"Processed {len(cfg.slide_paths)} slides")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
