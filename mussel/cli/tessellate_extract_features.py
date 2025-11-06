import os
import ssl
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

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
from mussel.models import ModelType, get_required_patch_encoder, get_default_patch_size
from mussel.utils import aggregate_slide_features_batch

ssl._create_default_https_context = ssl._create_unverified_context


defaults = ["_self_", {"seg_config": "default"}]


@dataclass
class TessellateExtractFeaturesConfig:
    """
    Configuration for tessellate-extract-features workflow.
    
    Supports both single-slide and batch processing modes.
    - Single mode: Provide slide_path, output_h5_path, output_pt_path
    - Batch mode: Provide slide_paths, output_dir
    
    Core Parameters (Single Mode):
        slide_path (str): Path to the whole-slide image.
        slide_id (Optional[str]): Optional slide ID. If None, uses slide filename without extension.
        output_h5_path (str): Path to save final HDF5 file with coordinates and features (post-filtering).
        output_pt_path (str): Path to save final features in PyTorch format (post-filtering).
    
    Core Parameters (Batch Mode):
        slide_paths (List[str]): Paths to the whole-slide images to process.
        slide_ids (Optional[List[str]]): Optional slide IDs. If None, uses slide filenames without extension.
        output_dir (str): Directory to save output files. Each slide will have separate output files.
        output_h5_suffix (str): Suffix for output HDF5 files (default: "features.h5").
        output_pt_suffix (str): Suffix for output PyTorch files (default: "features.pt").
        slide_batch_size (int): Number of slides to process in a single batch during slide-level aggregation (default: 8).
    
    Filtering Parameters (Optional):
        classifier_pkl (Optional[str]): Path to the classifier model in pickle format for filtering. If None, filtering is skipped.
        classifier_threshold (float): Threshold for the classifier to filter features.
    
    Model Parameters (Pre-Filter Extraction):
        prefilter_model_type (ModelType): Type of model for pre-filtering feature extraction.
        prefilter_model_path (Optional[str]): Path to pre-filtering model weights, if applicable.
    
    Model Parameters (Post-Filter Extraction):
        postfilter_model_type (Optional[ModelType]): Type of model for post-filtering extraction.
        postfilter_model_path (Optional[str]): Path to post-filtering model weights, if applicable.
        intermediate_h5_path (Optional[str]): Path for intermediate patch features (single mode, two-step).
        aggregation_method (str): Aggregation method for post-filtering: identity (single-step), mean/max/model (two-step).
        slide_model_type (Optional[ModelType]): Type of slide encoder model for post-filtering (when aggregation_method="model").
        slide_model_path (Optional[str]): Path to slide encoder model weights for post-filtering.
    
    Visualization Parameters (Single Mode):
        output_png_dir (Optional[str]): Directory to save patches as PNG files (post-filtering).
        output_mask_path (Optional[str]): Path to save the mask image.
        output_grid_mask_path (Optional[str]): Path to save grid mask image (post-filtering).
        output_thumbnail_path (Optional[str]): Path to save thumbnail image.
    
    Visualization Parameters (Batch Mode):
        output_png_dir_suffix (Optional[str]): Suffix for PNG output directories.
        output_mask_suffix (Optional[str]): Suffix for mask image files.
        output_grid_mask_suffix (Optional[str]): Suffix for grid mask files.
        output_thumbnail_suffix (Optional[str]): Suffix for thumbnail files.
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
    # Single mode parameters
    slide_path: Optional[str] = None
    slide_id: Optional[str] = None
    output_h5_path: Optional[str] = None
    output_pt_path: Optional[str] = None
    # Batch mode parameters
    slide_paths: Optional[List[str]] = None
    slide_ids: Optional[List[str]] = None
    output_dir: Optional[str] = None
    output_h5_suffix: str = "features.h5"
    output_pt_suffix: str = "features.pt"
    slide_batch_size: int = 8
    # Common parameters
    classifier_pkl: Optional[str] = None
    classifier_threshold: float = 0.75
    prefilter_model_type: ModelType = ModelType.CTRANSPATH
    prefilter_model_path: Optional[str] = None
    postfilter_model_type: Optional[ModelType] = None
    postfilter_model_path: Optional[str] = None
    # Single mode visualization
    output_png_dir: Optional[str] = None
    output_mask_path: Optional[str] = None
    output_grid_mask_path: Optional[str] = None
    output_thumbnail_path: Optional[str] = None
    # Batch mode visualization
    output_png_dir_suffix: Optional[str] = None
    output_mask_suffix: Optional[str] = None
    output_grid_mask_suffix: Optional[str] = None
    output_thumbnail_suffix: Optional[str] = None
    # Common visualization
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

    def __post_init__(self):
        """Set default patch size based on model type if not explicitly set."""
        # Only set patch size if seg_config.patch_size is at the default value (256)
        # This allows users to override if they explicitly set a different value
        if self.seg_config.patch_size == 256:  # Default value from SegConfig
            # Get the model type to use for determining patch size
            model_type = self.prefilter_model_type
            
            # Get recommended patch size for the model
            try:
                recommended_patch_size = get_default_patch_size(model_type)
                if recommended_patch_size != 256:
                    logger.info(
                        f"Setting seg_config.patch_size={recommended_patch_size} based on "
                        f"model_type={model_type.name} (recommended default for this model)"
                    )
                    self.seg_config.patch_size = recommended_patch_size
            except ValueError:
                # Model not in mapping, keep default
                pass


desc_doc = """== ${hydra.help.app_name} ==

tessellate-extract-features performs an integrated workflow that tessellates whole-slide images
and extracts features from the tiles using a foundation model. Optionally, it can filter tiles 
using a classifier and extract features again from the filtered tiles (dual extraction).

Supports both single-slide and batch processing modes:
- Single mode: Process one slide (provide slide_path, output_h5_path, output_pt_path)
- Batch mode: Process multiple slides (provide slide_paths, output_dir)

When processing multiple slides with slide-level model aggregation, batch mode provides 
significant performance benefits (6-8x speedup) by:
1. Loading the model only once for all slides
2. Processing slides in parallel on GPU
3. Reducing model initialization overhead

Workflow modes:
1. Without filtering (classifier_pkl=None): tessellate → extract features (2 steps)
2. With filtering, same model: tessellate → extract → filter (3 steps, optimized)
3. With filtering, different models: tessellate → extract → filter → re-extract (4 steps)
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
    """Tessellate and extract features from one or more slides, optionally with filtering."""
    # Detect mode based on configuration
    batch_mode = cfg.slide_paths is not None
    
    if batch_mode:
        logger.info("Running in batch mode (multiple slides)")
        _main_batch(cfg)
    else:
        logger.info("Running in single-slide mode")
        _main_single(cfg)


def _main_single(cfg: TessellateExtractFeaturesConfig):
    """Process a single slide."""
    if cfg.slide_path is None or cfg.output_h5_path is None or cfg.output_pt_path is None:
        raise ValueError(
            "Single-slide mode requires slide_path, output_h5_path, and output_pt_path to be specified"
        )
    
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


def _main_batch(cfg: TessellateExtractFeaturesConfig):
    """Process multiple slides in batch mode."""
    if cfg.slide_paths is None or cfg.output_dir is None:
        raise ValueError(
            "Batch mode requires slide_paths and output_dir to be specified"
        )
    
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
