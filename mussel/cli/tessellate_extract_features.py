import os
import ssl
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

import h5py
import torch
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
    process_slide_tessellation_only,
    create_visualizations,
)

from mussel.models import ModelType, get_required_patch_encoder, get_default_patch_size
from mussel.utils import aggregate_slide_features_batch, extract_patch_features_batch, save_torch_tensor

ssl._create_default_https_context = ssl._create_unverified_context


def _is_remote_path(path):
    """Check if a path is a remote URL scheme."""
    if not isinstance(path, str):
        return False
    return path.startswith(('az://', 'abfs://', 's3://', 'gs://', 'http://', 'https://'))


def _safe_path_join(base_path, *parts):
    """Safely join path components, preserving URL schemes for remote paths.
    
    Args:
        base_path: Base path (can be local or remote URL)
        *parts: Path components to join
        
    Returns:
        Joined path as string
    """
    if _is_remote_path(str(base_path)):
        # For remote paths, use string concatenation with /
        result = str(base_path).rstrip('/')
        for part in parts:
            result = f"{result}/{str(part).lstrip('/')}"
        return result
    else:
        # For local paths, use Path
        return str(Path(base_path) / Path(*parts))


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
            Supports remote paths (az://, s3://, etc.) when fsspec is installed.
        output_pt_path (str): Path to save final features in PyTorch format (post-filtering).
            Supports remote paths (az://, s3://, etc.) when fsspec is installed.
    
    Core Parameters (Batch Mode):
        slide_paths (List[str]): Paths to the whole-slide images to process.
        slide_ids (Optional[List[str]]): Optional slide IDs. If None, uses slide filenames without extension.
        output_dir (str): Directory to save output files. Each slide will have separate output files.
            Supports remote paths (az://, s3://, etc.) when fsspec is installed.
        output_h5_suffix (str): Suffix for output HDF5 files (default: "features.h5").
        output_pt_suffix (str): Suffix for output PyTorch files (default: "features.pt").
        slide_batch_size (int): Number of slides to process in a single batch during slide-level aggregation (default: 8).
    
    Filtering Parameters (Optional):
        classifier_pkl (Optional[str]): Path to the classifier model in pickle format for filtering. If None, filtering is skipped.
        classifier_threshold (float): Threshold for the classifier to filter features.
    
    Model Parameters (Pre-Filter Extraction):
        prefilter_model_type (ModelType): Type of model for pre-filtering feature extraction.
            This is a PATCH-LEVEL encoder. Use a single value (e.g., CTRANSPATH).
        prefilter_model_path (Optional[str]): Path to pre-filtering model weights, if applicable.
    
    Model Parameters (Post-Filter Patch-Level Extraction):
        model_type (Optional[ModelType] or List[ModelType]): Type of model(s) for PATCH-LEVEL feature extraction.
            - Single mode: Accepts a single ModelType (e.g., model_type=OPTIMUS)
            - Batch mode: Accepts a list of ModelTypes for multi-model extraction
                         (e.g., model_type=[OPTIMUS,VIRCHOW,UNI])
            - Command line usage: model_type=[OPTIMUS,VIRCHOW] (no quotes around the list)
            - When using slide encoders, the appropriate patch encoder is automatically selected
        model_path (Optional[str]): Path to model weights, if applicable.
        intermediate_h5_path (Optional[str]): Path for intermediate patch features (single mode, two-step).
        aggregation_method (str): Aggregation method for post-filtering: identity (single-step), mean/max/model (two-step).
    
    Model Parameters (Slide-Level Aggregation):
        slide_model_type (Optional[ModelType] or List[ModelType]): Type of model(s) for SLIDE-LEVEL encoding.
            - Single mode: Accepts a single ModelType (e.g., slide_model_type=GIGAPATH_SLIDE)
            - Batch mode: Accepts a list of ModelTypes for multi-model slide encoding
                         (e.g., slide_model_type=[GIGAPATH_SLIDE,TITAN_SLIDE])
            - Command line usage: slide_model_type=[GIGAPATH_SLIDE,TITAN_SLIDE] (no quotes)
            - Each slide encoder requires a specific patch encoder:
                * GIGAPATH_SLIDE requires GIGAPATH patch encoder
                * TITAN_SLIDE requires CONCH1_5 patch encoder
            - The required patch encoder is automatically paired and run as needed
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
        save_patch_tokens (bool): Whether to save full patch tokens instead of aggregated embeddings.
            - Default: False (saves aggregated single embedding per patch)
            - When True: Saves all patch tokens (e.g., 257 tokens for ViT models)
            - Impact: Setting to True increases file sizes 257x for ViT models
            - Recommendation: Keep False unless you need patch-level attention analysis
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
    model_type: Any = None  # Can be ModelType or List[ModelType]
    model_path: Optional[str] = None
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
    save_patch_tokens: bool = False  # Whether to save full patch tokens (e.g., 257 tokens for ViT) instead of aggregated embeddings
    seg_config: SegConfig = MISSING
    vis_config: VisConfig = field(default_factory=VisConfig)
    png_config: PngConfig = field(default_factory=PngConfig)
    intermediate_h5_path: Optional[str] = None
    aggregation_method: str = "identity"
    slide_model_type: Any = None  # Can be ModelType or List[ModelType]
    slide_model_path: Optional[str] = None

    def __post_init__(self):
        """Set default patch size based on model type if not explicitly set."""
        from omegaconf import ListConfig
        
        # Convert ListConfig to actual lists of ModelType enums
        if isinstance(self.model_type, ListConfig):
            self.model_type = [ModelType[name] for name in self.model_type]
        elif isinstance(self.model_type, str):
            self.model_type = ModelType[self.model_type]
        
        if isinstance(self.slide_model_type, ListConfig):
            self.slide_model_type = [ModelType[name] for name in self.slide_model_type]
        elif isinstance(self.slide_model_type, str):
            self.slide_model_type = ModelType[self.slide_model_type]
        
        # Only set patch size if seg_config.patch_size is at the default value
        # This allows users to override if they explicitly set a different value
        logger.debug(
            f"__post_init__: seg_config.patch_size={self.seg_config.patch_size}, "
            f"DEFAULT_PATCH_SIZE={SegConfig.DEFAULT_PATCH_SIZE}, "
            f"model_type={self.prefilter_model_type}"
        )
        
        if self.seg_config.patch_size == SegConfig.DEFAULT_PATCH_SIZE:
            # Get the model type to use for determining patch size
            model_type = self.prefilter_model_type
            
            # Get recommended patch size for the model
            try:
                recommended_patch_size = get_default_patch_size(model_type)
                if recommended_patch_size != SegConfig.DEFAULT_PATCH_SIZE:
                    logger.info(
                        f"Setting seg_config.patch_size={recommended_patch_size} based on "
                        f"model_type={model_type.name} (recommended default for this model)"
                    )
                    self.seg_config.patch_size = recommended_patch_size
            except ValueError:
                # Model not in mapping, keep default
                pass
        else:
            logger.debug(
                f"__post_init__: patch_size ({self.seg_config.patch_size}) already set, "
                f"not applying model-specific default"
            )


desc_doc = """== ${hydra.help.app_name} ==

tessellate-extract-features performs an integrated workflow that tessellates whole-slide images
and extracts features from the tiles using foundation models. Optionally, it can filter tiles 
using a classifier and extract features again from the filtered tiles (dual extraction).

Supports both single-slide and batch processing modes:
- Single mode: Process one slide (provide slide_path, output_h5_path, output_pt_path)
- Batch mode: Process multiple slides (provide slide_paths, output_dir)

Model Types:
- model_type: For PATCH-LEVEL feature extraction (e.g., OPTIMUS, VIRCHOW, UNI)
  * Single mode: Single model (e.g., model_type=OPTIMUS)
  * Batch mode: Accepts list (e.g., model_type=[OPTIMUS,VIRCHOH,UNI])
  * Command line: Use unquoted list notation: model_type=[MODEL1,MODEL2]
  
- slide_model_type: For SLIDE-LEVEL encoding/aggregation (e.g., GIGAPATH_SLIDE, TITAN_SLIDE)
  * Single mode: Single model (e.g., slide_model_type=GIGAPATH_SLIDE)
  * Batch mode: Accepts list (e.g., slide_model_type=[GIGAPATH_SLIDE,TITAN_SLIDE])
  * Command line: Use unquoted list notation: slide_model_type=[MODEL1,MODEL2]
  * Each slide encoder automatically uses its required patch encoder:
    - GIGAPATH_SLIDE → GIGAPATH (patch encoder)
    - TITAN_SLIDE → CONCH1_5 (patch encoder)

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
    # Set multiprocessing start method to avoid permission issues in containers
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Import ListConfig for type checking
    from omegaconf import ListConfig
    
    # Apply model-specific defaults if not explicitly set
    # Hydra doesn't call __post_init__ on structured configs, so we do it here
    
    # Convert model_type and slide_model_type from strings to ModelType enums
    if isinstance(cfg.model_type, str):
        cfg.model_type = ModelType[cfg.model_type]
    elif isinstance(cfg.model_type, ListConfig):
        cfg.model_type = [ModelType[name] for name in cfg.model_type]
    
    if isinstance(cfg.slide_model_type, str):
        cfg.slide_model_type = ModelType[cfg.slide_model_type]
    elif isinstance(cfg.slide_model_type, ListConfig):
        cfg.slide_model_type = [ModelType[name] for name in cfg.slide_model_type]
    
    if cfg.seg_config.patch_size == SegConfig.DEFAULT_PATCH_SIZE:
        try:
            # Use model_type if set, otherwise fall back to prefilter_model_type
            model_for_patch_size = cfg.model_type if cfg.model_type is not None else cfg.prefilter_model_type
            recommended_patch_size = get_default_patch_size(model_for_patch_size)
            if recommended_patch_size != SegConfig.DEFAULT_PATCH_SIZE:
                logger.info(
                    f"Setting seg_config.patch_size={recommended_patch_size} based on "
                    f"model_type={model_for_patch_size.name} (recommended default for this model)"
                )
                cfg.seg_config.patch_size = recommended_patch_size
        except (ValueError, AttributeError):
            # Model not in mapping or other issue, keep default
            pass
    
    # Detect mode based on configuration
    batch_mode = cfg.slide_paths is not None
    
    # Detect if model_type or slide_model_type are lists (multi-model mode)
    # Check for both list and ListConfig since __post_init__ may not have run yet
    is_model_list = isinstance(cfg.model_type, (list, ListConfig))
    is_slide_model_list = isinstance(cfg.slide_model_type, (list, ListConfig))
    multi_model_mode = is_model_list or is_slide_model_list
    
    logger.debug(f"model_type type: {type(cfg.model_type)}, value: {cfg.model_type}, is_list: {is_model_list}")
    logger.debug(f"slide_model_type type: {type(cfg.slide_model_type)}, value: {cfg.slide_model_type}, is_list: {is_slide_model_list}")
    logger.debug(f"multi_model_mode: {multi_model_mode}")
    
    if batch_mode:
        logger.info("Running in batch mode (multiple slides)")
        if multi_model_mode:
            logger.info("Multi-model optimization enabled: grouping by patch size")
            _main_batch_multi_model(cfg)
        else:
            _main_batch(cfg)
    else:
        logger.info("Running in single-slide mode")
        if multi_model_mode:
            logger.info("Multi-model mode with single slide")
            _main_single_multi_model(cfg)
        else:
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
    model_type = cfg.model_type
    if model_type is None:
        if cfg.aggregation_method == "model" and cfg.slide_model_type is not None:
            model_type = get_required_patch_encoder(cfg.slide_model_type)
            logger.info(
                f"Auto-inferring model_type={model_type.name} "
                f"from slide_model_type={cfg.slide_model_type.name}"
            )
            # Update patch_size if it was at default and the inferred model needs a different size
            if cfg.seg_config.patch_size == SegConfig.DEFAULT_PATCH_SIZE:
                try:
                    recommended_patch_size = get_default_patch_size(model_type)
                    if recommended_patch_size != SegConfig.DEFAULT_PATCH_SIZE:
                        logger.info(
                            f"Updating seg_config.patch_size={recommended_patch_size} based on "
                            f"inferred model_type={model_type.name}"
                        )
                        cfg.seg_config.patch_size = recommended_patch_size
                except ValueError:
                    pass
        else:
            model_type = cfg.prefilter_model_type
    
    model_path = cfg.model_path if cfg.model_path is not None else cfg.prefilter_model_path
    
    # Optimization: If filtering is enabled and models are the same, skip second extraction
    models_are_same = (model_type == cfg.prefilter_model_type and 
                       model_path == cfg.prefilter_model_path)
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
        model_type=model_type,
        model_path=model_path,
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


def _main_batch(cfg: TessellateExtractFeaturesConfig, patch_output_dir: Optional[str] = None):
    """Process multiple slides in batch mode with tile-level batching.
    
    Args:
        cfg: Configuration for tessellation and feature extraction.
        patch_output_dir: Optional separate output directory for patch-level features.
                         If None, patch features go in cfg.output_dir. If specified,
                         patch features go in patch_output_dir while slide features
                         go in cfg.output_dir. Used for slide models to separate
                         patch encoder outputs from slide encoder outputs.
    """
    if cfg.slide_paths is None or cfg.output_dir is None:
        raise ValueError(
            "Batch mode requires slide_paths and output_dir to be specified"
        )
    
    # Create output directory (only for local paths)
    output_dir_str = cfg.output_dir
    if not _is_remote_path(output_dir_str):
        output_dir_path = Path(output_dir_str)
        output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create patch output directory if specified
    patch_output_dir_str = patch_output_dir if patch_output_dir else output_dir_str
    if patch_output_dir and not _is_remote_path(patch_output_dir_str):
        patch_output_dir_path = Path(patch_output_dir_str)
        patch_output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for intermediate files if not keeping them
    temp_dir = None
    if not cfg.keep_intermediate_files:
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Using temporary directory for intermediate files: {temp_dir}")
    
    # Determine if filtering is enabled
    use_filtering = cfg.classifier_pkl is not None
    
    # Determine models for each extraction step
    model_type = cfg.model_type
    if model_type is None:
        if cfg.aggregation_method == "model" and cfg.slide_model_type is not None:
            model_type = get_required_patch_encoder(cfg.slide_model_type)
            logger.info(
                f"Auto-inferring model_type={model_type.name} "
                f"from slide_model_type={cfg.slide_model_type.name}"
            )
            # Update patch_size if it was at default and the inferred model needs a different size
            if cfg.seg_config.patch_size == SegConfig.DEFAULT_PATCH_SIZE:
                try:
                    recommended_patch_size = get_default_patch_size(model_type)
                    if recommended_patch_size != SegConfig.DEFAULT_PATCH_SIZE:
                        logger.info(
                            f"Updating seg_config.patch_size={recommended_patch_size} based on "
                            f"inferred model_type={model_type.name}"
                        )
                        cfg.seg_config.patch_size = recommended_patch_size
                except ValueError:
                    pass
        else:
            model_type = cfg.prefilter_model_type
    
    model_path = cfg.model_path if cfg.model_path is not None else cfg.prefilter_model_path
    
    # Optimization: If filtering is enabled and models are the same, skip second extraction
    models_are_same = (model_type == cfg.prefilter_model_type and 
                       model_path == cfg.prefilter_model_path)
    skip_second_extraction = use_filtering and models_are_same
    
    # Generate slide IDs if not provided
    slide_ids = cfg.slide_ids
    if slide_ids is None:
        slide_ids = [Path(sp).stem for sp in cfg.slide_paths]
    
    # Phase 1: Tessellate all slides and perform filtering if needed
    logger.info(f"\n=== Phase 1: Tessellating {len(cfg.slide_paths)} slides ===")
    
    slide_results = []
    for i, (slide_path, slide_id) in enumerate(zip(cfg.slide_paths, slide_ids)):
        logger.info(f"\nTessellating slide {i+1}/{len(cfg.slide_paths)}: {slide_id}")
        
        # Determine output mask path
        output_mask_path = None
        if cfg.output_mask_suffix:
            output_mask_path = _safe_path_join(output_dir_str, f"{slide_id}.{cfg.output_mask_suffix}")
        
        # Tessellate and filter (but don't extract features yet)
        result = process_slide_tessellation_only(
            slide_path=slide_path,
            slide_id=slide_id,
            cfg=cfg,
            temp_dir=temp_dir,
            base_path=output_dir_str,
            use_filtering=use_filtering,
            prefilter_model_type=cfg.prefilter_model_type,
            prefilter_model_path=cfg.prefilter_model_path,
            skip_second_extraction=skip_second_extraction,
            output_mask_path=output_mask_path,
        )
        
        if result is None:
            logger.warning(f"Skipping slide {slide_id} due to tessellation failure")
            continue
        
        # Add output paths to result
        result['slide_id'] = slide_id
        result['output_h5_path'] = _safe_path_join(output_dir_str, "h5", f"{slide_id}.{cfg.output_h5_suffix}")
        result['output_pt_path'] = _safe_path_join(output_dir_str, "pt", f"{slide_id}.{cfg.output_pt_suffix}")
        slide_results.append(result)
    
    if not slide_results:
        logger.error("No slides to process after tessellation phase")
        if temp_dir:
            import shutil
            shutil.rmtree(temp_dir)
        return
    
    # Create output subdirectories (h5, pt) if they're local paths
    for subdir in ["h5", "pt"]:
        subdir_path = _safe_path_join(output_dir_str, subdir)
        if not _is_remote_path(subdir_path):
            Path(subdir_path).mkdir(parents=True, exist_ok=True)
    
    # Phase 2: Batch extract patch features for all slides
    logger.info(f"\n=== Phase 2: Batch extracting patch features for {len(slide_results)} slides ===")
    
    # Prepare paths for batch extraction
    patch_h5_paths = [r['final_coords_h5_path'] for r in slide_results]
    slide_paths = [r['slide_path'] for r in slide_results]
    
    # Determine if we need two-step processing (patch features + slide aggregation)
    use_two_step = cfg.aggregation_method != "identity"
    
    if use_two_step:
        # Extract to intermediate patch feature files for later aggregation
        # Use patch_output_dir_str for patch features (separate from slide features for slide models)
        intermediate_h5_paths = [
            _safe_path_join(patch_output_dir_str, "tile_h5", f"{r['slide_id']}.patch.h5") 
            for r in slide_results
        ]
        
        # Create tile_h5 subdirectory if it's a local path
        tile_h5_dir = _safe_path_join(patch_output_dir_str, "tile_h5")
        if not _is_remote_path(tile_h5_dir):
            Path(tile_h5_dir).mkdir(parents=True, exist_ok=True)
        
        extract_patch_features_batch(
            patch_h5_paths=patch_h5_paths,
            slide_paths=slide_paths,
            output_h5_paths=intermediate_h5_paths,
            model_type=model_type,
            model_path=model_path,
            batch_size=cfg.batch_size,
            use_gpu=cfg.use_gpu,
            gpu_device_id=cfg.gpu_device_id,
            gpu_device_ids=cfg.gpu_device_ids,
            num_workers=cfg.num_workers,
            pin_memory=True,
            is_test_run=False,
            save_patch_tokens=cfg.save_patch_tokens,
        )
        
        # Add intermediate paths to results
        for r, intermediate_h5_path in zip(slide_results, intermediate_h5_paths):
            r['intermediate_h5_path'] = intermediate_h5_path
        
        # If patch_output_dir is specified, also save aggregated patch encoder features
        if patch_output_dir:
            logger.info(f"\n=== Phase 2b: Saving aggregated patch encoder features to {patch_output_dir} ===")
            
            # Create h5 and pt subdirectories if they're local paths
            for subdir in ["h5", "pt"]:
                subdir_path = _safe_path_join(patch_output_dir_str, subdir)
                if not _is_remote_path(subdir_path):
                    Path(subdir_path).mkdir(parents=True, exist_ok=True)
            
            patch_encoder_h5_paths = [
                _safe_path_join(patch_output_dir_str, "h5", f"{r['slide_id']}.{cfg.output_h5_suffix}")
                for r in slide_results
            ]
            patch_encoder_pt_paths = [
                _safe_path_join(patch_output_dir_str, "pt", f"{r['slide_id']}.{cfg.output_pt_suffix}")
                for r in slide_results
            ]
            
            # Aggregate patch features using simple aggregation (mean)
            aggregate_slide_features_batch(
                patch_features_h5_paths=intermediate_h5_paths,
                output_h5_paths=patch_encoder_h5_paths,
                output_pt_paths=patch_encoder_pt_paths,
                aggregation_method="mean",  # Simple aggregation for patch encoder
                model_type=None,
                model_path=None,
                use_gpu=False,
                gpu_device_id=None,
                gpu_device_ids=None,
                slide_batch_size=cfg.slide_batch_size,
            )
        
        # Phase 3: Batch aggregate to slide level
        logger.info(f"\n=== Phase 3: Batch aggregating {len(slide_results)} slides (aggregation_method={cfg.aggregation_method}) ===")
        
        output_h5_paths = [r['output_h5_path'] for r in slide_results]
        output_pt_paths = [r['output_pt_path'] for r in slide_results]
        
        aggregate_slide_features_batch(
            patch_features_h5_paths=intermediate_h5_paths,
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
    else:
        # Single-step: extract directly to final output (no aggregation)
        output_h5_paths = [r['output_h5_path'] for r in slide_results]
        
        extract_patch_features_batch(
            patch_h5_paths=patch_h5_paths,
            slide_paths=slide_paths,
            output_h5_paths=output_h5_paths,
            model_type=model_type,
            model_path=model_path,
            batch_size=cfg.batch_size,
            use_gpu=cfg.use_gpu,
            gpu_device_id=cfg.gpu_device_id,
            gpu_device_ids=cfg.gpu_device_ids,
            num_workers=cfg.num_workers,
            pin_memory=True,
            is_test_run=False,
            save_patch_tokens=cfg.save_patch_tokens,
        )
        
        # Also save as PT format for consistency
        for r in slide_results:
            with h5py.File(r['output_h5_path'], "r") as f:
                features = torch.from_numpy(f["features"][:])
                save_torch_tensor(r['output_pt_path'], features)
    
    # Phase 4: Create visualizations
    logger.info(f"\n=== Phase 4: Creating visualizations ===")
    for r in slide_results:
        slide_id = r['slide_id']
        
        output_grid_mask_path = None
        if cfg.output_grid_mask_suffix:
            output_grid_mask_path = _safe_path_join(output_dir_str, f"{slide_id}.{cfg.output_grid_mask_suffix}")
        
        output_png_dir = None
        if cfg.output_png_dir_suffix:
            output_png_dir = _safe_path_join(output_dir_str, f"{slide_id}.{cfg.output_png_dir_suffix}")
        
        output_thumbnail_path = None
        if cfg.output_thumbnail_suffix:
            output_thumbnail_path = _safe_path_join(output_dir_str, f"{slide_id}.{cfg.output_thumbnail_suffix}")
        
        create_visualizations(
            slide_path=r['slide_path'],
            final_coords=r['final_coords'],
            tessellate_h5_path=r['tessellate_h5_path'],
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
    logger.info(f"Output directory: {cfg.output_dir}")


def _main_batch_multi_model(cfg: TessellateExtractFeaturesConfig):
    """Process multiple slides with multiple models, optimized by grouping models with same patch size."""
    from collections import defaultdict
    from omegaconf import ListConfig
    from mussel.models.model_factory import get_default_patch_size, get_required_patch_encoder, ModelType
    
    if cfg.slide_paths is None or cfg.output_dir is None:
        raise ValueError("Batch mode requires slide_paths and output_dir")
    
    # Normalize model_type and slide_model_type to lists of ModelType enums
    patch_models = []
    if cfg.model_type is not None:
        if isinstance(cfg.model_type, (list, ListConfig)):
            # Convert strings to ModelType enums if needed
            patch_models = [ModelType[m] if isinstance(m, str) else m for m in cfg.model_type]
        else:
            patch_models = [cfg.model_type]
    
    slide_models = []
    if cfg.slide_model_type is not None:
        if isinstance(cfg.slide_model_type, (list, ListConfig)):
            # Convert strings to ModelType enums if needed
            slide_models = [ModelType[m] if isinstance(m, str) else m for m in cfg.slide_model_type]
        else:
            slide_models = [cfg.slide_model_type]
    
    all_models = patch_models + slide_models
    
    if not all_models:
        raise ValueError("Multi-model mode requires at least one model in model_type or slide_model_type")
    
    logger.info(f"Multi-model batch processing: {len(cfg.slide_paths)} slides × {len(all_models)} models")
    logger.info(f"Patch-level models: {[m.name for m in patch_models]}")
    logger.info(f"Slide-level models: {[m.name for m in slide_models]}")
    
    # Group models by required patch size
    models_by_patch_size = defaultdict(lambda: {'patch': [], 'slide': []})
    
    for model in patch_models:
        patch_size = get_default_patch_size(model)
        models_by_patch_size[patch_size]['patch'].append(model)
    
    for model in slide_models:
        # Slide models use their required patch encoder's patch size
        patch_encoder = get_required_patch_encoder(model)
        patch_size = get_default_patch_size(patch_encoder)
        models_by_patch_size[patch_size]['slide'].append(model)
    
    logger.info(f"\nGrouped by patch size:")
    for patch_size in sorted(models_by_patch_size.keys()):
        group = models_by_patch_size[patch_size]
        logger.info(f"  {patch_size}px:")
        if group['patch']:
            logger.info(f"    Patch models: {[m.name for m in group['patch']]}")
        if group['slide']:
            logger.info(f"    Slide models: {[m.name for m in group['slide']]}")
    
    # Calculate optimization benefit
    total_tessellations_without = len(all_models) * len(cfg.slide_paths)
    total_tessellations_with = len(models_by_patch_size) * len(cfg.slide_paths)
    savings_pct = (1 - total_tessellations_with / total_tessellations_without) * 100
    logger.info(f"\nTessellation optimization: {total_tessellations_with}/{total_tessellations_without} ({savings_pct:.0f}% fewer)")
    
    # Create output directory (only for local paths)
    output_dir_str = cfg.output_dir
    if not _is_remote_path(output_dir_str):
        output_dir_path = Path(output_dir_str)
        output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Process each patch size group
    for patch_size in sorted(models_by_patch_size.keys()):
        group = models_by_patch_size[patch_size]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing patch size: {patch_size}px")
        logger.info(f"{'='*80}")
        
        # Override patch size in config
        cfg_copy = OmegaConf.structured(cfg)
        cfg_copy.seg_config.patch_size = patch_size
        
        # Process patch-level models
        for model in group['patch']:
            logger.info(f"\n--- Patch model: {model.name} ---")
            cfg_copy.model_type = model
            cfg_copy.slide_model_type = None
            cfg_copy.aggregation_method = "identity"
            
            # Set output paths with model subdirectory
            model_output_dir = _safe_path_join(output_dir_str, model.name)
            
            # Create model subdirectory (only for local paths)
            if not _is_remote_path(model_output_dir):
                Path(model_output_dir).mkdir(parents=True, exist_ok=True)
            
            # Call regular batch processing for this model
            cfg_copy.output_h5_suffix = f"features.h5"
            cfg_copy.output_pt_suffix = f"features.pt"
            cfg_copy.output_dir = model_output_dir
            
            _main_batch(cfg_copy)
        
        # Process slide-level models (with slide batching!)
        for model in group['slide']:
            logger.info(f"\n--- Slide model: {model.name} ---")
            
            # Infer required patch encoder
            patch_encoder = get_required_patch_encoder(model)
            logger.info(f"Using patch encoder: {patch_encoder.name}")
            
            cfg_copy.model_type = patch_encoder
            cfg_copy.slide_model_type = model
            cfg_copy.aggregation_method = "model"
            
            # Separate output directories: patch encoder features and slide encoder features
            slide_output_dir = _safe_path_join(output_dir_str, model.name)
            patch_output_dir = _safe_path_join(output_dir_str, patch_encoder.name)
            cfg_copy.output_dir = slide_output_dir
            
            # Call regular batch processing with separate patch output directory
            _main_batch(cfg_copy, patch_output_dir=patch_output_dir)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Multi-model batch processing complete!")
    logger.info(f"Processed {len(cfg.slide_paths)} slides × {len(all_models)} models")
    logger.info(f"Output directory: {output_dir_str}")
    logger.info(f"{'='*80}")


def _main_single_multi_model(cfg: TessellateExtractFeaturesConfig):
    """Process single slide with multiple models, optimized by grouping models with same patch size."""
    # Convert to batch mode with single slide
    cfg_batch = OmegaConf.structured(cfg)
    cfg_batch.slide_paths = [cfg.slide_path]
    cfg_batch.slide_ids = [cfg.slide_id] if cfg.slide_id else None
    
    # Use parent directory of output paths as output_dir
    if cfg.output_h5_path:
        cfg_batch.output_dir = str(Path(cfg.output_h5_path).parent)
    else:
        raise ValueError("Single-slide multi-model mode requires output_h5_path")
    
    _main_batch_multi_model(cfg_batch)


if __name__ == "__main__":
    main()
