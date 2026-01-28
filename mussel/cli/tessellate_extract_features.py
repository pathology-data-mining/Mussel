import logging
import os
import shutil
import ssl
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

import h5py
import torch
import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
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
from mussel.utils import (
    aggregate_slide_features_batch,
    extract_patch_features_batch,
    get_model_path_from_dir,
    save_torch_tensor,
    resolve_remote_paths,
)


def _is_remote_path(path):
    """Check if a path is a remote URL scheme."""
    if not isinstance(path, str):
        return False
    return path.startswith(
        ("az://", "abfs://", "s3://", "gs://", "http://", "https://")
    )


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
        result = str(base_path).rstrip("/")
        for part in parts:
            result = f"{result}/{str(part).lstrip('/')}"
        return result
    else:
        # For local paths, use Path
        return str(Path(base_path) / Path(*parts))


defaults = ["_self_", {"seg_config": "default"}]

# Note: get_model_path_from_dir is imported from mussel.utils


def get_classifier_pkl_from_model_dir(model_dir: Optional[str], classifier_pkl: Optional[str]) -> Optional[str]:
    """
    Get classifier pkl path. If classifier_pkl is None and model_dir is provided,
    look for classifier.pkl in model_dir.

    Args:
        model_dir: Directory containing pre-downloaded models
        classifier_pkl: Direct path to classifier pkl file

    Returns:
        Path to classifier pkl if found, None otherwise
    """
    if classifier_pkl is not None:
        return classifier_pkl
    
    if model_dir is None:
        return None
    
    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        return None
    
    # Check for classifier.pkl in model_dir
    classifier_file = model_dir_path / "classifier.pkl"
    if classifier_file.exists() and classifier_file.is_file():
        logger.info(f"✓ Using local classifier file: {classifier_file}")
        return str(classifier_file)
    
    return None



def get_batch_size_for_model(cfg, model_type) -> int:
    """
    Get the appropriate batch size for a given model type.

    Args:
        cfg: Configuration object
        model_type: ModelType enum or string name

    Returns:
        Batch size to use for this model (from model_batch_sizes if defined, else default batch_size)
    """
    # Get model name
    if hasattr(model_type, "name"):
        model_name = model_type.name
    else:
        model_name = str(model_type)

    # Return per-model batch size if defined, else default
    return cfg.model_batch_sizes.get(model_name, cfg.batch_size)


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

    Model Parameters:
        prefilter_model_type (ModelType): Type of model for pre-filtering feature extraction.
            This is a PATCH-LEVEL encoder. Use a single value (e.g., CTRANSPATH).
        model_type (Optional[ModelType] or List[ModelType]): Type of model(s) for PATCH-LEVEL feature extraction.
            - Single mode: Accepts a single ModelType (e.g., model_type=OPTIMUS)
            - Batch mode: Accepts a list of ModelTypes for multi-model extraction
                         (e.g., model_type=[OPTIMUS,VIRCHOW,UNI])
            - Command line usage: model_type=[OPTIMUS,VIRCHOW] (no quotes around the list)
            - When using slide encoders, the appropriate patch encoder is automatically selected
        slide_model_type (Optional[ModelType] or List[ModelType]): Type of model(s) for SLIDE-LEVEL encoding.
            - Single mode: Accepts a single ModelType (e.g., slide_model_type=GIGAPATH_SLIDE)
            - Batch mode: Accepts a list of ModelTypes for multi-model slide encoding
                         (e.g., slide_model_type=[GIGAPATH_SLIDE,TITAN_SLIDE])
            - Command line usage: slide_model_type=[GIGAPATH_SLIDE,TITAN_SLIDE] (no quotes)
            - Each slide encoder requires a specific patch encoder:
                * GIGAPATH_SLIDE requires GIGAPATH patch encoder
                * TITAN_SLIDE requires CONCH1_5 patch encoder
            - The required patch encoder is automatically paired and run as needed
        model_dir (Optional[str]): Directory containing pre-downloaded models.
            - When specified, the system looks for model subdirectories named after each model type
            - For example: /mnt/batch_models/GIGAPATH_SLIDE, /mnt/batch_models/CONCH1_5
            - Can also contain classifier_pkl file for filtering
            - This allows using cached/staged models instead of downloading from HuggingFace Hub
            - Particularly useful in batch processing environments (e.g., Azure Batch)
        pre_download_models (bool): Whether to pre-download models to model_dir before processing.
        intermediate_h5_path (Optional[str]): Path for intermediate patch features (single mode, two-step).
        aggregation_method (str): Aggregation method for post-filtering: identity (single-step), mean/max/model (two-step).
        ssl_verify (bool): Whether to verify SSL certificates when downloading models or accessing remote resources (default: True).

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
    prefilter_model_type: Optional[ModelType] = None  # No prefilter by default
    prefilter_model_path: Optional[str] = None  # Path to prefilter model file
    model_type: Any = None  # Can be ModelType or List[ModelType]
    model_dir: Optional[str] = None  # Directory containing pre-downloaded models
    pre_download_models: bool = False  # Whether to pre-download models to model_dir
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
    model_batch_sizes: Dict[str, int] = field(
        default_factory=dict
    )  # Per-model batch sizes (e.g., {"VIRCHOW2": 256, "OPTIMUS": 384})
    use_gpu: bool = True
    gpu_device_id: Optional[int] = None
    gpu_device_ids: Optional[List[int]] = None
    keep_intermediate_files: bool = False
    save_features_to_h5: bool = True  # Save final features to .h5 files; if False, only .pt files are saved (but tile_h5 is kept for patch encoders)
    seg_config: SegConfig = MISSING
    vis_config: VisConfig = field(default_factory=VisConfig)
    png_config: PngConfig = field(default_factory=PngConfig)
    intermediate_h5_path: Optional[str] = None
    aggregation_method: str = "identity"
    slide_model_type: Any = None  # Can be ModelType or List[ModelType]
    ssl_verify: bool = True  # Whether to verify SSL certificates for remote operations

    def __post_init__(self):
        """Set default patch size based on model type if not explicitly set."""
        from omegaconf import ListConfig, DictConfig

        # Convert ListConfig to actual lists of ModelType enums
        if isinstance(self.model_type, ListConfig):
            self.model_type = [ModelType[name] for name in self.model_type]
        elif isinstance(self.model_type, str):
            self.model_type = ModelType[self.model_type]

        if isinstance(self.slide_model_type, ListConfig):
            self.slide_model_type = [ModelType[name] for name in self.slide_model_type]
        elif isinstance(self.slide_model_type, str):
            self.slide_model_type = ModelType[self.slide_model_type]

        # Convert DictConfig to regular dict for model_batch_sizes
        if isinstance(self.model_batch_sizes, DictConfig):
            self.model_batch_sizes = dict(self.model_batch_sizes)

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
cs.store(
    name="tessellate_extract_features_config", node=TessellateExtractFeaturesConfig
)


@hydra.main(
    version_base=None, config_path=".", config_name="tessellate_extract_features_config"
)
def main(
    cfg: TessellateExtractFeaturesConfig,
):
    """Tessellate and extract features from one or more slides, optionally with filtering."""
    # Set multiprocessing start method to avoid permission issues in containers
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
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
            model_for_patch_size = (
                cfg.model_type
                if cfg.model_type is not None
                else cfg.prefilter_model_type
            )
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

    # Detect multi-model mode:
    # 1. If model_type or slide_model_type are lists
    # 2. If BOTH model_type and slide_model_type are specified (even as single values)
    # Check for both list and ListConfig since __post_init__ may not have run yet
    is_model_list = isinstance(cfg.model_type, (list, ListConfig))
    is_slide_model_list = isinstance(cfg.slide_model_type, (list, ListConfig))
    has_both_models = cfg.model_type is not None and cfg.slide_model_type is not None
    multi_model_mode = is_model_list or is_slide_model_list or has_both_models

    logger.debug(
        f"model_type type: {type(cfg.model_type)}, value: {cfg.model_type}, is_list: {is_model_list}"
    )
    logger.debug(
        f"slide_model_type type: {type(cfg.slide_model_type)}, value: {cfg.slide_model_type}, is_list: {is_slide_model_list}"
    )
    logger.debug(f"has_both_models: {has_both_models}")
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


@resolve_remote_paths('model_dir', 'classifier_pkl', auto_detect=False)
def _main_single(cfg: TessellateExtractFeaturesConfig):
    """Process a single slide."""
    # Check if output_dir is provided instead of output_h5_path/output_pt_path
    if cfg.output_dir:
        # Convert single-slide mode with output_dir to use explicit paths
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate slide_id if not provided
        slide_id = cfg.slide_id if cfg.slide_id else Path(cfg.slide_path).stem

        # Generate output paths from output_dir
        cfg.output_h5_path = str(output_dir / "h5" / f"{slide_id}.{cfg.output_h5_suffix}")
        cfg.output_pt_path = str(output_dir / "pt" / f"{slide_id}.{cfg.output_pt_suffix}")

        # Create subdirectories
        Path(cfg.output_h5_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.output_pt_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Using output_dir: {cfg.output_dir}")
        logger.info(f"Generated output_h5_path: {cfg.output_h5_path}")
        logger.info(f"Generated output_pt_path: {cfg.output_pt_path}")
    
    if (
        cfg.slide_path is None
        or cfg.output_h5_path is None
        or cfg.output_pt_path is None
    ):
        raise ValueError(
            "Single-slide mode requires slide_path and either (output_h5_path, output_pt_path) or output_dir"
        )

    # Create temporary directory for intermediate files if not keeping them
    temp_dir = None
    base_path = Path(cfg.output_h5_path).parent

    if not cfg.keep_intermediate_files:
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Using temporary directory for intermediate files: {temp_dir}")

    # Resolve classifier_pkl from model_dir if not explicitly provided
    classifier_pkl = get_classifier_pkl_from_model_dir(cfg.model_dir, cfg.classifier_pkl)
    
    # Determine if filtering is enabled
    use_filtering = classifier_pkl is not None

    # Determine models for each extraction step
    model_type = cfg.model_type
    if model_type is None:
        if cfg.slide_model_type is not None:
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

    # Resolve model paths from model_dir
    model_path = get_model_path_from_dir(cfg.model_dir, model_type) if model_type else None
    if model_path is None and cfg.model_dir and model_type:
        logger.info(f"Model {model_type.name} not found in model_dir, will download from HuggingFace")
    elif model_path is None and model_type:
        logger.info(f"No model_dir configured, will download {model_type.name} from HuggingFace")
    
    prefilter_model_path = cfg.prefilter_model_path if cfg.prefilter_model_path else (
        get_model_path_from_dir(cfg.model_dir, cfg.prefilter_model_type) if cfg.prefilter_model_type else None
    )
    if prefilter_model_path is None and cfg.prefilter_model_type and cfg.model_dir:
        logger.info(f"Model {cfg.prefilter_model_type.name} not found in model_dir, will download from HuggingFace")
    elif prefilter_model_path is None and cfg.prefilter_model_type and not cfg.model_dir:
        logger.info(f"No model_dir configured, will download {cfg.prefilter_model_type.name} from HuggingFace")

    # Resolve slide model path if using model aggregation
    slide_model_path = None
    if cfg.slide_model_type:
        slide_model_path = get_model_path_from_dir(cfg.model_dir, cfg.slide_model_type)
        if slide_model_path is None and cfg.model_dir:
            logger.info(f"Model {cfg.slide_model_type.name} not found in model_dir, will download from HuggingFace")
            slide_model_path = cfg.slide_model_type.path
        elif slide_model_path is None:
            logger.info(f"No model_dir configured, will download {cfg.slide_model_type.name} from HuggingFace")
            slide_model_path = cfg.slide_model_type.path

    # Optimization: If filtering is enabled and models are the same, skip second extraction
    models_are_same = (
        model_type == cfg.prefilter_model_type
        and model_path == prefilter_model_path
    )
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
        prefilter_model_path=prefilter_model_path,
        model_type=model_type,
        model_path=model_path,
        skip_second_extraction=skip_second_extraction,
        output_mask_path=cfg.output_mask_path,
        two_step_mode=False,  # Single-slide mode doesn't use two-step
        slide_model_path=slide_model_path,
    )

    if result is None:
        # Processing failed or was completed early
        if temp_dir:

            shutil.rmtree(temp_dir)
        return

    # Create visualizations
    create_visualizations(
        slide_path=cfg.slide_path,
        final_coords=result["final_coords"],
        tessellate_h5_path=result["tessellate_h5_path"],
        cfg=cfg,
        output_grid_mask_path=cfg.output_grid_mask_path,
        output_png_dir=cfg.output_png_dir,
        output_thumbnail_path=cfg.output_thumbnail_path,
    )

    # Clean up temporary directory if not keeping intermediate files
    if temp_dir:

        shutil.rmtree(temp_dir)
        logger.info("Cleaned up temporary files.")


@resolve_remote_paths('model_dir', 'classifier_pkl', auto_detect=False)
def _main_batch(
    cfg: TessellateExtractFeaturesConfig, patch_output_dir: Optional[str] = None
):
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

    # Resolve classifier_pkl from model_dir if not explicitly provided
    classifier_pkl = get_classifier_pkl_from_model_dir(cfg.model_dir, cfg.classifier_pkl)
    
    # Determine if filtering is enabled
    use_filtering = classifier_pkl is not None

    # Determine models for each extraction step
    model_type = cfg.model_type
    if model_type is None:
        if cfg.slide_model_type is not None:
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

    # Resolve model paths from model_dir
    model_path = get_model_path_from_dir(cfg.model_dir, model_type) if model_type else None
    if model_path is None and cfg.model_dir and model_type:
        logger.info(f"Model {model_type.name} not found in model_dir, will download from HuggingFace")
    elif model_path is None and model_type:
        logger.info(f"No model_dir configured, will download {model_type.name} from HuggingFace")
    
    prefilter_model_path = cfg.prefilter_model_path if cfg.prefilter_model_path else (
        get_model_path_from_dir(cfg.model_dir, cfg.prefilter_model_type) if cfg.prefilter_model_type else None
    )
    if prefilter_model_path is None and cfg.prefilter_model_type and cfg.model_dir:
        logger.info(f"Model {cfg.prefilter_model_type.name} not found in model_dir, will download from HuggingFace")
    elif prefilter_model_path is None and cfg.prefilter_model_type and not cfg.model_dir:
        logger.info(f"No model_dir configured, will download {cfg.prefilter_model_type.name} from HuggingFace")

    # Resolve slide model path if using model aggregation
    slide_model_path = None
    if cfg.slide_model_type:
        slide_model_path = get_model_path_from_dir(cfg.model_dir, cfg.slide_model_type)
        if slide_model_path is None and cfg.model_dir:
            logger.info(f"Model {cfg.slide_model_type.name} not found in model_dir, will download from HuggingFace")
            slide_model_path = cfg.slide_model_type.path
        elif slide_model_path is None:
            logger.info(f"No model_dir configured, will download {cfg.slide_model_type.name} from HuggingFace")
            slide_model_path = cfg.slide_model_type.path

    # Optimization: If filtering is enabled and models are the same, skip second extraction
    models_are_same = (
        model_type == cfg.prefilter_model_type
        and model_path == prefilter_model_path
    )
    skip_second_extraction = use_filtering and models_are_same

    # Generate slide IDs if not provided
    slide_ids = cfg.slide_ids
    if slide_ids is None:
        slide_ids = [Path(sp).stem for sp in cfg.slide_paths]

    # Phase 1: Tessellate all slides and perform filtering if needed
    logger.info(f"\n=== Phase 1: Tessellating {len(cfg.slide_paths)} slides ===")

    slide_results = []
    for i, (slide_path, slide_id) in enumerate(zip(cfg.slide_paths, slide_ids)):
        logger.info(f"\nTessellating slide {i + 1}/{len(cfg.slide_paths)}: {slide_id}")

        try:
            # Determine output mask path
            output_mask_path = None
            if cfg.output_mask_suffix:
                output_mask_path = _safe_path_join(
                    output_dir_str, f"{slide_id}.{cfg.output_mask_suffix}"
                )

            # Tessellate and filter (but don't extract features yet)
            result = process_slide_tessellation_only(
                slide_path=slide_path,
                slide_id=slide_id,
                cfg=cfg,
                temp_dir=temp_dir,
                base_path=output_dir_str,
                use_filtering=use_filtering,
                prefilter_model_type=cfg.prefilter_model_type,
                prefilter_model_path=prefilter_model_path,
                skip_second_extraction=skip_second_extraction,
                output_mask_path=output_mask_path,
            )

            if result is None:
                logger.warning(f"Skipping slide {slide_id} due to tessellation failure")
                continue
        except Exception as e:
            logger.error(f"Error processing slide {slide_id}: {e}")
            logger.warning(f"Skipping slide {slide_id} and continuing with remaining slides")
            continue

        # Add output paths to result
        result["slide_id"] = slide_id
        result["output_h5_path"] = _safe_path_join(
            output_dir_str, "h5", f"{slide_id}.{cfg.output_h5_suffix}"
        )
        result["output_pt_path"] = _safe_path_join(
            output_dir_str, "pt", f"{slide_id}.{cfg.output_pt_suffix}"
        )
        slide_results.append(result)

    if not slide_results:
        logger.error("No slides to process after tessellation phase")
        if temp_dir:

            shutil.rmtree(temp_dir)
        return

    # Create output subdirectories (h5, pt, tile_h5) if they're local paths
    # When save_features_to_h5=false, create tile_h5 and pt (skip h5)
    subdirs = ["tile_h5", "pt"] if not cfg.save_features_to_h5 else ["h5", "pt"]
    for subdir in subdirs:
        subdir_path = _safe_path_join(output_dir_str, subdir)
        if not _is_remote_path(subdir_path):
            Path(subdir_path).mkdir(parents=True, exist_ok=True)

    # Phase 2: Batch extract patch features for all slides
    logger.info(
        f"\n=== Phase 2: Batch extracting patch features for {len(slide_results)} slides ==="
    )
    logger.info(f"Model: {model_type.name}")
    logger.info(f"Batch size: {get_batch_size_for_model(cfg, model_type)}")

    # Prepare paths for batch extraction
    patch_h5_paths = [r["final_coords_h5_path"] for r in slide_results]
    slide_paths = [r["slide_path"] for r in slide_results]

    # Track all h5 files created for potential feature stripping
    h5_files_to_strip = []

    # Determine if we need two-step processing (patch features + slide aggregation)
    # If a slide model is specified, force two-step processing with model aggregation
    if cfg.slide_model_type is not None and cfg.aggregation_method == "identity":
        logger.info(f"Auto-enabling two-step processing with aggregation_method='model' for slide_model_type={cfg.slide_model_type.name}")
        cfg.aggregation_method = "model"
    
    use_two_step = cfg.aggregation_method != "identity"

    if use_two_step:
        # Extract to intermediate patch feature files for later aggregation
        # Use patch_output_dir_str for patch features (separate from slide features for slide models)
        intermediate_h5_paths = [
            _safe_path_join(
                patch_output_dir_str, "tile_h5", f"{r['slide_id']}.patch.h5"
            )
            for r in slide_results
        ]

        # Create tile_h5 subdirectory if it's a local path
        tile_h5_dir = _safe_path_join(patch_output_dir_str, "tile_h5")
        if not _is_remote_path(tile_h5_dir):
            Path(tile_h5_dir).mkdir(parents=True, exist_ok=True)

        try:
            extract_patch_features_batch(
                patch_h5_paths=patch_h5_paths,
                slide_paths=slide_paths,
                output_h5_paths=intermediate_h5_paths,
                model_type=model_type,
                model_path=model_path,
                batch_size=get_batch_size_for_model(cfg, model_type),
                use_gpu=cfg.use_gpu,
                gpu_device_id=cfg.gpu_device_id,
                gpu_device_ids=cfg.gpu_device_ids,
                num_workers=cfg.num_workers,
                pin_memory=True,
                is_test_run=False,
            )
        except Exception as e:
            logger.error(f"Error during batch feature extraction: {e}")
            logger.warning("Continuing with available results...")

        # Add intermediate paths to results and track for feature stripping
        for r, intermediate_h5_path in zip(slide_results, intermediate_h5_paths):
            r["intermediate_h5_path"] = intermediate_h5_path
            if not cfg.save_features_to_h5:
                h5_files_to_strip.append(intermediate_h5_path)

        # If patch_output_dir is specified, also save patch encoder features
        if patch_output_dir:
            logger.info(
                f"\n=== Phase 2b: Saving patch encoder features to {patch_output_dir} ==="
            )

            # Create h5 and pt subdirectories if they're local paths
            subdirs = ["tile_h5", "pt"] if not cfg.save_features_to_h5 else ["tile_h5", "h5", "pt"]
            for subdir in subdirs:
                subdir_path = _safe_path_join(patch_output_dir_str, subdir)
                if not _is_remote_path(subdir_path):
                    Path(subdir_path).mkdir(parents=True, exist_ok=True)

            # For tile encoders, just copy the intermediate h5 files and extract PT files
            # No aggregation needed since they already contain tile-level features
            for i, r in enumerate(slide_results):
                intermediate_h5_path = intermediate_h5_paths[i]
                
                # Copy tile_h5 file (only if source and destination are different)
                tile_h5_dest = _safe_path_join(
                    patch_output_dir_str,
                    "tile_h5",
                    f"{r['slide_id']}.patch.h5",
                )
                if not _is_remote_path(tile_h5_dest):
                    # Only copy if source and destination are different
                    if os.path.abspath(intermediate_h5_path) != os.path.abspath(tile_h5_dest):
                        shutil.copy2(intermediate_h5_path, tile_h5_dest)
                        logger.debug(f"Copied tile features to {tile_h5_dest}")
                    else:
                        logger.debug(f"Tile features already at destination: {tile_h5_dest}")
                
                # Extract and save PT file
                pt_dest = _safe_path_join(
                    patch_output_dir_str,
                    "pt",
                    f"{r['slide_id']}.{cfg.output_pt_suffix}",
                )
                with h5py.File(intermediate_h5_path, "r") as f:
                    features = torch.from_numpy(f["features"][:])
                save_torch_tensor(pt_dest, features)
                logger.debug(f"Saved PT features to {pt_dest}")
                
                # Optionally copy/create h5 file with features
                if cfg.save_features_to_h5:
                    h5_dest = _safe_path_join(
                        patch_output_dir_str,
                        "h5",
                        f"{r['slide_id']}.{cfg.output_h5_suffix}",
                    )
                    if not _is_remote_path(h5_dest):
                        shutil.copy2(intermediate_h5_path, h5_dest)
                        logger.debug(f"Copied h5 features to {h5_dest}")
                else:
                    # Track the tile_h5 destination for stripping
                    tile_h5_dest = _safe_path_join(
                        patch_output_dir_str,
                        "tile_h5",
                        f"{r['slide_id']}.patch.h5",
                    )
                    if tile_h5_dest not in h5_files_to_strip and not _is_remote_path(tile_h5_dest):
                        h5_files_to_strip.append(tile_h5_dest)

        # Phase 3: Batch aggregate to slide level OR copy patch features for tile encoders
        if patch_output_dir and cfg.slide_model_type is None:
            # For tile encoders (GIGAPATH, CONCH1_5) without slide aggregation,
            # just copy the patch h5 files since they're already tile-level features
            logger.info(
                f"\n=== Phase 3: Copying tile features to slide output (tile encoder) ==="
            )
            logger.info(f"Patch encoder: {model_type.name}")
            
            for i, r in enumerate(slide_results):
                intermediate_h5_path = intermediate_h5_paths[i]
                
                # Copy h5 file if save_features_to_h5 is enabled
                if cfg.save_features_to_h5:
                    h5_dest = r["output_h5_path"]
                    if not _is_remote_path(h5_dest):
                        shutil.copy2(intermediate_h5_path, h5_dest)
                        logger.debug(f"Copied h5 to {h5_dest}")
                
                # Copy PT file
                pt_dest = r["output_pt_path"]
                with h5py.File(intermediate_h5_path, "r") as f:
                    features = torch.from_numpy(f["features"][:])
                save_torch_tensor(pt_dest, features)
                logger.debug(f"Saved PT to {pt_dest}")
        elif cfg.aggregation_method == "model" and cfg.slide_model_type is None:
            # aggregation_method=model requires a slide_model_type
            logger.error(f"aggregation_method='model' requires slide_model_type to be specified")
            logger.error(f"Current model_type={model_type.name} is a tile encoder, not a slide encoder")
            logger.error(f"Either specify a slide_model_type or use aggregation_method='mean'/'max'")
            raise ValueError(
                f"aggregation_method='model' requires slide_model_type. "
                f"Current model_type={model_type.name} cannot be used for slide-level aggregation."
            )
        else:
            # True slide aggregation for models that aggregate patches
            logger.info(
                f"\n=== Phase 3: Batch aggregating {len(slide_results)} slides (aggregation_method={cfg.aggregation_method}) ==="
            )
            if cfg.slide_model_type:
                logger.info(f"Slide model: {cfg.slide_model_type.name}")
                logger.info(f"Patch encoder: {model_type.name}")
            logger.info(f"Slide batch size: {cfg.slide_batch_size}")

            # Skip H5 output if save_features_to_h5=false mode is enabled
            output_h5_paths = None if not cfg.save_features_to_h5 else [r["output_h5_path"] for r in slide_results]
            output_pt_paths = [r["output_pt_path"] for r in slide_results]

            try:
                aggregate_slide_features_batch(
                    patch_features_h5_paths=intermediate_h5_paths,
                    output_h5_paths=output_h5_paths,
                    output_pt_paths=output_pt_paths,
                    aggregation_method=cfg.aggregation_method,
                    model_type=cfg.slide_model_type,
                    model_path=slide_model_path,
                    model_dir=cfg.model_dir,
                    use_gpu=cfg.use_gpu,
                    gpu_device_id=cfg.gpu_device_id,
                    gpu_device_ids=cfg.gpu_device_ids,
                    slide_batch_size=cfg.slide_batch_size,
                )
            except Exception as e:
                logger.error(f"Error during batch aggregation: {e}")
                logger.warning("Continuing with available results...")
        
        # If save_features_to_h5=false, strip features from all tracked h5 files (keep coords only)
        # This must be done AFTER aggregation since aggregation needs the features
        if not cfg.save_features_to_h5 and h5_files_to_strip:
            logger.info(f"Converting {len(h5_files_to_strip)} h5 files to coords-only (save_features_to_h5=false)")
            for h5_path in h5_files_to_strip:
                # Check if file still has features before stripping
                try:
                    if not os.path.exists(h5_path):
                        logger.debug(f"Skipping non-existent file: {h5_path}")
                        continue
                    
                    with h5py.File(h5_path, "r") as f:
                        if "features" in f:
                            coords = f["coords"][:]
                            has_features = True
                        else:
                            has_features = False
                    
                    if has_features:
                        # Delete file and recreate with coords only
                        os.remove(h5_path)
                        with h5py.File(h5_path, "w") as f:
                            f.create_dataset("coords", data=coords, compression="gzip")
                        logger.debug(f"Stripped features from {h5_path}")
                except Exception as e:
                    logger.warning(f"Could not strip features from {h5_path}: {e}")
            logger.info("Feature stripping complete")
    else:
        # Single-step: extract directly to final output (no aggregation)
        logger.info(f"\n=== Single-step extraction: {model_type.name} ===")
        logger.info(f"Batch size: {get_batch_size_for_model(cfg, model_type)}")
        
        if not cfg.save_features_to_h5:
            # save_features_to_h5=false mode: extract to temp, save PT + coords-only H5
            temp_h5_dir = tempfile.mkdtemp()
            temp_output_h5_paths = [
                os.path.join(temp_h5_dir, f"{r['slide_id']}.features.h5")
                for r in slide_results
            ]
            
            # Create tile_h5 subdirectory for coords-only H5 files
            tile_h5_dir = _safe_path_join(output_dir_str, "tile_h5")
            if not _is_remote_path(tile_h5_dir):
                Path(tile_h5_dir).mkdir(parents=True, exist_ok=True)
            
            coords_h5_paths = [
                _safe_path_join(output_dir_str, "tile_h5", f"{r['slide_id']}.patch.h5")
                for r in slide_results
            ]
        else:
            # save_features_to_h5=true mode: save features H5 + coords-only H5 for batch mode
            temp_output_h5_paths = [r["output_h5_path"] for r in slide_results]

            # Also create coords-only H5 files (patch.h5) in batch mode for downstream processes
            tile_h5_dir = _safe_path_join(output_dir_str, "tile_h5")
            if not _is_remote_path(tile_h5_dir):
                Path(tile_h5_dir).mkdir(parents=True, exist_ok=True)

            coords_h5_paths = [
                _safe_path_join(output_dir_str, "tile_h5", f"{r['slide_id']}.patch.h5")
                for r in slide_results
            ]

        extract_patch_features_batch(
            patch_h5_paths=patch_h5_paths,
            slide_paths=slide_paths,
            output_h5_paths=temp_output_h5_paths,
            model_type=model_type,
            model_path=model_path,
            batch_size=get_batch_size_for_model(cfg, model_type),
            use_gpu=cfg.use_gpu,
            gpu_device_id=cfg.gpu_device_id,
            gpu_device_ids=cfg.gpu_device_ids,
            num_workers=cfg.num_workers,
            pin_memory=True,
            is_test_run=False,
        )

        # Save PT files and coords-only H5
        for i, r in enumerate(slide_results):
            with h5py.File(temp_output_h5_paths[i], "r") as f:
                features = torch.from_numpy(f["features"][:])
                save_torch_tensor(r["output_pt_path"], features)

                # Always create coords-only H5 in batch mode for downstream processes
                coords = f["coords"][:]
                with h5py.File(coords_h5_paths[i], "w") as f_out:
                    f_out.create_dataset("coords", data=coords, compression="gzip")
                    logger.debug(f"Saved coords-only H5 to {coords_h5_paths[i]}")
        
        # Clean up temp H5 files if save_features_to_h5=false
        if not cfg.save_features_to_h5:
            shutil.rmtree(temp_h5_dir)
            logger.info("Cleaned up temporary H5 files (save_features_to_h5=false mode)")

    # Phase 4: Create visualizations
    logger.info(f"\n=== Phase 4: Creating visualizations ===")
    for r in slide_results:
        slide_id = r["slide_id"]

        output_grid_mask_path = None
        if cfg.output_grid_mask_suffix:
            output_grid_mask_path = _safe_path_join(
                output_dir_str, f"{slide_id}.{cfg.output_grid_mask_suffix}"
            )

        output_png_dir = None
        if cfg.output_png_dir_suffix:
            output_png_dir = _safe_path_join(
                output_dir_str, f"{slide_id}.{cfg.output_png_dir_suffix}"
            )

        output_thumbnail_path = None
        if cfg.output_thumbnail_suffix:
            output_thumbnail_path = _safe_path_join(
                output_dir_str, f"{slide_id}.{cfg.output_thumbnail_suffix}"
            )

        create_visualizations(
            slide_path=r["slide_path"],
            final_coords=r["final_coords"],
            tessellate_h5_path=r["tessellate_h5_path"],
            cfg=cfg,
            output_grid_mask_path=output_grid_mask_path,
            output_png_dir=output_png_dir,
            output_thumbnail_path=output_thumbnail_path,
        )

    # Clean up temporary directory if not keeping intermediate files
    if temp_dir:

        shutil.rmtree(temp_dir)
        logger.info("Cleaned up temporary files.")

    logger.info(f"\n=== Batch processing complete! ===")
    logger.info(f"Processed {len(cfg.slide_paths)} slides")
    logger.info(f"Output directory: {cfg.output_dir}")


@resolve_remote_paths('model_dir', 'classifier_pkl', auto_detect=False)
def _main_batch_multi_model(cfg: TessellateExtractFeaturesConfig):
    """Process multiple slides with multiple models, optimized by grouping models with same patch size."""
    from collections import defaultdict
    from omegaconf import ListConfig
    from mussel.models.model_factory import (
        get_default_patch_size,
        get_required_patch_encoder,
        ModelType,
    )

    if cfg.slide_paths is None or cfg.output_dir is None:
        raise ValueError("Batch mode requires slide_paths and output_dir")

    # Normalize model_type and slide_model_type to lists of ModelType enums
    patch_models = []
    if cfg.model_type is not None:
        if isinstance(cfg.model_type, (list, ListConfig)):
            # Convert strings to ModelType enums if needed
            patch_models = [
                ModelType[m] if isinstance(m, str) else m for m in cfg.model_type
            ]
        else:
            patch_models = [cfg.model_type]

    slide_models = []
    if cfg.slide_model_type is not None:
        if isinstance(cfg.slide_model_type, (list, ListConfig)):
            # Convert strings to ModelType enums if needed
            slide_models = [
                ModelType[m] if isinstance(m, str) else m for m in cfg.slide_model_type
            ]
        else:
            slide_models = [cfg.slide_model_type]

    all_models = patch_models + slide_models

    if not all_models:
        raise ValueError(
            "Multi-model mode requires at least one model in model_type or slide_model_type"
        )

    logger.info(
        f"Multi-model batch processing: {len(cfg.slide_paths)} slides × {len(all_models)} models"
    )
    logger.info(f"Patch-level models: {[m.name for m in patch_models]}")
    logger.info(f"Slide-level models: {[m.name for m in slide_models]}")

    # Group models by required patch size
    models_by_patch_size = defaultdict(lambda: {"patch": [], "slide": []})

    for model in patch_models:
        patch_size = get_default_patch_size(model)
        models_by_patch_size[patch_size]["patch"].append(model)

    for model in slide_models:
        # Slide models use their required patch encoder's patch size
        patch_encoder = get_required_patch_encoder(model)
        patch_size = get_default_patch_size(patch_encoder)
        models_by_patch_size[patch_size]["slide"].append(model)

    logger.info(f"\nGrouped by patch size:")
    for patch_size in sorted(models_by_patch_size.keys()):
        group = models_by_patch_size[patch_size]
        logger.info(f"  {patch_size}px:")
        if group["patch"]:
            logger.info(f"    Patch models: {[m.name for m in group['patch']]}")
        if group["slide"]:
            logger.info(f"    Slide models: {[m.name for m in group['slide']]}")

    # Calculate optimization benefit
    total_tessellations_without = len(all_models) * len(cfg.slide_paths)
    total_tessellations_with = len(models_by_patch_size) * len(cfg.slide_paths)
    savings_pct = (1 - total_tessellations_with / total_tessellations_without) * 100
    logger.info(
        f"\nTessellation optimization: {total_tessellations_with}/{total_tessellations_without} ({savings_pct:.0f}% fewer)"
    )

    # Create output directory (only for local paths)
    output_dir_str = cfg.output_dir
    if not _is_remote_path(output_dir_str):
        output_dir_path = Path(output_dir_str)
        output_dir_path.mkdir(parents=True, exist_ok=True)

    # Process each patch size group
    for patch_size in sorted(models_by_patch_size.keys()):
        group = models_by_patch_size[patch_size]

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing patch size: {patch_size}px")
        logger.info(f"{'=' * 80}")

        # Override patch size in config
        cfg_copy = OmegaConf.structured(cfg)
        cfg_copy.seg_config.patch_size = patch_size

        # Process patch-level models
        for model in group["patch"]:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing patch model: {model.name}")
            logger.info(f"Patch size: {patch_size}px")
            logger.info(f"{'=' * 60}")
            
            try:
                cfg_copy.model_type = model
                cfg_copy.slide_model_type = None
                cfg_copy.aggregation_method = "identity"

                # Model path will be resolved from model_dir in _main_batch
                # No need to set model_path here

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
            except Exception as e:
                logger.error(f"Error processing patch model {model.name}: {e}")
                logger.warning(f"Skipping model {model.name} and continuing with remaining models")

        # Process slide-level models (with slide batching!)
        for model in group["slide"]:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing slide model: {model.name}")
            
            try:
                # Infer required patch encoder
                patch_encoder = get_required_patch_encoder(model)
                logger.info(f"Patch encoder: {patch_encoder.name}")
                logger.info(f"Patch size: {patch_size}px")
                logger.info(f"{'=' * 60}")

                cfg_copy.model_type = patch_encoder
                cfg_copy.slide_model_type = model
                cfg_copy.aggregation_method = "model"

                # Model paths will be resolved from model_dir in _main_batch
                # No need to set model_path or slide_model_path here

                # Separate output directories: patch encoder features and slide encoder features
                slide_output_dir = _safe_path_join(output_dir_str, model.name)
                patch_output_dir = _safe_path_join(output_dir_str, patch_encoder.name)
                cfg_copy.output_dir = slide_output_dir

                # Call regular batch processing with separate patch output directory
                _main_batch(cfg_copy, patch_output_dir=patch_output_dir)
            except Exception as e:
                logger.error(f"Error processing slide model {model.name}: {e}")
                logger.warning(f"Skipping model {model.name} and continuing with remaining models")

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Multi-model batch processing complete!")
    logger.info(f"Processed {len(cfg.slide_paths)} slides × {len(all_models)} models")
    logger.info(f"Output directory: {output_dir_str}")
    logger.info(f"{'=' * 80}")


def _main_single_multi_model(cfg: TessellateExtractFeaturesConfig):
    """Process single slide with multiple models, optimized by grouping models with same patch size."""
    # Convert to batch mode with single slide
    cfg_batch = OmegaConf.structured(cfg)
    cfg_batch.slide_paths = [cfg.slide_path]
    cfg_batch.slide_ids = [cfg.slide_id] if cfg.slide_id else None

    # Use parent directory of output paths as output_dir, or output_dir if provided
    if cfg.output_h5_path:
        cfg_batch.output_dir = str(Path(cfg.output_h5_path).parent)
    elif cfg.output_dir:
        cfg_batch.output_dir = str(cfg.output_dir)
    else:
        raise ValueError("Single-slide multi-model mode requires output_h5_path or output_dir")

    _main_batch_multi_model(cfg_batch)


if __name__ == "__main__":
    main()
