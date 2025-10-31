import os
import pickle
import ssl
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import h5py
import hydra
import numpy as np
import torch
import tiffslide
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import MISSING, OmegaConf
from shapely.geometry import Polygon

from mussel.cli.tessellate import (
    SegConfig,
    BiopsySegConfig,
    ResectionSegConfig,
    TcgaSegConfig,
    VisConfig,
    PngConfig,
)
from mussel.models import ModelType, get_required_patch_encoder
from mussel.utils import save_features, filter_features, save_hdf5
from mussel.utils.segment import draw_slide_mask, save_patches_png, segment_tissue

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
        postfilter_model_type (Optional[ModelType]): Type of model for post-filtering extraction. 
            If None and aggregation_method="model" with slide_model_type specified, automatically infers 
            the required patch encoder from slide_model_type. Otherwise, uses prefilter_model_type.
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
    # If postfilter_model_type is not specified, infer from slide_model_type when using model aggregation
    postfilter_model_type = cfg.postfilter_model_type
    if postfilter_model_type is None:
        # Check if we're using slide-level aggregation with a model
        if cfg.aggregation_method == "model" and cfg.slide_model_type is not None:
            # Infer the required patch encoder from the slide encoder
            postfilter_model_type = get_required_patch_encoder(cfg.slide_model_type)
            logger.info(
                f"Auto-inferring postfilter_model_type={postfilter_model_type.name} "
                f"from slide_model_type={cfg.slide_model_type.name}"
            )
        else:
            # Default to using the same model as prefilter
            postfilter_model_type = cfg.prefilter_model_type
    
    postfilter_model_path = cfg.postfilter_model_path if cfg.postfilter_model_path is not None else cfg.prefilter_model_path
    
    # Optimization: If filtering is enabled and models are the same, skip second extraction
    models_are_same = (postfilter_model_type == cfg.prefilter_model_type and 
                       postfilter_model_path == cfg.prefilter_model_path)
    skip_second_extraction = use_filtering and models_are_same
    
    # Determine total steps based on filtering and model optimization
    if not use_filtering:
        total_steps = 2  # tessellate → extract
    elif skip_second_extraction:
        total_steps = 3  # tessellate → extract → filter
    else:
        total_steps = 4  # tessellate → extract → filter → re-extract
    
    # Step 1: Tessellate
    logger.info(f"Step 1/{total_steps}: Tessellating whole-slide image...")
    if cfg.keep_intermediate_files:
        # Use a persistent path based on output path
        tessellate_h5_path = str(base_path / f"{Path(cfg.slide_path).stem}.tessellate.h5")
    else:
        tessellate_h5_path = os.path.join(temp_dir, "tessellate.h5")
    
    if values := segment_tissue(
        slide_path=cfg.slide_path,
        slide_id=cfg.slide_id,
        output_h5_path=tessellate_h5_path,
        **OmegaConf.to_container(cfg.seg_config),
    ):
        polygon, grid, coords, _ = values
    else:
        logger.error("Tessellation failed")
        if temp_dir:
            import shutil
            shutil.rmtree(temp_dir)
        return

    logger.info(f"Tessellation complete. Found {len(coords)} tiles.")

    # Optional: Save mask visualization (tissue segmentation boundary)
    if cfg.output_mask_path:
        mask = draw_slide_mask(
            cfg.slide_path,
            polygon,
            **OmegaConf.to_container(cfg.vis_config),
        )
        mask.save(cfg.output_mask_path)

    # Coordinate source for final extraction (will be updated if filtering is used)
    final_coords_h5_path = tessellate_h5_path
    final_coords = coords
    
    if use_filtering:
        # Step 2: Extract features (for filtering and possibly final output)
        logger.info(f"Step 2/{total_steps}: Extracting features using {cfg.prefilter_model_type.name}...")
        if cfg.keep_intermediate_files:
            prefilter_features_h5_path = str(base_path / f"{Path(cfg.slide_path).stem}.prefilter_features.h5")
            prefilter_features_pt_path = str(base_path / f"{Path(cfg.slide_path).stem}.prefilter_features.pt")
        else:
            prefilter_features_h5_path = os.path.join(temp_dir, "prefilter_features.h5")
            prefilter_features_pt_path = os.path.join(temp_dir, "prefilter_features.pt")

        save_features(
            slide_path=cfg.slide_path,
            gpu_device_id=cfg.gpu_device_id,
            model_type=cfg.prefilter_model_type,
            model_path=cfg.prefilter_model_path,
            use_gpu=cfg.use_gpu,
            output_h5_path=prefilter_features_h5_path,
            output_pt_path=prefilter_features_pt_path,
            patch_h5_path=tessellate_h5_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            gpu_device_ids=cfg.gpu_device_ids,
        )

        logger.info(f"Feature extraction complete.")

        # Step 3: Filter features
        logger.info(f"Step 3/{total_steps}: Filtering features using classifier...")
        logger.info(f"Loading classifier from {cfg.classifier_pkl}")
        with open(cfg.classifier_pkl, "rb") as f:
            classifier = pickle.load(f)

        with h5py.File(prefilter_features_h5_path, "r") as features_h5:
            if prefilter_features_pt_path and os.path.exists(prefilter_features_pt_path):
                features = torch.load(prefilter_features_pt_path, weights_only=True)
            else:
                features = np.array(features_h5["features"])
                features = torch.Tensor(features)
            logger.info(
                f"Loaded {features.shape[0]} features of dimension {features.shape[1]}"
            )
            filtered_features, filtered_coords = filter_features(
                features,
                features_h5["coords"][:],
                classifier,
                cfg.classifier_threshold,
            )

            logger.info(
                f"Filtering complete. {len(filtered_coords)} tiles passed the threshold (out of {len(coords)})."
            )

            if skip_second_extraction:
                # Optimization: Save filtered features directly to output (no re-extraction needed)
                logger.info("Using same model for pre-filter and post-filter - skipping second extraction")
                asset_dict = {"coords": filtered_coords}
                if cfg.save_features_to_h5:
                    asset_dict["features"] = filtered_features.numpy()
                save_hdf5(
                    cfg.output_h5_path,
                    asset_dict,
                    attr_h5_path=prefilter_features_h5_path,
                    mode="w",
                )
                torch.save(filtered_features, cfg.output_pt_path)
            else:
                # Save filtered coordinates to a temporary h5 file for second extraction
                if cfg.keep_intermediate_files:
                    filtered_coords_h5_path = str(base_path / f"{Path(cfg.slide_path).stem}.filtered_coords.h5")
                else:
                    filtered_coords_h5_path = os.path.join(temp_dir, "filtered_coords.h5")
                
                save_hdf5(
                    filtered_coords_h5_path,
                    {"coords": filtered_coords},
                    attr_h5_path=prefilter_features_h5_path,
                    mode="w",
                )
                
                # Update coordinate source for final extraction
                final_coords_h5_path = filtered_coords_h5_path
        
        # Update final coords
        final_coords = filtered_coords

    # Final step: Extract features (on all tiles or filtered tiles) - only if not already done
    if not skip_second_extraction:
        if use_filtering:
            step_num = 4
            logger.info(f"Step {step_num}/{total_steps}: Extracting features (second extraction) on filtered tiles using {postfilter_model_type.name}...")
        else:
            step_num = 2
            logger.info(f"Step {step_num}/{total_steps}: Extracting features using {postfilter_model_type.name}...")
        
        save_features(
            slide_path=cfg.slide_path,
            gpu_device_id=cfg.gpu_device_id,
            model_type=postfilter_model_type,
            model_path=postfilter_model_path,
            use_gpu=cfg.use_gpu,
            output_h5_path=cfg.output_h5_path,
            output_pt_path=cfg.output_pt_path,
            patch_h5_path=final_coords_h5_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            gpu_device_ids=cfg.gpu_device_ids,
            intermediate_h5_path=cfg.intermediate_h5_path,
            aggregation_method=cfg.aggregation_method,
            slide_model_type=cfg.slide_model_type,
            slide_model_path=cfg.slide_model_path,
        )

        logger.info(f"Feature extraction complete.")

    # Create grid visualization
    if cfg.output_grid_mask_path:
        logger.info(f"Creating grid mask with {len(final_coords)} tiles")
        # Read patch_size from the tessellate h5 file to create proper grid boxes
        with h5py.File(tessellate_h5_path, "r") as h5:
            native_patch_size = h5.attrs["patch_size"]
        
        # Create Polygon boxes for each coordinate
        grid_polygons = []
        for coord in final_coords:
            x, y = coord
            poly = Polygon([
                [x, y],
                [x, y + native_patch_size],
                [x + native_patch_size, y + native_patch_size],
                [x + native_patch_size, y],
            ])
            grid_polygons.append(poly)
        
        # Draw and save the grid mask
        grid_mask = draw_slide_mask(
            cfg.slide_path,
            grid_polygons,
            **OmegaConf.to_container(cfg.vis_config),
        )
        grid_mask.save(cfg.output_grid_mask_path)

    # Save PNG patches
    if cfg.output_png_dir:
        logger.info(f"Saving patches to {cfg.output_png_dir}")
        save_patches_png(
            cfg.slide_path,
            final_coords,
            save_dir=cfg.output_png_dir,
            num_workers=cfg.num_workers,
            patch_size=cfg.seg_config.patch_size,
            filter_black_white=cfg.png_config.filter_black_white,
            white_threshold=cfg.png_config.white_threshold,
            black_threshold=cfg.png_config.black_threshold,
        )

    # Optional: Save thumbnail
    if cfg.output_thumbnail_path:
        with tiffslide.TiffSlide(cfg.slide_path) as wsi:
            logger.info(f"Saving thumbnail to {cfg.output_thumbnail_path}")
            thumbnail = wsi.get_thumbnail(cfg.thumbnail_size)
            with open(cfg.output_thumbnail_path, "wb") as f:
                thumbnail.save(f)

    # Clean up temporary directory if not keeping intermediate files
    if temp_dir:
        import shutil
        shutil.rmtree(temp_dir)
        logger.info("Cleaned up temporary files.")


if __name__ == "__main__":
    main()
