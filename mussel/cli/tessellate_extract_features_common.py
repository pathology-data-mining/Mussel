"""Common functionality for tessellate-extract-features workflows."""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
import tiffslide
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
from shapely.geometry import Polygon

from mussel.utils import (
    save_features,
    filter_features,
    save_hdf5,
    save_torch_tensor,
    is_remote_path,
    safe_path_join,
)
from mussel.utils.segment import draw_slide_mask, save_patches_png, segment_tissue


def process_slide_tessellation_and_filtering(
    slide_path: str,
    slide_id: Optional[str],
    output_h5_path: str,
    output_pt_path: str,
    cfg,
    temp_dir: str,
    base_path: Union[str, Path],
    use_filtering: bool,
    prefilter_model_type,
    prefilter_model_path: Optional[str],
    model_type,
    model_path: Optional[str],
    skip_second_extraction: bool,
    output_mask_path: Optional[str] = None,
    two_step_mode: bool = False,
    slide_model_path: Optional[str] = None,
) -> Optional[dict]:
    """
    Process a single slide through tessellation, feature extraction, and optional filtering.
    
    This function contains the core logic shared between single-slide and batch processing.
    
    Args:
        slide_path: Path to the whole-slide image
        slide_id: Optional slide identifier
        output_h5_path: Path to save final HDF5 output
        output_pt_path: Path to save final PyTorch output
        cfg: Configuration object with seg_config, vis_config, etc.
        temp_dir: Temporary directory for intermediate files
        base_path: Base path for output files
        use_filtering: Whether to apply filtering
        prefilter_model_type: Model type for pre-filtering
        prefilter_model_path: Path to pre-filter model weights
        model_type: Model type for post-filtering
        model_path: Path to post-filter model weights
        skip_second_extraction: Whether to skip second extraction (when models are same)
        output_mask_path: Optional path to save mask visualization
        two_step_mode: Whether using two-step aggregation (for batch processing)
        slide_model_path: Path to slide encoder model weights (for slide-level aggregation)
        
    Returns:
        Dict with intermediate paths for batch aggregation if needed, None otherwise
    """
    # Step 1: Tessellate
    logger.info(f"Tessellating slide: {slide_path}")
    if cfg.keep_intermediate_files:
        tessellate_h5_path = safe_path_join(base_path, f"{Path(slide_path).stem}.tessellate.h5")
    else:
        tessellate_h5_path = os.path.join(temp_dir, f"{Path(slide_path).stem}.tessellate.h5")
    
    if values := segment_tissue(
        slide_path=slide_path,
        slide_id=slide_id,
        output_h5_path=tessellate_h5_path,
        **OmegaConf.to_container(cfg.seg_config),
    ):
        polygon, grid, coords, _ = values
    else:
        logger.error(f"Tessellation failed for {slide_path}")
        return None
    
    logger.info(f"Tessellation complete. Found {len(coords)} tiles.")
    
    # Optional: Save mask visualization
    if output_mask_path:
        mask = draw_slide_mask(
            slide_path,
            polygon,
            **OmegaConf.to_container(cfg.vis_config),
        )
        mask.save(output_mask_path)
    
    # Coordinate source for final extraction
    final_coords_h5_path = tessellate_h5_path
    final_coords = coords
    
    if use_filtering:
        # Extract features for filtering
        logger.info(f"Extracting features for filtering: {slide_path}")
        if cfg.keep_intermediate_files:
            prefilter_features_h5_path = safe_path_join(base_path, f"{Path(slide_path).stem}.prefilter_features.h5")
            prefilter_features_pt_path = safe_path_join(base_path, f"{Path(slide_path).stem}.prefilter_features.pt")
        else:
            prefilter_features_h5_path = os.path.join(temp_dir, f"{Path(slide_path).stem}.prefilter_features.h5")
            prefilter_features_pt_path = os.path.join(temp_dir, f"{Path(slide_path).stem}.prefilter_features.pt")
        
        save_features(
            slide_path=slide_path,
            gpu_device_id=cfg.gpu_device_id,
            model_type=prefilter_model_type,
            model_path=prefilter_model_path,
            use_gpu=cfg.use_gpu,
            output_h5_path=prefilter_features_h5_path,
            output_pt_path=prefilter_features_pt_path,
            patch_h5_path=tessellate_h5_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            gpu_device_ids=cfg.gpu_device_ids,
        )
        
        # Filter features
        logger.info(f"Filtering features: {slide_path}")
        with open(cfg.classifier_pkl, "rb") as f:
            classifier = pickle.load(f)
        
        with h5py.File(prefilter_features_h5_path, "r") as features_h5:
            if prefilter_features_pt_path and os.path.exists(prefilter_features_pt_path):
                features = torch.load(prefilter_features_pt_path, weights_only=True)
            else:
                features = np.array(features_h5["features"])
                features = torch.from_numpy(features)
            
            filtered_features, filtered_coords = filter_features(
                features,
                features_h5["coords"][:],
                classifier,
                cfg.classifier_threshold,
            )
            
            logger.info(
                f"Filtering complete. {len(filtered_coords)} tiles passed (out of {len(coords)})."
            )
            
            if skip_second_extraction:
                # Save filtered features directly
                asset_dict = {"coords": filtered_coords}
                if cfg.save_features_to_h5:
                    asset_dict["features"] = filtered_features.numpy()
                save_hdf5(
                    output_h5_path,
                    asset_dict,
                    attr_h5_path=prefilter_features_h5_path,
                    mode="w",
                )
                save_torch_tensor(output_pt_path, filtered_features)
                return None  # No further processing needed
            else:
                # Save filtered coordinates for second extraction
                if cfg.keep_intermediate_files:
                    filtered_coords_h5_path = safe_path_join(base_path, f"{Path(slide_path).stem}.filtered_coords.h5")
                else:
                    filtered_coords_h5_path = os.path.join(temp_dir, f"{Path(slide_path).stem}.filtered_coords.h5")
                
                save_hdf5(
                    filtered_coords_h5_path,
                    {"coords": filtered_coords},
                    attr_h5_path=prefilter_features_h5_path,
                    mode="w",
                )
                
                final_coords_h5_path = filtered_coords_h5_path
        
        final_coords = filtered_coords
    
    # Final extraction step
    if two_step_mode and cfg.aggregation_method != "identity":
        # Extract patch features for batch aggregation later
        logger.info(f"Extracting patch features: {slide_path}")
        intermediate_h5_path = safe_path_join(base_path, f"{Path(slide_path).stem}.patch.h5")
        
        save_features(
            slide_path=slide_path,
            gpu_device_id=cfg.gpu_device_id,
            model_type=model_type,
            model_path=model_path,
            use_gpu=cfg.use_gpu,
            output_h5_path=intermediate_h5_path,
            output_pt_path=None,  # Don't save PT yet
            patch_h5_path=final_coords_h5_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            gpu_device_ids=cfg.gpu_device_ids,
            intermediate_h5_path=None,
            aggregation_method="identity",  # Don't aggregate yet
            slide_model_type=None,
            slide_model_path=None,
        )
        
        # Return paths for batch aggregation
        return {
            'intermediate_h5_path': intermediate_h5_path,
            'output_h5_path': output_h5_path,
            'output_pt_path': output_pt_path,
            'final_coords': final_coords,
            'slide_path': slide_path,
            'tessellate_h5_path': tessellate_h5_path,
        }
    else:
        # Single-step extraction or no aggregation
        logger.info(f"Extracting features: {slide_path}")
        save_features(
            slide_path=slide_path,
            gpu_device_id=cfg.gpu_device_id,
            model_type=model_type,
            model_path=model_path,
            use_gpu=cfg.use_gpu,
            output_h5_path=output_h5_path,
            output_pt_path=output_pt_path,
            patch_h5_path=final_coords_h5_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            gpu_device_ids=cfg.gpu_device_ids,
            intermediate_h5_path=getattr(cfg, 'intermediate_h5_path', None),
            aggregation_method=cfg.aggregation_method,
            slide_model_type=getattr(cfg, 'slide_model_type', None),
            slide_model_path=slide_model_path,
        )
        
        return {
            'final_coords': final_coords,
            'slide_path': slide_path,
            'tessellate_h5_path': tessellate_h5_path,
        }


def process_slide_tessellation_only(
    slide_path: str,
    slide_id: Optional[str],
    cfg,
    temp_dir: str,
    base_path: Union[str, Path],
    use_filtering: bool,
    prefilter_model_type,
    prefilter_model_path: Optional[str],
    skip_second_extraction: bool,
    output_mask_path: Optional[str] = None,
) -> Optional[dict]:
    """
    Process a single slide through tessellation and optional filtering (without feature extraction).
    
    This is used in batch mode to prepare slides for batch feature extraction.
    
    Args:
        slide_path: Path to the whole-slide image
        slide_id: Optional slide identifier
        cfg: Configuration object with seg_config, vis_config, etc.
        temp_dir: Temporary directory for intermediate files
        base_path: Base path for output files
        use_filtering: Whether to apply filtering
        prefilter_model_type: Model type for pre-filtering
        prefilter_model_path: Path to pre-filter model weights
        skip_second_extraction: Whether to skip second extraction (when models are same)
        output_mask_path: Optional path to save mask visualization
        
    Returns:
        Dict with paths and coordinates for batch feature extraction, None if failed
    """
    # Step 1: Tessellate
    logger.info(f"Tessellating slide: {slide_path}")
    if cfg.keep_intermediate_files:
        tessellate_h5_path = safe_path_join(base_path, f"{Path(slide_path).stem}.tessellate.h5")
    else:
        tessellate_h5_path = os.path.join(temp_dir, f"{Path(slide_path).stem}.tessellate.h5")
    
    if values := segment_tissue(
        slide_path=slide_path,
        slide_id=slide_id,
        output_h5_path=tessellate_h5_path,
        **OmegaConf.to_container(cfg.seg_config),
    ):
        polygon, grid, coords, _ = values
    else:
        logger.error(f"Tessellation failed for {slide_path}")
        return None
    
    logger.info(f"Tessellation complete. Found {len(coords)} tiles.")
    
    # Optional: Save mask visualization
    if output_mask_path:
        mask = draw_slide_mask(
            slide_path,
            polygon,
            **OmegaConf.to_container(cfg.vis_config),
        )
        mask.save(output_mask_path)
    
    # Coordinate source for final extraction
    final_coords_h5_path = tessellate_h5_path
    final_coords = coords
    prefilter_features_h5_path = None  # Initialize to avoid NameError when use_filtering is False
    
    if use_filtering:
        # Extract features for filtering
        logger.info(f"Extracting features for filtering: {slide_path}")
        if cfg.keep_intermediate_files:
            prefilter_features_h5_path = safe_path_join(base_path, f"{Path(slide_path).stem}.prefilter_features.h5")
            prefilter_features_pt_path = safe_path_join(base_path, f"{Path(slide_path).stem}.prefilter_features.pt")
        else:
            prefilter_features_h5_path = os.path.join(temp_dir, f"{Path(slide_path).stem}.prefilter_features.h5")
            prefilter_features_pt_path = os.path.join(temp_dir, f"{Path(slide_path).stem}.prefilter_features.pt")
        
        save_features(
            slide_path=slide_path,
            gpu_device_id=cfg.gpu_device_id,
            model_type=prefilter_model_type,
            model_path=prefilter_model_path,
            use_gpu=cfg.use_gpu,
            output_h5_path=prefilter_features_h5_path,
            output_pt_path=prefilter_features_pt_path,
            patch_h5_path=tessellate_h5_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            gpu_device_ids=cfg.gpu_device_ids,
        )
        
        # Filter features
        logger.info(f"Filtering features: {slide_path}")
        with open(cfg.classifier_pkl, "rb") as f:
            classifier = pickle.load(f)
        
        with h5py.File(prefilter_features_h5_path, "r") as features_h5:
            if prefilter_features_pt_path and os.path.exists(prefilter_features_pt_path):
                features = torch.load(prefilter_features_pt_path, weights_only=True)
            else:
                features = np.array(features_h5["features"])
                features = torch.from_numpy(features)
            
            filtered_features, filtered_coords = filter_features(
                features,
                features_h5["coords"][:],
                classifier,
                cfg.classifier_threshold,
            )
            
            logger.info(
                f"Filtering complete. {len(filtered_coords)} tiles passed (out of {len(coords)})."
            )
            
            if skip_second_extraction:
                # If not doing second extraction, we're done - need to save final features
                # But we'll handle this in the batch processing logic
                final_coords = filtered_coords
                final_coords_h5_path = prefilter_features_h5_path
            else:
                # Create filtered coords h5 for second extraction
                if cfg.keep_intermediate_files:
                    filtered_coords_h5_path = safe_path_join(base_path, f"{Path(slide_path).stem}.filtered_coords.h5")
                else:
                    filtered_coords_h5_path = os.path.join(temp_dir, f"{Path(slide_path).stem}.filtered_coords.h5")
                
                asset_dict = {"coords": filtered_coords}
                save_hdf5(
                    filtered_coords_h5_path,
                    asset_dict,
                    attr_h5_path=tessellate_h5_path,
                    mode="w",
                )
                
                final_coords_h5_path = filtered_coords_h5_path
                final_coords = filtered_coords
    
    # Return paths for batch feature extraction
    return {
        'slide_path': slide_path,
        'tessellate_h5_path': tessellate_h5_path,
        'final_coords_h5_path': final_coords_h5_path,
        'final_coords': final_coords,
        'skip_second_extraction': skip_second_extraction,
        'prefilter_features_h5_path': prefilter_features_h5_path if use_filtering else None,
    }


def create_visualizations(
    slide_path: str,
    final_coords,
    tessellate_h5_path: str,
    cfg,
    output_grid_mask_path: Optional[str] = None,
    output_png_dir: Optional[str] = None,
    output_thumbnail_path: Optional[str] = None,
):
    """Create optional visualizations (grid mask, PNG patches, thumbnail)."""
    # Create grid visualization
    if output_grid_mask_path:
        logger.info(f"Creating grid mask with {len(final_coords)} tiles")
        with h5py.File(tessellate_h5_path, "r") as h5:
            native_patch_size = h5["coords"].attrs["patch_size"]
        for coord in final_coords:
            x, y = coord
            poly = Polygon([
                [x, y],
                [x, y + native_patch_size],
                [x + native_patch_size, y + native_patch_size],
                [x + native_patch_size, y],
            ])
            grid_polygons.append(poly)
        
        grid_mask = draw_slide_mask(
            slide_path,
            grid_polygons,
            **OmegaConf.to_container(cfg.vis_config),
        )
        grid_mask.save(output_grid_mask_path)
    
    # Save PNG patches
    if output_png_dir:
        logger.info(f"Saving patches to {output_png_dir}")
        save_patches_png(
            slide_path,
            final_coords,
            save_dir=output_png_dir,
            num_workers=cfg.num_workers,
            patch_size=cfg.seg_config.patch_size,
            filter_black_white=cfg.png_config.filter_black_white,
            white_threshold=cfg.png_config.white_threshold,
            black_threshold=cfg.png_config.black_threshold,
        )
    
    # Save thumbnail
    if output_thumbnail_path:
        with tiffslide.TiffSlide(slide_path) as wsi:
            logger.info(f"Saving thumbnail to {output_thumbnail_path}")
            thumbnail = wsi.get_thumbnail(cfg.thumbnail_size)
            with open(output_thumbnail_path, "wb") as f:
                thumbnail.save(f)
