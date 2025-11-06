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
from mussel.utils import save_features, filter_features, save_hdf5, aggregate_slide_features_batch
from mussel.utils.segment import draw_slide_mask, save_patches_png, segment_tissue

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


def process_single_slide(
    slide_path,
    slide_id,
    output_h5_path,
    output_pt_path,
    cfg,
    temp_dir,
    use_filtering,
    postfilter_model_type,
    postfilter_model_path,
    skip_second_extraction,
):
    """Process a single slide through tessellation and feature extraction.
    
    Returns intermediate patch features path if needed for batch aggregation, None otherwise.
    """
    base_path = Path(output_h5_path).parent
    
    # Step 1: Tessellate
    logger.info(f"Tessellating slide: {slide_path}")
    if cfg.keep_intermediate_files:
        tessellate_h5_path = str(base_path / f"{Path(slide_path).stem}.tessellate.h5")
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
    if cfg.output_mask_suffix:
        output_mask_path = str(base_path / f"{Path(slide_path).stem}{cfg.output_mask_suffix}")
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
            prefilter_features_h5_path = str(base_path / f"{Path(slide_path).stem}.prefilter_features.h5")
            prefilter_features_pt_path = str(base_path / f"{Path(slide_path).stem}.prefilter_features.pt")
        else:
            prefilter_features_h5_path = os.path.join(temp_dir, f"{Path(slide_path).stem}.prefilter_features.h5")
            prefilter_features_pt_path = os.path.join(temp_dir, f"{Path(slide_path).stem}.prefilter_features.pt")
        
        save_features(
            slide_path=slide_path,
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
        
        # Filter features
        logger.info(f"Filtering features: {slide_path}")
        with open(cfg.classifier_pkl, "rb") as f:
            classifier = pickle.load(f)
        
        with h5py.File(prefilter_features_h5_path, "r") as features_h5:
            if prefilter_features_pt_path and os.path.exists(prefilter_features_pt_path):
                features = torch.load(prefilter_features_pt_path, weights_only=True)
            else:
                features = np.array(features_h5["features"])
                features = torch.Tensor(features)
            
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
                torch.save(filtered_features, output_pt_path)
                return None  # No batch aggregation needed
            else:
                # Save filtered coordinates for second extraction
                if cfg.keep_intermediate_files:
                    filtered_coords_h5_path = str(base_path / f"{Path(slide_path).stem}.filtered_coords.h5")
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
    # If using two-step mode with model aggregation, we'll batch this later
    use_two_step = cfg.aggregation_method != "identity"
    
    if not skip_second_extraction:
        if use_two_step:
            # Extract patch features, will aggregate in batch later
            logger.info(f"Extracting patch features: {slide_path}")
            intermediate_h5_path = str(base_path / f"{Path(slide_path).stem}.patch.h5")
            
            save_features(
                slide_path=slide_path,
                gpu_device_id=cfg.gpu_device_id,
                model_type=postfilter_model_type,
                model_path=postfilter_model_path,
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
            
            # Return path for batch aggregation
            return {
                'intermediate_h5_path': intermediate_h5_path,
                'output_h5_path': output_h5_path,
                'output_pt_path': output_pt_path,
                'final_coords': final_coords,
                'slide_path': slide_path,
                'tessellate_h5_path': tessellate_h5_path,
            }
        else:
            # Single-step extraction
            logger.info(f"Extracting features (single-step): {slide_path}")
            save_features(
                slide_path=slide_path,
                gpu_device_id=cfg.gpu_device_id,
                model_type=postfilter_model_type,
                model_path=postfilter_model_path,
                use_gpu=cfg.use_gpu,
                output_h5_path=output_h5_path,
                output_pt_path=output_pt_path,
                patch_h5_path=final_coords_h5_path,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                gpu_device_ids=cfg.gpu_device_ids,
            )
    
    # Create grid visualization
    if cfg.output_grid_mask_suffix:
        output_grid_mask_path = str(base_path / f"{Path(slide_path).stem}{cfg.output_grid_mask_suffix}")
        logger.info(f"Creating grid mask with {len(final_coords)} tiles")
        with h5py.File(tessellate_h5_path, "r") as h5:
            native_patch_size = h5.attrs["patch_size"]
        
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
        
        grid_mask = draw_slide_mask(
            slide_path,
            grid_polygons,
            **OmegaConf.to_container(cfg.vis_config),
        )
        grid_mask.save(output_grid_mask_path)
    
    # Save PNG patches
    if cfg.output_png_dir_suffix:
        output_png_dir = str(base_path / f"{Path(slide_path).stem}{cfg.output_png_dir_suffix}")
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
    if cfg.output_thumbnail_suffix:
        output_thumbnail_path = str(base_path / f"{Path(slide_path).stem}{cfg.output_thumbnail_suffix}")
        with tiffslide.TiffSlide(slide_path) as wsi:
            logger.info(f"Saving thumbnail to {output_thumbnail_path}")
            thumbnail = wsi.get_thumbnail(cfg.thumbnail_size)
            with open(output_thumbnail_path, "wb") as f:
                thumbnail.save(f)
    
    return None  # No batch aggregation needed


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
        output_h5_path = str(output_dir / f"{slide_id}.{cfg.output_h5_suffix}")
        output_pt_path = str(output_dir / f"{slide_id}.{cfg.output_pt_suffix}")
        
        result = process_single_slide(
            slide_path=slide_path,
            slide_id=slide_id,
            output_h5_path=output_h5_path,
            output_pt_path=output_pt_path,
            cfg=cfg,
            temp_dir=temp_dir,
            use_filtering=use_filtering,
            postfilter_model_type=postfilter_model_type,
            postfilter_model_path=postfilter_model_path,
            skip_second_extraction=skip_second_extraction,
        )
        
        if result is not None:
            slides_for_batch_aggregation.append(result)
    
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
