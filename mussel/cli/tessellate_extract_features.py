import os
import ssl
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import hydra
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
from mussel.models import ModelType
from mussel.utils import save_features
from mussel.utils.segment import draw_slide_mask, save_patches_png, segment_tissue

ssl._create_default_https_context = ssl._create_unverified_context


defaults = ["_self_", {"seg_config": "default"}]


@dataclass
class TessellateExtractFeaturesConfig:
    """
    slide_path (str): Path to the whole-slide image.
    slide_id (Optional[str]): Optional slide ID. If None, the slide filename without extension is used.
    output_h5_path (str): Path to save the final HDF5 file with tile coordinates and features.
    output_pt_path (str): Path to save the final features in PyTorch format.
    model_type (ModelType): Type of model to use for feature extraction.
    model_path (Optional[str]): Path to the model weights file, if applicable.
    output_png_dir (Optional[str]): Directory to save patches as PNG files.
    output_mask_path (Optional[str]): Path to save the mask image.
    output_grid_mask_path (Optional[str]): Path to save the grid mask image.
    output_thumbnail_path (Optional[str]): Path to save the thumbnail image.
    thumbnail_size (tuple): Size of the thumbnail image.
    seg_config (SegConfig): Configuration for segmentation parameters.
    vis_config (VisConfig): Configuration for visualization parameters.
    png_config (PngConfig): Configuration for PNG saving parameters.
    num_workers (int): Number of workers for saving patches and feature extraction.
    batch_size (int): Batch size for feature extraction.
    use_gpu (bool): Whether to use GPU for feature extraction.
    gpu_device_id (Optional[int]): Specific GPU device ID to use, if applicable.
    gpu_device_ids (Optional[List[int]]): List of GPU device IDs to use, if applicable.
    keep_intermediate_files (bool): Whether to keep intermediate tessellation file.
    intermediate_h5_path (Optional[str]): Path for intermediate patch features (two-step mode).
    aggregation_method (str): Aggregation method: identity (single-step), mean/max/model (two-step).
    slide_model_type (Optional[ModelType]): Type of slide encoder model (when aggregation_method="model").
    slide_model_path (Optional[str]): Path to slide encoder model weights.
    """

    defaults: List[Any] = field(default_factory=lambda: defaults)
    slide_path: str = MISSING
    slide_id: Optional[str] = None
    output_h5_path: str = MISSING
    output_pt_path: str = MISSING
    model_type: ModelType = ModelType.CTRANSPATH
    model_path: Optional[str] = None
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
    seg_config: SegConfig = MISSING
    vis_config: VisConfig = field(default_factory=VisConfig)
    png_config: PngConfig = field(default_factory=PngConfig)
    intermediate_h5_path: Optional[str] = None
    aggregation_method: str = "identity"
    slide_model_type: Optional[ModelType] = None
    slide_model_path: Optional[str] = None


desc_doc = """== ${hydra.help.app_name} ==

tessellate-extract-features performs an integrated workflow that tessellates a whole-slide image 
and extracts features from the tiles using a foundation model. This combines the functionality 
of tessellate and extract_features into a single command.
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
    """Tessellate and extract features from a whole-slide image in one workflow."""
    # Create temporary directory for intermediate files if not keeping them
    temp_dir = None
    base_path = Path(cfg.output_h5_path).parent
    
    if not cfg.keep_intermediate_files:
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Using temporary directory for intermediate files: {temp_dir}")
    
    # Step 1: Tessellate
    logger.info("Step 1/2: Tessellating whole-slide image...")
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

    # Optional: Save grid visualization (all tiles)
    if cfg.output_grid_mask_path:
        logger.info(f"Creating grid mask with {len(coords)} tiles")
        grid_mask = draw_slide_mask(
            cfg.slide_path,
            grid,
            **OmegaConf.to_container(cfg.vis_config),
        )
        grid_mask.save(cfg.output_grid_mask_path)

    # Optional: Save PNG patches
    if cfg.output_png_dir:
        logger.info(f"Saving patches to {cfg.output_png_dir}")
        save_patches_png(
            cfg.slide_path,
            coords,
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

    # Step 2: Extract features using foundation model
    logger.info(f"Step 2/2: Extracting features using {cfg.model_type.name}...")

    save_features(
        slide_path=cfg.slide_path,
        gpu_device_id=cfg.gpu_device_id,
        model_type=cfg.model_type,
        model_path=cfg.model_path,
        use_gpu=cfg.use_gpu,
        output_h5_path=cfg.output_h5_path,
        output_pt_path=cfg.output_pt_path,
        patch_h5_path=tessellate_h5_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        gpu_device_ids=cfg.gpu_device_ids,
        intermediate_h5_path=cfg.intermediate_h5_path,
        aggregation_method=cfg.aggregation_method,
        slide_model_type=cfg.slide_model_type,
        slide_model_path=cfg.slide_model_path,
    )

    logger.info(f"Feature extraction complete.")

    # Clean up temporary directory if not keeping intermediate files
    if temp_dir:
        import shutil
        shutil.rmtree(temp_dir)
        logger.info("Cleaned up temporary files.")


if __name__ == "__main__":
    main()
