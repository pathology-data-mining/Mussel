import os
import shutil
import ssl
import tempfile
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import hydra
import torch
import tiffslide
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
from mussel.cli.tessellate_extract_features_common import _build_grid_polygons
from mussel.models import ModelType, get_default_patch_size
from mussel.utils import save_features, filter_features, save_hdf5, load_classifier, load_features_from_h5
from mussel.utils.segment import draw_slide_mask, save_patches_png, segment_tissue

logger = logging.getLogger(__name__)

defaults = ["_self_", {"seg_config": "default"}]


@dataclass
class FilterTessellateConfig:
    """
    slide_path (str): Path to the whole-slide image.
    slide_id (Optional[str]): Optional slide ID. If None, the slide filename without extension is used.
    output_h5_path (str): Path to save the final filtered HDF5 file with tile coordinates.
    output_pt_path (str): Path to save the final filtered features in PyTorch format.
    classifier_pkl (str): Path to the classifier model in pickle format.
    classifier_threshold (float): Threshold for the classifier to filter features.
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
    save_features_to_h5 (bool): Whether to save the filtered features to HDF5.
    keep_intermediate_files (bool): Whether to keep intermediate files (tessellation and features).
    ssl_verify (bool): Whether to verify SSL certificates when downloading models or accessing remote resources (default: True).
    """

    defaults: List[Any] = field(default_factory=lambda: defaults)
    slide_path: str = MISSING
    slide_id: Optional[str] = None
    output_h5_path: str = MISSING
    output_pt_path: str = MISSING
    classifier_pkl: str = MISSING
    classifier_threshold: float = 0.75
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
    save_features_to_h5: bool = False
    keep_intermediate_files: bool = False
    ssl_verify: bool = True  # Whether to verify SSL certificates for remote operations
    seg_config: SegConfig = MISSING
    vis_config: VisConfig = field(default_factory=VisConfig)
    png_config: PngConfig = field(default_factory=PngConfig)

    def __post_init__(self):
        """Set default patch size based on model type if not explicitly set."""
        # Only set patch size if seg_config.patch_size is at the default value
        # This allows users to override if they explicitly set a different value
        if self.seg_config.patch_size == SegConfig.DEFAULT_PATCH_SIZE:
            # Get recommended patch size for the model
            try:
                recommended_patch_size = get_default_patch_size(self.model_type)
                if recommended_patch_size != SegConfig.DEFAULT_PATCH_SIZE:
                    logger.info(
                        f"Setting seg_config.patch_size={recommended_patch_size} based on "
                        f"model_type={self.model_type.name} (recommended default for this model)"
                    )
                    self.seg_config.patch_size = recommended_patch_size
            except ValueError:
                # Model not in mapping, keep default
                pass


desc_doc = """== ${hydra.help.app_name} ==

filter-tessellate performs an integrated workflow that tessellates a whole-slide image, 
extracts features from the tiles using a foundation model, and filters them using a supplied 
classifier model. This combines the functionality of tessellate, extract_features, and 
filter_features into a single command.
"""

parameter_doc = f"""
== Available Parameters ==
{FilterTessellateConfig.__doc__}
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
cs.store(name="filter_tessellate_config", node=FilterTessellateConfig)


@hydra.main(version_base=None, config_path=".", config_name="filter_tessellate_config")
def main(
    cfg: FilterTessellateConfig,
):
    """Tessellate, extract CTRANSPATH features, and filter tiles in one workflow."""
    # Create temporary directory for intermediate files if not keeping them
    temp_dir = None
    base_path = Path(cfg.output_h5_path).parent

    if not cfg.keep_intermediate_files:
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Using temporary directory for intermediate files: {temp_dir}")

    try:
        _main(cfg, temp_dir, base_path)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files.")


def _main(cfg: FilterTessellateConfig, temp_dir, base_path):
    # Step 1: Tessellate
    logger.info("Step 1/3: Tessellating whole-slide image...")
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

    # Step 2: Extract features using foundation model
    logger.info(f"Step 2/3: Extracting features using {cfg.model_type.name}...")
    if cfg.keep_intermediate_files:
        features_h5_path = str(base_path / f"{Path(cfg.slide_path).stem}.features.h5")
        features_pt_path = str(base_path / f"{Path(cfg.slide_path).stem}.features.pt")
    else:
        features_h5_path = os.path.join(temp_dir, "features.h5")
        features_pt_path = os.path.join(temp_dir, "features.pt")

    save_features(
        slide_path=cfg.slide_path,
        gpu_device_id=cfg.gpu_device_id,
        model_type=cfg.model_type,
        model_path=cfg.model_path,
        use_gpu=cfg.use_gpu,
        output_h5_path=features_h5_path,
        output_pt_path=features_pt_path,
        patch_h5_path=tessellate_h5_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        gpu_device_ids=cfg.gpu_device_ids,
    )

    logger.info(f"Feature extraction complete.")

    # Step 3: Filter features
    logger.info("Step 3/3: Filtering features using classifier...")
    logger.info(f"Loading classifier from {cfg.classifier_pkl}")
    classifier = load_classifier(cfg.classifier_pkl)

    features, coords_all = load_features_from_h5(features_h5_path, features_pt_path)
    logger.info(
        f"Loaded {features.shape[0]} features of dimension {features.shape[1]}"
    )
    features, coords = filter_features(
        features,
        coords_all,
        classifier,
        cfg.classifier_threshold,
    )

    logger.info(f"Saving filtered results to {cfg.output_h5_path}")
    asset_dict = {"coords": coords}
    if cfg.save_features_to_h5:
        asset_dict["features"] = features.numpy()
    save_hdf5(
        cfg.output_h5_path,
        asset_dict,
        attr_h5_path=features_h5_path,
        mode="w",
        ssl_verify=cfg.ssl_verify,
    )

    torch.save(features, cfg.output_pt_path)

    logger.info(
        f"Filter-tessellate complete. {len(coords)} tiles passed the threshold."
    )

    # Create filtered grid visualization (post-filtering)
    if cfg.output_grid_mask_path:
        logger.info(f"Creating filtered grid mask with {len(coords)} tiles")
        filtered_grid = _build_grid_polygons(coords, tessellate_h5_path)
        grid_mask = draw_slide_mask(
            cfg.slide_path,
            filtered_grid,
            **OmegaConf.to_container(cfg.vis_config),
        )
        grid_mask.save(cfg.output_grid_mask_path)

    # Save PNG patches and thumbnail using filtered coordinates (post-filtering)
    if cfg.output_png_dir:
        logger.info(f"Saving filtered patches to {cfg.output_png_dir}")
        save_patches_png(
            cfg.slide_path,
            coords,
            save_dir=cfg.output_png_dir,
            num_workers=cfg.num_workers,
            mpp=cfg.seg_config.mpp,
            patch_size=cfg.seg_config.patch_size,
            filter_black_white=cfg.png_config.filter_black_white,
            white_threshold=cfg.png_config.white_threshold,
            black_threshold=cfg.png_config.black_threshold,
            slide_mpp_override=cfg.seg_config.slide_mpp_override,
        )

    if cfg.output_thumbnail_path:
        with tiffslide.TiffSlide(cfg.slide_path) as wsi:
            logger.info(f"Saving thumbnail to {cfg.output_thumbnail_path}")
            thumbnail = wsi.get_thumbnail(cfg.thumbnail_size)
            with open(cfg.output_thumbnail_path, "wb") as f:
                thumbnail.save(f)


if __name__ == "__main__":
    main()
