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

from mussel.models import ModelType
from mussel.utils import save_features, filter_features, save_hdf5
from mussel.utils.segment import draw_slide_mask, save_patches_png, segment_tissue

ssl._create_default_https_context = ssl._create_unverified_context


@dataclass
class SegConfig:
    """
    patch_size (int): Patch size at specified mpp (microns per pixel).
    step_size (int): Optional step size. Defaults to the patch size.
    mpp (float): Desired microns per pixel
    seg_level (int): Tessellation pyramid level. If negative, use best level for factor=64 downsample.
    segment_threshold (int): Pixel threshold value . If pixel value smaller than or equal to threshold, it is set to 0, otherwise it is set to the maximum value (segment_max_value).
    segment_max_value (int): Maximum pixel value.
    median_blur_ksize (int): Aperture linear size. it must be odd and greater than 1. image is blurred with median filter.
    morphology_ex_kernel (int): Kernel for morphological closing transformation.
    ref_patch_size (int): Reference patch size to use for tissue area and hole area thresholding.
    use_otsu (bool): If True, apply otsu thresholding.
    tissue_area_threshold (int): Tissue area threshold. Foreground contour area needs to exceed this threshold (scaled by reference patch size) to be included as foreground.
    hole_area_threshold (int): Hole area threshold. Hole contour area needs to exceed this threshold (scaled by reference patch size) to be included as a hole.
    max_num_holes (int): Maximum number of holes.
    keep_ids (List[int]): List of contour IDs to keep.
    exclude_ids (List[int]): List of contour IDs to exclude.
    """

    patch_size: int = 256
    step_size: Optional[int] = None  # if None, defaults to patch_size
    mpp: float = 0.5
    seg_level: int = -1
    segment_threshold: int = 20
    segment_max_value: int = 255
    median_blur_ksize: int = 7
    morphology_ex_kernel: int = 0
    ref_patch_size: int = 512
    use_otsu: bool = False
    tissue_area_threshold: int = 100
    hole_area_threshold: int = 16
    max_num_holes: int = 8
    keep_ids: List[int] = field(default_factory=list)
    exclude_ids: List[int] = field(default_factory=list)


@dataclass
class BiopsySegConfig(SegConfig):
    segment_threshold: int = 15
    median_blur_ksize: int = 11
    morphology_ex_kernel: int = 2
    tissue_area_threshold: int = 1
    hole_area_threshold: int = 1
    max_num_holes: int = 2


@dataclass
class ResectionSegConfig(SegConfig):
    segment_threshold: int = 15
    median_blur_ksize: int = 11
    morphology_ex_kernel: int = 4
    tissue_area_threshold: int = 100
    hole_area_threshold: int = 16
    max_num_holes: int = 8


@dataclass
class TcgaSegConfig(SegConfig):
    segment_threshold: int = 8
    median_blur_ksize: int = 7
    morphology_ex_kernel: int = 4
    tissue_area_threshold: int = 16
    hole_area_threshold: int = 4
    max_num_holes: int = 8


@dataclass
class VisConfig:
    """
    vis_level (int): pyramid level to visualize. If negative, use best level for factor=64 downsample.
    outline (str): color of the outline of the tissue mask.
    fill (tuple): RGBA color of the filled tissue mask.
    custom_downsample (Optional[int]): custom downsample factor for visualization. If None, use the default downsample factor.
    """

    vis_level: int = -1
    outline = "black"
    fill = (255, 0, 0, 80)
    custom_downsample: Optional[int] = None


@dataclass
class PngConfig:
    """
    filter_black_white (bool): If True, filter out black and white patches.
    white_threshold (int): Threshold for white patches.
    black_threshold (int): Threshold for black patches.
    """

    filter_black_white: bool = True
    white_threshold: int = 15
    black_threshold: int = 50


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
    model_path (str): Path to the CTRANSPATH model weights.
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
    """

    defaults: List[Any] = field(default_factory=lambda: defaults)
    slide_path: str = MISSING
    slide_id: Optional[str] = None
    output_h5_path: str = MISSING
    output_pt_path: str = MISSING
    classifier_pkl: str = MISSING
    classifier_threshold: float = 0.75
    model_path: str = MISSING
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
    seg_config: SegConfig = MISSING
    vis_config: VisConfig = field(default_factory=VisConfig)
    png_config: PngConfig = field(default_factory=PngConfig)


desc_doc = """== ${hydra.help.app_name} ==

filter-tessellate performs an integrated workflow that tessellates a whole-slide image, 
extracts CTRANSPATH features from the tiles, and filters them using a supplied classifier model.
This combines the functionality of tessellate, extract_features (with CTRANSPATH), and 
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
        if temp_dir:
            import shutil
            shutil.rmtree(temp_dir)
        return

    logger.info(f"Tessellation complete. Found {len(coords)} tiles.")

    # Optional: Save mask and grid visualizations
    if cfg.output_mask_path:
        mask = draw_slide_mask(
            cfg.slide_path,
            polygon,
            **OmegaConf.to_container(cfg.vis_config),
        )
        mask.save(cfg.output_mask_path)

    if cfg.output_grid_mask_path:
        grid_mask = draw_slide_mask(
            cfg.slide_path,
            grid,
            **OmegaConf.to_container(cfg.vis_config),
        )
        grid_mask.save(cfg.output_grid_mask_path)

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

    if cfg.output_thumbnail_path:
        with tiffslide.TiffSlide(cfg.slide_path) as wsi:
            logger.info(f"Saving thumbnail to {cfg.output_thumbnail_path}")
            thumbnail = wsi.get_thumbnail(cfg.thumbnail_size)
            with open(cfg.output_thumbnail_path, "wb") as f:
                thumbnail.save(f)

    # Step 2: Extract CTRANSPATH features
    logger.info("Step 2/3: Extracting CTRANSPATH features...")
    if cfg.keep_intermediate_files:
        features_h5_path = str(base_path / f"{Path(cfg.slide_path).stem}.features.h5")
        features_pt_path = str(base_path / f"{Path(cfg.slide_path).stem}.features.pt")
    else:
        features_h5_path = os.path.join(temp_dir, "features.h5")
        features_pt_path = os.path.join(temp_dir, "features.pt")

    save_features(
        slide_path=cfg.slide_path,
        gpu_device_id=cfg.gpu_device_id,
        model_type=ModelType.CTRANSPATH,
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
    with open(cfg.classifier_pkl, "rb") as f:
        classifier = pickle.load(f)

    with h5py.File(features_h5_path, "r") as features_h5:
        if features_pt_path and os.path.exists(features_pt_path):
            features = torch.load(features_pt_path, weights_only=True)
        else:
            features = np.array(features_h5["features"])
            features = torch.Tensor(features)
        logger.info(
            f"Loaded {features.shape[0]} features of dimension {features.shape[1]}"
        )
        features, coords = filter_features(
            features,
            features_h5["coords"][:],
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
        )

        torch.save(features, cfg.output_pt_path)

    logger.info(
        f"Filter-tessellate complete. {len(coords)} tiles passed the threshold."
    )

    # Clean up temporary directory if not keeping intermediate files
    if temp_dir:
        import shutil
        shutil.rmtree(temp_dir)
        logger.info("Cleaned up temporary files.")


if __name__ == "__main__":
    main()
