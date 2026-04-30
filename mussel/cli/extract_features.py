import logging
import os
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import hydra
import numpy as np
import torch
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch.utils.data import DataLoader
from tqdm import tqdm

from mussel.datasets import WholeSlideImageTileCoordDataset
from mussel.models import ModelType, get_model_factory
from mussel.utils import (aggregate_slide_features_batch,
                          ensure_directory_exists,
                          extract_patch_features_batch,
                          get_slide_ids_from_paths, resolve_remote_paths,
                          save_features)
from mussel.utils.file import save_hdf5, save_torch_tensor
from mussel.utils.ml import collate_features

logger = logging.getLogger(__name__)


@dataclass
class ExtractFeaturesConfig:
    """
    Configuration for extract-features command.

    Supports three input modes:

    1. Single Slide Mode:
       - Provide: patch_h5_path, slide_path, output_h5_path
       - Extracts features from one slide using patch coordinates from HDF5 file
       - Output: Single H5 and PT file with features for one slide

    2. Batch Slides Mode:
       - Provide: patch_h5_paths, slide_paths, output_dir
       - Extracts features from multiple slides (batch processing)
       - Output: Multiple H5/PT files (one per slide) in output_dir

    3. Patch Directory Mode:
       - Provide: patch_path (directory), output_h5_path, output_pt_path
       - Extracts features from pre-extracted patch images in a directory
       - Output: Single H5 and PT file containing features for ALL patches in the directory
       - Note: Outputs are aggregated - one file per patch directory, not per patch image

    Single Mode Parameters:
        patch_h5_path (str): Path to the HDF5 file containing patches.
        slide_path (str): Path to the whole slide image.
        output_h5_path (str): Path to save the computed features in HDF5 format.
        output_pt_path (Optional[str]): Path to save the computed features in PyTorch format.

    Batch Mode Parameters:
        patch_h5_paths (List[str]): Paths to HDF5 files containing patches for multiple slides.
        slide_paths (List[str]): Paths to whole slide images.
        slide_ids (Optional[List[str]]): Optional slide IDs. If None, uses slide filenames without extension.
        output_dir (str): Directory to save output files.
        output_h5_suffix (str): Suffix for HDF5 output files (default: "features.h5").
        output_pt_suffix (str): Suffix for PyTorch output files (default: "features.pt").
        slide_batch_size (int): Number of slides to process in a single batch during slide-level aggregation (default: 8).

    Model Parameters:
        model_type (ModelType): Type of model to use for patch-level feature extraction.
        model_path (Optional[str]): Path to the patch encoder model weights, if applicable.
        model_dir (Optional[str]): Directory containing pre-downloaded models (for convenience).
        patch_path (Optional[str]): Directory containing pre-tiled patch images (Patch Directory Mode).
        intermediate_h5_path (Optional[str]): Path for intermediate patch features (two-step mode, single mode only).
        aggregation_method (str): Aggregation method: identity (single-step), mean/max/model (two-step).
        slide_model_type (Optional[ModelType]): Type of slide encoder model (when aggregation_method="model").
        slide_model_path (Optional[str]): Path to slide encoder model weights.
        ssl_verify (bool): Whether to verify SSL certificates when downloading models or accessing remote resources (default: True).

    Processing Parameters:
        batch_size (int): Batch size for processing patches or tiles.
        use_gpu (bool): Whether to use GPU for computation.
        gpu_device_id (Optional[int]): Specific GPU device ID to use, if applicable.
        gpu_device_ids (Optional[List[int]]): List of GPU device IDs to use, if applicable.
        num_workers (int): Number of worker threads for data loading.
    """

    # Single mode parameters
    patch_h5_path: Optional[str] = None
    slide_path: Optional[str] = None
    output_h5_path: Optional[str] = None
    output_pt_path: Optional[str] = None
    # Batch mode parameters
    patch_h5_paths: Optional[List[str]] = None
    slide_paths: Optional[List[str]] = None
    slide_ids: Optional[List[str]] = None
    output_dir: Optional[str] = None
    output_h5_suffix: str = "features.h5"
    output_pt_suffix: str = "features.pt"
    slide_batch_size: int = 8
    # Model parameters
    model_type: ModelType = ModelType.CLIP
    model_path: Optional[str] = None
    model_dir: Optional[str] = None  # Directory containing pre-downloaded models
    model_save_path: Optional[str] = None
    patch_path: Optional[str] = None
    intermediate_h5_path: Optional[str] = None
    aggregation_method: str = "identity"
    slide_model_type: Optional[ModelType] = None
    slide_model_path: Optional[str] = None
    ssl_verify: bool = True  # Whether to verify SSL certificates for remote operations
    # Processing parameters
    batch_size: int = 64
    use_gpu: bool = True
    gpu_device_id: Optional[int] = None
    gpu_device_ids: Optional[List[int]] = None
    num_workers: int = 16
    is_test_run: bool = False
    embedding_precision: str = "float32"
    """Numeric precision for saved patch embeddings. "float32" (default) keeps full precision;
    "float16" halves storage size. Note: "float16" with aggregation_method="model" feeds
    reduced-precision features to the slide encoder, which may affect quality."""


desc_doc = """== ${hydra.help.app_name} ==

Extract features (embeddings) from whole slide images (WSI) or patch directories using a
pathology foundation model. The embeddings are written to PyTorch tensor file (.pt)
and HDF5 (.h5) files.

Four Input Modes:

1. Single Slide Mode:
   Process one slide from patch coordinates in HDF5 file
   Args: patch_h5_path=<path> slide_path=<path> output_h5_path=<path> [output_pt_path=<path>]
   Output: Single H5 and PT file with features for one slide

2. Batch Slides Mode:
   Process multiple slides efficiently (batch processing)
   Args: patch_h5_paths=[<path1>,<path2>,...] slide_paths=[<path1>,<path2>,...] output_dir=<dir>
   Output: Multiple H5/PT files (one per slide) in output_dir
   Note: Models loaded once for all slides - significant performance benefit

3. Patch Directory Mode:
   Process pre-extracted patch images from a directory
   Args: patch_path=<dir> output_h5_path=<path> output_pt_path=<path> slide_path=None patch_h5_path=None
   Output: Single H5 and PT file containing features for ALL patches in directory
   Note: Supports S3 paths (s3://bucket/prefix/)

4. Multi-Slide (Sample) Mode:
   Process multiple slides per sample with optional subsampling
   Args: sample_batch_csv_path=<csv> output_dir=<dir> [sample_id=<id>] [max_tiles_per_sample=<n>]
   CSV format: sample_id,slide_path,patch_h5_path
   Output: One H5/PT per sample with concatenated features and slide_idx tracking
   Note: Supports {sample_id} placeholder in output paths

Aggregation Methods (automatically selected based on aggregation_method):
   - identity: Keep all patch features - single-step mode (default, no aggregation)
   - mean/max: Simple pooling aggregation - two-step mode
   - model: Use a slide encoder model - two-step mode (e.g., GIGAPATH_SLIDE for Prov-GigaPath)
"""

parameter_doc = f"""== Available Parameters ==
{ExtractFeaturesConfig.__doc__}
"""

cs = ConfigStore.instance()
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=desc_doc, footer=parameter_doc)),
    provider="hydra",
)
cs.store(name="extract_features_config", node=ExtractFeaturesConfig)


@hydra.main(version_base=None, config_path=".", config_name="extract_features_config")
def main(cfg: ExtractFeaturesConfig):
    """Extract features from whole slide image(s) using a foundation model."""
    # Detect mode based on configuration
    batch_mode = cfg.slide_paths is not None

    if batch_mode:
        logger.info("Running in batch mode (multiple slides)")
        _main_batch(cfg)
    else:
        logger.info("Running in single-slide mode")
        _main_single(cfg)


@resolve_remote_paths()
def _main_single(cfg: ExtractFeaturesConfig):
    """Process a single slide."""
    if (
        cfg.patch_h5_path is None
        or cfg.slide_path is None
        or cfg.output_h5_path is None
    ):
        raise ValueError(
            "Single-slide mode requires patch_h5_path, slide_path, and output_h5_path to be specified"
        )

    save_features(
        slide_path=cfg.slide_path,
        gpu_device_id=cfg.gpu_device_id,
        model_type=cfg.model_type,
        model_path=cfg.model_path,
        use_gpu=cfg.use_gpu,
        output_h5_path=cfg.output_h5_path,
        output_pt_path=cfg.output_pt_path,
        patch_h5_path=cfg.patch_h5_path,
        patch_path=cfg.patch_path,
        model_save_path=cfg.model_save_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        gpu_device_ids=cfg.gpu_device_ids,
        is_test_run=cfg.is_test_run,
        intermediate_h5_path=cfg.intermediate_h5_path,
        aggregation_method=cfg.aggregation_method,
        slide_model_type=cfg.slide_model_type,
        slide_model_path=cfg.slide_model_path,
        embedding_precision=cfg.embedding_precision,
    )


@resolve_remote_paths()
def _main_batch(cfg: ExtractFeaturesConfig):
    """Process multiple slides in batch mode."""
    if cfg.patch_h5_paths is None or cfg.slide_paths is None or cfg.output_dir is None:
        raise ValueError(
            "Batch mode requires patch_h5_paths, slide_paths, and output_dir to be specified"
        )

    # Create output directory
    output_dir = Path(cfg.output_dir)
    ensure_directory_exists(output_dir)

    # Generate slide IDs if not provided
    slide_ids = get_slide_ids_from_paths(cfg.slide_paths, cfg.slide_ids)

    # Validate input lengths
    if not (len(cfg.patch_h5_paths) == len(cfg.slide_paths)):
        raise ValueError(
            f"patch_h5_paths and slide_paths must have the same length: "
            f"patch_h5_paths={len(cfg.patch_h5_paths)}, slide_paths={len(cfg.slide_paths)}"
        )

    if slide_ids and len(slide_ids) != len(cfg.slide_paths):
        raise ValueError(
            f"slide_ids must have the same length as slide_paths: "
            f"slide_ids={len(slide_ids)}, slide_paths={len(cfg.slide_paths)}"
        )

    # Prepare output paths
    output_h5_paths = [
        str(output_dir / f"{slide_id}.{cfg.output_h5_suffix}") for slide_id in slide_ids
    ]
    output_pt_paths = [
        str(output_dir / f"{slide_id}.{cfg.output_pt_suffix}") for slide_id in slide_ids
    ]

    # Check if we need two-step processing
    use_two_step = cfg.aggregation_method != "identity"

    logger.info(f"Batch extracting features for {len(cfg.slide_paths)} slides")

    if use_two_step:
        # Extract to intermediate patch feature files for later aggregation
        intermediate_h5_paths = [
            str(output_dir / f"{slide_id}.patch.h5") for slide_id in slide_ids
        ]

        extract_patch_features_batch(
            patch_h5_paths=cfg.patch_h5_paths,
            slide_paths=cfg.slide_paths,
            output_h5_paths=intermediate_h5_paths,
            model_type=cfg.model_type,
            model_path=cfg.model_path,
            model_dir=cfg.model_dir,
            batch_size=cfg.batch_size,
            use_gpu=cfg.use_gpu,
            gpu_device_id=cfg.gpu_device_id,
            gpu_device_ids=cfg.gpu_device_ids,
            num_workers=cfg.num_workers,
            pin_memory=True,
            is_test_run=cfg.is_test_run,
            embedding_precision=cfg.embedding_precision,
        )

        # Aggregate to slide level using batch processing
        logger.info(
            f"Batch aggregating {len(cfg.slide_paths)} slides with aggregation_method={cfg.aggregation_method}"
        )
        aggregate_slide_features_batch(
            patch_features_h5_paths=intermediate_h5_paths,
            output_h5_paths=output_h5_paths,
            output_pt_paths=output_pt_paths,
            aggregation_method=cfg.aggregation_method,
            model_type=cfg.slide_model_type,
            model_path=cfg.slide_model_path,
            model_dir=cfg.model_dir,
            use_gpu=cfg.use_gpu,
            gpu_device_id=cfg.gpu_device_id,
            gpu_device_ids=cfg.gpu_device_ids,
            slide_batch_size=cfg.slide_batch_size,
        )
    else:
        # Single-step: extract directly to final output (no aggregation)
        extract_patch_features_batch(
            patch_h5_paths=cfg.patch_h5_paths,
            slide_paths=cfg.slide_paths,
            output_h5_paths=output_h5_paths,
            model_type=cfg.model_type,
            model_path=cfg.model_path,
            model_dir=cfg.model_dir,
            batch_size=cfg.batch_size,
            use_gpu=cfg.use_gpu,
            gpu_device_id=cfg.gpu_device_id,
            gpu_device_ids=cfg.gpu_device_ids,
            num_workers=cfg.num_workers,
            pin_memory=True,
            is_test_run=cfg.is_test_run,
            embedding_precision=cfg.embedding_precision,
        )

        # Save as PT format for consistency
        for output_h5, output_pt in zip(output_h5_paths, output_pt_paths):
            with h5py.File(output_h5, "r") as f:
                features = torch.from_numpy(f["features"][:])
                torch.save(features, output_pt)

    logger.info(f"Batch processing complete. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
