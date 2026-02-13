import os
import ssl
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from tqdm import tqdm

from mussel.datasets import WholeSlideImageTileCoordDataset
from mussel.models import ModelType, get_model_factory
from mussel.utils import (
    save_features,
    extract_patch_features_batch,
    aggregate_slide_features_batch,
    resolve_remote_paths,
    get_slide_ids_from_paths,
    ensure_directory_exists,
)
from mussel.utils.file import save_hdf5, save_torch_tensor
from mussel.utils.ml import collate_features
from mussel.utils.multi_slide import (
    SampleSlideGroup,
    SlideInfo,
    compute_subsampling_indices,
    load_sample_batch_csv,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractFeaturesConfig:
    """
    Configuration for extract-features command.

    Supports four input modes:

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

    4. Multi-Slide (Sample) Mode:
       - Provide: sample_batch_csv_path, output_dir (with {sample_id} placeholder)
       - Extracts features from multiple slides belonging to the same sample
       - Optional: sample_id to filter to a single sample, max_tiles_per_sample for subsampling
       - Output: One H5/PT file per sample with concatenated features from all slides

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

    Multi-Slide (Sample) Mode Parameters:
        sample_batch_csv_path (str): Path to CSV file with columns: sample_id, slide_path, patch_h5_path.
        sample_id (Optional[str]): Filter to process only this sample (for parallel job submission).
        max_tiles_per_sample (Optional[int]): Maximum tiles per sample; subsample proportionally if exceeded.
        random_seed (int): Random seed for reproducible subsampling (default: 42).

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
    # Multi-slide (sample) mode parameters
    sample_batch_csv_path: Optional[str] = None
    sample_id: Optional[str] = None
    max_tiles_per_sample: Optional[int] = None
    random_seed: int = 42
    # Processing parameters
    batch_size: int = 64
    use_gpu: bool = True
    gpu_device_id: Optional[int] = None
    gpu_device_ids: Optional[List[int]] = None
    num_workers: int = 16
    is_test_run: bool = False


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
    multi_slide_mode = cfg.sample_batch_csv_path is not None
    batch_mode = cfg.slide_paths is not None

    if multi_slide_mode:
        logger.info("Running in multi-slide (sample) mode")
        _main_multi_slide(cfg)
    elif batch_mode:
        logger.info("Running in batch mode (multiple slides)")
        _main_batch(cfg)
    else:
        logger.info("Running in single-slide mode")
        _main_single(cfg)


@resolve_remote_paths()
def _main_single(cfg: ExtractFeaturesConfig):
    """Process a single slide."""
    if cfg.patch_h5_path is None or cfg.slide_path is None or cfg.output_h5_path is None:
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
    output_h5_paths = [str(output_dir / f"{slide_id}.{cfg.output_h5_suffix}") for slide_id in slide_ids]
    output_pt_paths = [str(output_dir / f"{slide_id}.{cfg.output_pt_suffix}") for slide_id in slide_ids]
    
    # Check if we need two-step processing
    use_two_step = cfg.aggregation_method != "identity"
    
    logger.info(f"Batch extracting features for {len(cfg.slide_paths)} slides")
    
    if use_two_step:
        # Extract to intermediate patch feature files for later aggregation
        intermediate_h5_paths = [
            str(output_dir / f"{slide_id}.patch.h5") 
            for slide_id in slide_ids
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
        )
        
        # Aggregate to slide level using batch processing
        logger.info(f"Batch aggregating {len(cfg.slide_paths)} slides with aggregation_method={cfg.aggregation_method}")
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
        )
        
        # Save as PT format for consistency
        for output_h5, output_pt in zip(output_h5_paths, output_pt_paths):
            with h5py.File(output_h5, "r") as f:
                features = torch.from_numpy(f["features"][:])
                torch.save(features, output_pt)
    
    logger.info(f"Batch processing complete. Output saved to {output_dir}")


@resolve_remote_paths()
def _main_multi_slide(cfg: ExtractFeaturesConfig):
    """Process multiple slides per sample (multi-slide mode)."""
    if cfg.sample_batch_csv_path is None or cfg.output_dir is None:
        raise ValueError(
            "Multi-slide mode requires sample_batch_csv_path and output_dir to be specified"
        )

    # Load and optionally filter samples
    samples = load_sample_batch_csv(cfg.sample_batch_csv_path)

    if cfg.sample_id is not None:
        if cfg.sample_id not in samples:
            raise ValueError(
                f"Sample ID '{cfg.sample_id}' not found in CSV. "
                f"Available samples: {list(samples.keys())}"
            )
        samples = {cfg.sample_id: samples[cfg.sample_id]}
        logger.info(f"Filtered to single sample: {cfg.sample_id}")

    logger.info(f"Processing {len(samples)} sample(s)")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    ensure_directory_exists(output_dir)

    # Load model once for all samples
    gpu_device_id = cfg.gpu_device_ids if cfg.gpu_device_ids else cfg.gpu_device_id

    logger.info("Loading model checkpoint (once for all samples)")
    model_factory = get_model_factory(cfg.model_type)
    if model_factory is None:
        raise ValueError(f"Model type {cfg.model_type} not recognized")
    model = model_factory.get_model(cfg.model_path, cfg.use_gpu, gpu_device_id)
    model_fun = model.get_model_fun()
    preprocessing = model.get_preprocessing_fun()

    # Process each sample
    for sample_id, sample_group in samples.items():
        logger.info(f"Processing sample: {sample_id} ({len(sample_group.slides)} slides)")

        # Generate output paths with {sample_id} placeholder support
        output_h5_path = str(output_dir / f"{sample_id}.{cfg.output_h5_suffix}")
        output_pt_path = str(output_dir / f"{sample_id}.{cfg.output_pt_suffix}")

        # Support {sample_id} placeholder in output_h5_path/output_pt_path if provided
        if cfg.output_h5_path and "{sample_id}" in cfg.output_h5_path:
            output_h5_path = cfg.output_h5_path.replace("{sample_id}", sample_id)
        if cfg.output_pt_path and "{sample_id}" in cfg.output_pt_path:
            output_pt_path = cfg.output_pt_path.replace("{sample_id}", sample_id)

        extract_features_multi_slide(
            sample_group=sample_group,
            output_h5_path=output_h5_path,
            output_pt_path=output_pt_path,
            model_fun=model_fun,
            preprocessing=preprocessing,
            max_tiles_per_sample=cfg.max_tiles_per_sample,
            random_seed=cfg.random_seed,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            is_test_run=cfg.is_test_run,
        )

    logger.info(f"Multi-slide processing complete. Output saved to {output_dir}")


def extract_features_multi_slide(
    sample_group: SampleSlideGroup,
    output_h5_path: str,
    output_pt_path: str,
    model_fun,
    preprocessing,
    max_tiles_per_sample: Optional[int] = None,
    random_seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 16,
    is_test_run: bool = False,
) -> None:
    """Extract and concatenate features from multiple slides belonging to a sample.

    Args:
        sample_group: SampleSlideGroup containing sample_id and list of SlideInfo.
        output_h5_path: Path to save concatenated features in HDF5 format.
        output_pt_path: Path to save concatenated features in PyTorch format.
        model_fun: Model inference function.
        preprocessing: Preprocessing function for images.
        max_tiles_per_sample: Maximum total tiles; subsample proportionally if exceeded.
        random_seed: Random seed for reproducible subsampling.
        batch_size: Batch size for feature extraction.
        num_workers: Number of data loader workers.
        is_test_run: If True, only process first 3 batches per slide.
    """
    sample_id = sample_group.sample_id
    slides = sample_group.slides

    # Apply subsampling if needed
    if max_tiles_per_sample is not None and max_tiles_per_sample > 0:
        slides = compute_subsampling_indices(
            slides,
            max_tiles=max_tiles_per_sample,
            strategy="proportional",
            random_seed=random_seed,
        )

    logger.info(
        f"Sample {sample_id}: {sample_group.total_tiles} total tiles, "
        f"{sum(s.num_selected_tiles for s in slides)} selected tiles"
    )

    all_features = []
    all_coords = []
    all_slide_idx = []

    # Process each slide
    for slide_idx, slide in enumerate(slides):
        logger.info(
            f"  Slide {slide_idx + 1}/{len(slides)}: {slide.slide_path} "
            f"({slide.num_selected_tiles}/{slide.num_tiles} tiles)"
        )

        # Read coordinates and attributes from H5 file
        with h5py.File(slide.patch_h5_path, "r") as h5f:
            coords = h5f["coords"][:]
            attrs = {key: h5f["coords"].attrs[key] for key in h5f["coords"].attrs}

        # Create dataset with optional limit_to_indices for subsampling
        dataset = WholeSlideImageTileCoordDataset(
            coords=coords,
            attrs=attrs,
            slide_path=slide.slide_path,
            use_imagenet_rgb_dist=preprocessing is None,
            preprocess=preprocessing,
            limit_to_indices=slide.selected_indices,
            init_wsi_in_worker=num_workers > 0,
        )

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=dataset.worker_init if num_workers > 0 else None,
            collate_fn=collate_features,
            shuffle=False,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )

        # Extract features for this slide
        slide_features = []
        slide_coords = []

        for count, (batch, batch_coords) in enumerate(
            tqdm(loader, desc=f"Slide {slide_idx + 1}")
        ):
            if is_test_run and count > 2:
                break

            # Skip empty batches
            if batch.numel() == 0:
                logger.warning(f"Skipping empty batch {count}")
                continue

            features = model_fun(batch).numpy()
            slide_features.append(features)
            slide_coords.append(batch_coords)

        if slide_features:
            slide_features = np.concatenate(slide_features, axis=0)
            slide_coords = np.concatenate(slide_coords, axis=0)

            all_features.append(slide_features)
            all_coords.append(slide_coords)
            all_slide_idx.append(np.full(len(slide_features), slide_idx, dtype=np.int32))

            logger.info(f"    Extracted {len(slide_features)} feature vectors")

    # Concatenate all slides
    if not all_features:
        raise ValueError(f"No features extracted for sample {sample_id}")

    features = np.concatenate(all_features, axis=0)
    coords = np.concatenate(all_coords, axis=0)
    slide_idx_array = np.concatenate(all_slide_idx, axis=0)

    logger.info(
        f"Sample {sample_id}: Total {features.shape[0]} features, "
        f"shape {features.shape}"
    )

    # Save to HDF5
    ensure_directory_exists(output_h5_path, is_file_path=True)

    asset_dict = {
        "features": features,
        "coords": coords,
        "slide_idx": slide_idx_array,
    }
    save_hdf5(output_h5_path, asset_dict, mode="w")

    # Add sample metadata as H5 attributes
    with h5py.File(output_h5_path, "a") as h5f:
        h5f.attrs["sample_id"] = sample_id
        h5f.attrs["num_slides"] = len(slides)

    logger.info(f"Saved features to {output_h5_path}")

    # Save to PyTorch format
    features_tensor = torch.from_numpy(features)
    save_torch_tensor(output_pt_path, features_tensor)
    logger.info(f"Saved features to {output_pt_path}")


if __name__ == "__main__":
    main()
