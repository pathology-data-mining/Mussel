import gc
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from functools import singledispatch
from tqdm import tqdm

from mussel.datasets import WholeSlideImageH5Dataset, WholeSlideImageTileCoordDataset
from mussel.models import (
    ModelType,
    get_model_factory,
    validate_slide_encoder_compatibility,
    get_required_patch_encoder,
    get_default_patch_size,
)

from .file import save_hdf5
from .ml import collate_features
from .timer import timed

logger = logging.getLogger(__name__)


def get_model_path_from_dir(model_dir, model_type):
    """
    Get model path from model_dir if available.
    
    Args:
        model_dir: Directory containing pre-downloaded models
        model_type: ModelType enum
        
    Returns:
        Path to model if found in model_dir, None otherwise
    """
    if model_dir is None:
        return None
    
    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        return None
    
    # Check for model directory named after the model type
    model_subdir = model_dir_path / model_type.name
    if model_subdir.exists() and model_subdir.is_dir():
        logger.info(f"Found {model_type.name} in model_dir: {model_subdir}")
        return str(model_subdir)
    
    # Check for model directory named with lowercase
    model_subdir_lower = model_dir_path / model_type.name.lower()
    if model_subdir_lower.exists() and model_subdir_lower.is_dir():
        logger.info(f"Found {model_type.name} in model_dir: {model_subdir_lower}")
        return str(model_subdir_lower)
    
    # Check for .pth file (pickled models) - uppercase
    model_file_pth = model_dir_path / f"{model_type.name}.pth"
    if model_file_pth.exists() and model_file_pth.is_file():
        logger.info(f"Found {model_type.name} in model_dir: {model_file_pth}")
        return str(model_file_pth)
    
    # Check for .pth file (pickled models) - lowercase
    model_file_pth_lower = model_dir_path / f"{model_type.name.lower()}.pth"
    if model_file_pth_lower.exists() and model_file_pth_lower.is_file():
        logger.info(f"Found {model_type.name} in model_dir: {model_file_pth_lower}")
        return str(model_file_pth_lower)
    
    return None


@singledispatch
def process_dataset(
    dataset,
    loader,
    model_fun,
    patch_h5_path=None,
    output_h5_path=None,
):
    """
    Args:
            dataset: dataset object
            loader: dataloader object
            model_fun: function to extract features from a batch of images
            patch_h5_path: path to the h5 file containing patch coordinates (if any)
            output_h5_path: path to save the extracted features (if any)
    """
    pass


@process_dataset.register(WholeSlideImageTileCoordDataset)
def _(dataset: WholeSlideImageTileCoordDataset, loader, model_fun, is_test_run=False):
    """Process a WholeSlideImageTileCoordDataset to extract features.

    Args:
        dataset: WholeSlideImageTileCoordDataset instance.
        loader: DataLoader for the dataset.
        model_fun: Function to extract features from a batch of images.
        is_test_run: If True, only process first 3 batches (default: False).

    Returns:
        Tuple of (features array, labels array).
    """
    all_features = []
    all_labels = []
    for count, (batch, labels) in enumerate(tqdm(loader, desc="Extracting features")):
        if is_test_run and count > 2:
            break

        features = model_fun(batch)
        all_features.append(features.numpy())
        all_labels.append(labels)
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_features, all_labels


@process_dataset.register(ImageFolder)
def _(
    dataset: ImageFolder,
    loader,
    model_fun,
    patch_h5_path=None,
    output_h5_path=None,
    is_test_run=False,
):
    """Process an ImageFolder dataset to extract features and save to HDF5.

    Args:
        dataset: ImageFolder instance.
        loader: DataLoader for the dataset.
        model_fun: Function to extract features from a batch of images.
        patch_h5_path: Path to the h5 file containing patch coordinates (unused).
        output_h5_path: Path to save the extracted features.
        is_test_run: If True, only process first 3 batches (default: False).

    Returns:
        Path to the output HDF5 file.
    """
    asset_dict = {
        "image_paths": np.array([x[0] for x in dataset.imgs]).astype("T"),
        "class_to_idx": np.array(
            [np.asarray([k, v], dtype="T") for k, v in dataset.class_to_idx.items()]
        ),
    }
    save_hdf5(output_h5_path, asset_dict, attr_h5_path=None, mode="w")
    for count, (batch, labels) in enumerate(tqdm(loader, desc="Extracting features")):
        if is_test_run and count > 2:
            break
        labels = labels.numpy()

        features = model_fun(batch)
        features = features.numpy()
        asset_dict = {
            "features": features,
            "class": labels,
        }
        save_hdf5(output_h5_path, asset_dict, attr_h5_path=None, mode="a")
    return output_h5_path


@process_dataset.register(WholeSlideImageH5Dataset)
def _(
    dataset: WholeSlideImageH5Dataset,
    loader,
    model_fun,
    patch_h5_path=None,
    output_h5_path=None,
    is_test_run=False,
):
    """Process a WholeSlideImageH5Dataset to extract features and save to HDF5.

    Args:
        dataset: WholeSlideImageH5Dataset instance.
        loader: DataLoader for the dataset.
        model_fun: Function to extract features from a batch of images.
        patch_h5_path: Path to the h5 file containing patch coordinates.
        output_h5_path: Path to save the extracted features.
        is_test_run: If True, only process first 3 batches (default: False).

    Returns:
        Path to the output HDF5 file.
    """
    mode = "w"
    for count, (batch, coords) in enumerate(tqdm(loader, desc="Extracting features")):
        if is_test_run and count > 2:
            break

        features = model_fun(batch)
        features = features.numpy()
        
        asset_dict = {"features": features, "coords": coords}
        save_hdf5(output_h5_path, asset_dict, attr_h5_path=patch_h5_path, mode=mode)
        mode = "a"
    return output_h5_path


def _apply_slide_aggregation(
    features,
    aggregation_method="identity",
    slide_model_type=None,
    slide_model_path=None,
    use_gpu=True,
    gpu_device_id=None,
    gpu_device_ids=None,
    coords=None,
    patch_size=None,
):
    """Apply slide-level aggregation to patch features.

    Helper function shared by get_features() and aggregate_slide_features().

    Args:
        features: Numpy array of patch-level features.
        aggregation_method: Method for aggregating features (default: "identity").
            Options: "identity", "mean", "max", "model".
        slide_model_type: Type of slide encoder model (only when aggregation_method="model").
        slide_model_path: Optional path to slide encoder model weights.
        use_gpu: Whether to use GPU for model-based aggregation (default: True).
        gpu_device_id: GPU device ID to use.
        gpu_device_ids: List of GPU device IDs for multi-GPU.
        coords: Optional numpy array of patch coordinates (required for some slide encoders like GIGAPATH_SLIDE, TITAN_SLIDE).
        patch_size: Optional patch size at level 0 (required for TITAN_SLIDE).
            If not provided, will be extracted from h5 file 'coords' attributes or default to 256.

    Returns:
        Numpy array of aggregated features.
    """
    if aggregation_method == "identity":
        # No aggregation - keep all patch features
        logger.info("Using identity aggregation (no aggregation)")
        return features
    elif aggregation_method == "mean":
        # Mean pooling across patches
        aggregated_features = np.mean(features, axis=0, keepdims=True)
        logger.info(
            f"Applied mean pooling: {features.shape} -> {aggregated_features.shape}"
        )
        return aggregated_features
    elif aggregation_method == "max":
        # Max pooling across patches
        aggregated_features = np.max(features, axis=0, keepdims=True)
        logger.info(
            f"Applied max pooling: {features.shape} -> {aggregated_features.shape}"
        )
        return aggregated_features
    elif aggregation_method == "model":
        # Model-based aggregation using a slide encoder
        if slide_model_type is None:
            raise ValueError(
                "slide_model_type must be provided when using model-based aggregation"
            )

        logger.info(f"Using model-based aggregation with {slide_model_type}")

        if gpu_device_ids:
            gpu_device_id = gpu_device_ids

        # Load the slide encoder model
        model_factory = get_model_factory(slide_model_type)
        if model_factory is None:
            raise ValueError(f"Slide model type {slide_model_type} not recognized")
        model = model_factory.get_model(slide_model_path, use_gpu, gpu_device_id)
        model_fun = model.get_model_fun()

        # Convert features to tensor and apply model
        features_tensor = torch.from_numpy(features).unsqueeze(0)  # Add batch dimension

        # Some slide encoders require coordinates and/or patch size
        # GIGAPATH_SLIDE: requires (features, coords)
        # TITAN_SLIDE: requires (features, coords, patch_size)
        with torch.no_grad():
            try:
                if slide_model_type == ModelType.TITAN_SLIDE:
                    # TITAN requires features, coords, and patch_size
                    if coords is None:
                        raise ValueError("TITAN_SLIDE requires coordinates")
                    if patch_size is None:
                        # Get the default patch size for the required patch encoder
                        patch_encoder = get_required_patch_encoder(slide_model_type)
                        patch_size = get_default_patch_size(patch_encoder)
                        logger.warning(
                            f"patch_size not provided, using default for {patch_encoder}: {patch_size}"
                        )
                    coords_tensor = torch.from_numpy(coords).long().unsqueeze(
                        0
                    )  # Add batch dimension and convert to int64
                    aggregated_features = (
                        model_fun(features_tensor, coords_tensor, patch_size).cpu().numpy()
                    )
                elif slide_model_type == ModelType.GIGAPATH_SLIDE:
                    # GIGAPATH requires features and coords
                    if coords is None:
                        raise ValueError("GIGAPATH_SLIDE requires coordinates")

                    coords_tensor = torch.from_numpy(coords).unsqueeze(
                        0
                    )  # Add batch dimension
                    aggregated_features = (
                        model_fun(features_tensor, coords_tensor).numpy()
                    )
                else:
                    # Other slide encoders may only need features
                    aggregated_features = model_fun(features_tensor).cpu().numpy()
            except torch.cuda.OutOfMemoryError as e:
                num_patches = features.shape[0]
                logger.error(
                    f"GPU Out of Memory during slide aggregation with {slide_model_type.name}"
                )
                logger.error(f"Number of patches: {num_patches:,}")
                logger.error(f"Feature dimensions: {features.shape}")
                logger.error(
                    f"\nSuggestions to fix OOM for {slide_model_type.name}:"
                )
                logger.error("  1. Use more aggressive tissue segmentation to reduce patches")
                logger.error("     (increase tissue_area_threshold or step_size)")
                logger.error("  2. Use mean/max pooling instead of model-based aggregation:")
                logger.error("     aggregation_method=mean or aggregation_method=max")
                logger.error(f"  3. Process a smaller slide (current: {num_patches:,} patches)")
                logger.error(f"\nOriginal error: {str(e)}")
                raise RuntimeError(
                    f"GPU Out of Memory: {slide_model_type.name} cannot process {num_patches:,} patches. "
                    f"Maximum recommended: ~10,000 patches. "
                    f"Try using mean pooling (aggregation_method=mean) or more aggressive segmentation."
                ) from e

        logger.info(
            f"Applied model aggregation: {features.shape} -> {aggregated_features.shape}"
        )
        return aggregated_features
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")


def get_features(
    coords,
    slide_path,
    attrs,
    model_type=ModelType.CLIP,
    model_path=None,
    model=None,
    batch_size=64,
    use_gpu=True,
    gpu_device_id=None,
    gpu_device_ids=None,
    pin_memory=True,
    num_workers=16,
    is_test_run=False,
    use_slide_encoder=False,
    slide_model_type=None,
    slide_model_path=None,
    aggregation_method="identity",
):
    """Extract features from whole slide image tiles.

    Args:
        coords: Tile coordinates array.
        slide_path: Path to the whole slide image.
        attrs: Dictionary of tile attributes (patch_size, patch_level, mpp, etc.).
        model_type: Type of foundation model to use (default: ModelType.CLIP).
            When using model-based aggregation with a slide encoder, this will be automatically
            set to the required patch encoder if not already specified correctly.
        model_path: Optional path to model weights.
        model: Optional pre-loaded model instance. If provided, model_type and model_path are ignored.
        batch_size: Batch size for feature extraction (default: 64).
        use_gpu: Whether to use GPU for inference (default: True).
        gpu_device_id: GPU device ID to use.
        gpu_device_ids: List of GPU device IDs for multi-GPU.
        pin_memory: Whether to pin memory for data loading (default: True).
        num_workers: Number of worker processes for data loading (default: 16).
        is_test_run: If True, only process first 3 batches (default: False).
        use_slide_encoder: If True, apply slide-level encoding after patch extraction.
        slide_model_type: Type of slide encoder model (only when use_slide_encoder=True).
            The required patch encoder will be automatically inferred and used.
        slide_model_path: Optional path to slide encoder model weights.
        aggregation_method: Aggregation method when using slide encoder (default: "identity").
            Options: "identity", "mean", "max", "model".

    Returns:
        Tuple of (features array, labels array).
    """
    if model is None:
        logger.info("loading model checkpoint")

        if gpu_device_ids:
            gpu_device_id = gpu_device_ids

        # Auto-set aggregation_method to "model" if slide_model_type is specified
        if (
            use_slide_encoder
            and slide_model_type is not None
            and aggregation_method != "model"
        ):
            logger.info(
                f"Auto-setting aggregation_method to 'model' since slide_model_type "
                f"({slide_model_type}) is specified"
            )
            aggregation_method = "model"

        # Auto-infer patch encoder from slide encoder if using model-based aggregation
        if (
            use_slide_encoder
            and aggregation_method == "model"
            and slide_model_type is not None
        ):
            required_patch_encoder = get_required_patch_encoder(slide_model_type)
            if model_type != required_patch_encoder:
                logger.info(
                    f"Auto-selecting patch encoder {required_patch_encoder} "
                    f"as required by slide encoder {slide_model_type}"
                )
                model_type = required_patch_encoder
            # Validate compatibility
            validate_slide_encoder_compatibility(model_type, slide_model_type)

        model_factory = get_model_factory(model_type)
        if model_factory is None:
            raise ValueError("model not recognized")
        model = model_factory.get_model(model_path, use_gpu, gpu_device_id)
    else:
        logger.info("using pre-loaded model")
    preprocessing = model.get_preprocessing_fun()

    dataset = WholeSlideImageTileCoordDataset(
        coords=coords,
        attrs=attrs,
        slide_path=slide_path,
        use_imagenet_rgb_dist=preprocessing is None,
        preprocess=preprocessing,
        init_wsi_in_worker=num_workers > 0,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=dataset.worker_init if num_workers > 0 else None,
        collate_fn=collate_features,
        shuffle=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,  # Increased from 2 to 4 for better GPU utilization
    )

    features, labels = process_dataset(
        dataset, loader, model_fun=model.get_model_fun(), is_test_run=is_test_run
    )

    # Apply slide-level encoding if requested
    if use_slide_encoder:
        logger.info("Applying slide-level encoding")

        # Extract patch_size from attrs if available
        patch_size = attrs.get("patch_size") if attrs else None
        if patch_size is not None:
            logger.info(f"Using patch_size from attrs: {patch_size}")

        features = _apply_slide_aggregation(
            features,
            aggregation_method=aggregation_method,
            slide_model_type=slide_model_type,
            slide_model_path=slide_model_path,
            use_gpu=use_gpu,
            gpu_device_id=gpu_device_id,
            gpu_device_ids=gpu_device_ids,
            coords=coords,
            patch_size=patch_size,
        )

    return features, labels


@timed
def extract_patch_features(
    patch_h5_path,
    slide_path,
    output_h5_path,
    model_type=ModelType.CLIP,
    model_path=None,
    model_save_path=None,
    patch_path=None,
    batch_size=64,
    use_gpu=True,
    gpu_device_id=None,
    gpu_device_ids=None,
    num_workers=16,
    pin_memory=True,
    is_test_run=False,
):
    """Extract patch-level features from whole slide image (Step 1: Patch Encoding).

    This function performs patch-level feature extraction, converting image patches
    into feature embeddings using a foundation model. The output contains features
    for individual patches/tiles.

    Args:
        patch_h5_path: Path to the h5 file containing patch coordinates.
        slide_path: Path to the whole slide image.
        output_h5_path: Path to save the extracted patch-level features in HDF5 format.
        model_type: Type of foundation model to use (default: ModelType.CLIP).
        model_path: Optional path to model weights.
        model_save_path: Optional path to save the model.
        patch_path: Optional path to folder with pre-extracted patches.
        batch_size: Batch size for feature extraction (default: 64).
        use_gpu: Whether to use GPU for inference (default: True).
        gpu_device_id: GPU device ID to use.
        gpu_device_ids: List of GPU device IDs for multi-GPU.
        num_workers: Number of worker processes for data loading (default: 16).
        pin_memory: Whether to pin memory for data loading (default: True).
        is_test_run: If True, only process first 3 batches (default: False).

    Returns:
        Path to the output HDF5 file containing patch-level features.
    """
    if gpu_device_ids:
        gpu_device_id = gpu_device_ids

    logger.info("Step 1: Extracting patch-level features")
    logger.info("loading model checkpoint")
    logger.info(f"Using batch_size={batch_size} for model {model_type}")

    model_factory = get_model_factory(model_type)
    if model_factory is None:
        raise ValueError("model not recognized")
    model = model_factory.get_model(model_path, use_gpu, gpu_device_id)
    if model_save_path is not None:
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"saving model to {model_save_path}")
        model.save(model_save_path)
    preprocessing = model.get_preprocessing_fun()

    if patch_path:
        dataset = ImageFolder(
            root=patch_path,
            transform=preprocessing,
        )

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=None,
            shuffle=False,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    elif patch_h5_path:
        dataset = WholeSlideImageH5Dataset(
            h5_path=patch_h5_path,
            slide_path=slide_path,
            preprocess=preprocessing,
            use_imagenet_rgb_dist=preprocessing is None,
            init_wsi_in_worker=num_workers > 0,
        )

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_features,
            worker_init_fn=dataset.worker_init if num_workers > 0 else None,
            shuffle=False,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    else:
        raise ValueError("Either patch_path or patch_h5_path must be provided")

    process_dataset(
        dataset,
        loader,
        model_fun=model.get_model_fun(),
        patch_h5_path=patch_h5_path,
        output_h5_path=output_h5_path,
        is_test_run=is_test_run,
    )

    logger.info(f"Patch-level features saved to {output_h5_path}")
    return output_h5_path


def extract_patch_features_batch(
    patch_h5_paths,
    slide_paths,
    output_h5_paths,
    model_type=ModelType.CLIP,
    model_path=None,
    model_dir=None,
    batch_size=64,
    use_gpu=True,
    gpu_device_id=None,
    gpu_device_ids=None,
    num_workers=16,
    pin_memory=True,
    is_test_run=False,
):
    """Extract patch-level features from multiple slides in batch mode.

    This function performs patch-level feature extraction for multiple slides,
    loading the model only once and processing tiles from all slides together
    in batches. This provides significant performance benefits by:
    1. Loading the patch encoder model only once (vs N times for N slides)
    2. Better GPU utilization through continuous batching
    3. Reducing model initialization overhead

    Args:
        patch_h5_paths: List of paths to h5 files containing patch coordinates.
        slide_paths: List of paths to whole slide images.
        output_h5_paths: List of paths to save extracted patch-level features.
        model_type: Type of foundation model to use (default: ModelType.CLIP).
        model_path: Optional path to model weights.
        model_dir: Optional directory containing pre-downloaded models.
        batch_size: Batch size for feature extraction (default: 64).
        use_gpu: Whether to use GPU for inference (default: True).
        gpu_device_id: GPU device ID to use.
        gpu_device_ids: List of GPU device IDs for multi-GPU.
        num_workers: Number of worker processes for data loading (default: 16).
        pin_memory: Whether to pin memory for data loading (default: True).
        is_test_run: If True, only process first 3 batches per slide (default: False).

    Returns:
        List of paths to output HDF5 files containing patch-level features.
    """
    # Validate inputs
    if not patch_h5_paths:
        return []
    
    if not (len(patch_h5_paths) == len(slide_paths) == len(output_h5_paths)):
        raise ValueError(
            f"Input lists must have the same length: "
            f"patch_h5_paths={len(patch_h5_paths)}, "
            f"slide_paths={len(slide_paths)}, "
            f"output_h5_paths={len(output_h5_paths)}"
        )
    
    num_slides = len(patch_h5_paths)
    logger.info(f"Batch extracting patch-level features for {num_slides} slides")
    
    if gpu_device_ids:
        gpu_device_id = gpu_device_ids

    # Resolve model_path from model_dir if provided
    if model_path is None and model_dir is not None:
        model_path = get_model_path_from_dir(model_dir, model_type)
        if model_path:
            logger.info(f"Using model from model_dir: {model_path}")

    # Load the model once for all slides
    logger.info("Loading model checkpoint (once for all slides)")
    logger.info(f"Using batch_size={batch_size} for model {model_type}")
    model_factory = get_model_factory(model_type)
    if model_factory is None:
        raise ValueError("model not recognized")
    model = model_factory.get_model(model_path, use_gpu, gpu_device_id)
    preprocessing = model.get_preprocessing_fun()
    model_fun = model.get_model_fun()

    # Process each slide with the shared model
    for i, (patch_h5_path, slide_path, output_h5_path) in enumerate(
        zip(patch_h5_paths, slide_paths, output_h5_paths)
    ):
        logger.info(f"Processing slide {i+1}/{num_slides}: {slide_path}")
        
        # Create dataset and loader for this slide
        dataset = WholeSlideImageH5Dataset(
            h5_path=patch_h5_path,
            slide_path=slide_path,
            preprocess=preprocessing,
            use_imagenet_rgb_dist=preprocessing is None,
            init_wsi_in_worker=num_workers > 0,
        )

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_features,
            worker_init_fn=dataset.worker_init if num_workers > 0 else None,
            shuffle=False,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )

        # Process this slide's tiles
        process_dataset(
            dataset,
            loader,
            model_fun=model_fun,
            patch_h5_path=patch_h5_path,
            output_h5_path=output_h5_path,
            is_test_run=is_test_run,
        )

        logger.info(f"Patch-level features saved to {output_h5_path}")

    # Clean up model to free GPU/CPU memory
    del model
    del model_fun
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Model cleaned up, memory freed")
    
    logger.info(f"Batch extraction complete for {num_slides} slides")
    return output_h5_paths


@timed
def aggregate_slide_features_batch(
    patch_features_h5_paths,
    output_h5_paths=None,
    output_pt_paths=None,
    aggregation_method="identity",
    model_type=None,
    model_path=None,
    model_dir=None,
    use_gpu=True,
    gpu_device_id=None,
    gpu_device_ids=None,
    slide_batch_size=8,
):
    """Aggregate patch-level features to slide-level for multiple slides (Step 2: Batch Slide Encoding).

    This function takes patch-level features from multiple slides and aggregates them into slide-level
    representations in batches. This is more efficient than processing slides one-by-one when using
    model-based aggregation, as it:
    1. Loads the model only once
    2. Processes multiple slides in parallel on GPU
    3. Reduces model initialization overhead

    Args:
        patch_features_h5_paths: List of paths to HDF5 files with patch-level features.
        output_h5_paths: Optional list of paths to save slide-level features in HDF5 format.
        output_pt_paths: Optional list of paths to save slide-level features in PyTorch format.
        aggregation_method: Method for aggregating features (default: "identity").
            - "identity": No aggregation, keeps all patch features (backward compatible)
            - "mean": Mean pooling across patches
            - "max": Max pooling across patches
            - "model": Use a slide encoder model for aggregation
        model_type: Type of slide encoder model to use (only when aggregation_method="model").
        model_path: Optional path to slide encoder model weights.
        model_dir: Optional directory containing pre-downloaded models.
        use_gpu: Whether to use GPU for model-based aggregation (default: True).
        gpu_device_id: GPU device ID to use.
        gpu_device_ids: List of GPU device IDs for multi-GPU.
        slide_batch_size: Number of slides to process in a single batch (default: 8).

    Returns:
        Tuple of (output_h5_paths, output_pt_paths) if saving.
    """
    logger.info(f"Step 2: Batch aggregating patch features to slide level for {len(patch_features_h5_paths)} slides")
    
    num_slides = len(patch_features_h5_paths)
    
    # For non-model aggregation methods, process each slide directly
    # without loading a model
    if aggregation_method != "model":
        logger.info(f"Processing {num_slides} slides with aggregation_method={aggregation_method}")
        successful_slides = []
        failed_slides = []
        
        for i, patch_h5_path in enumerate(patch_features_h5_paths):
            output_h5 = output_h5_paths[i] if output_h5_paths else None
            output_pt = output_pt_paths[i] if output_pt_paths else None
            slide_name = Path(patch_h5_path).stem
            
            try:
                with h5py.File(patch_h5_path, "r") as file:
                    features = file["features"][:]
                    logger.info(f"Loaded patch features with shape: {features.shape} for slide {i+1}/{num_slides}")
                    
                    # Load coordinates if available
                    coords = file["coords"][:] if "coords" in file else None
                    
                    # Load patch_size from coords attributes if available
                    patch_size = None
                    if "coords" in file and "patch_size" in file["coords"].attrs:
                        patch_size = file["coords"].attrs["patch_size"]
                    
                    # Apply aggregation using shared helper function
                    aggregated_features = _apply_slide_aggregation(
                        features,
                        aggregation_method=aggregation_method,
                        slide_model_type=model_type,
                        slide_model_path=model_path,
                        use_gpu=use_gpu,
                        gpu_device_id=gpu_device_id,
                        gpu_device_ids=gpu_device_ids,
                        coords=coords,
                        patch_size=patch_size,
                    )
                    
                    # Save to HDF5 if requested
                    if output_h5:
                        logger.info(f"Saving aggregated features to {output_h5}")
                        asset_dict = {"features": aggregated_features}
                        
                        # Copy coordinates if they exist and we're using identity
                        if aggregation_method == "identity" and "coords" in file:
                            asset_dict["coords"] = file["coords"][:]
                        
                        save_hdf5(output_h5, asset_dict, attr_h5_path=None, mode="w")
                    
                    # Save to PyTorch if requested
                    if output_pt:
                        logger.info(f"Saving aggregated features to {output_pt}")
                        features_tensor = torch.from_numpy(aggregated_features)
                        from .file import save_torch_tensor
                        save_torch_tensor(output_pt, features_tensor)
                    
                    successful_slides.append(slide_name)
                    
            except Exception as e:
                logger.error(f"Failed to process slide {slide_name} (slide {i+1}/{num_slides}): {str(e)}")
                logger.error(f"  Input: {patch_h5_path}")
                if output_h5:
                    logger.error(f"  Output (not created): {output_h5}")
                failed_slides.append(slide_name)
                # Continue with next slide
                continue
        
        if failed_slides:
            logger.warning(f"\n=== Batch Processing Summary ===")
            logger.warning(f"Successfully processed: {len(successful_slides)}/{num_slides} slides")
            logger.warning(f"Failed: {len(failed_slides)} slides")
            logger.warning(f"Failed slides: {', '.join(failed_slides)}")
        else:
            logger.info(f"\n=== All {num_slides} slides processed successfully ===")
        
        return output_h5_paths, output_pt_paths
    
    # Model-based aggregation with batching
    logger.info(f"Using batch processing with slide_batch_size={slide_batch_size}")
    
    if gpu_device_ids:
        gpu_device_id = gpu_device_ids
    
    # Resolve model_path from model_dir if provided
    if model_path is None and model_dir is not None:
        model_path = get_model_path_from_dir(model_dir, model_type)
        if model_path:
            logger.info(f"Using slide model from model_dir: {model_path}")
    
    # Load the slide encoder model once
    logger.info(f"Loading slide encoder model: {model_type}")
    logger.info(f"Using slide_batch_size={slide_batch_size} for slide model {model_type}")
    model_factory = get_model_factory(model_type)
    if model_factory is None:
        raise ValueError(f"Slide model type {model_type} not recognized")
    model = model_factory.get_model(model_path, use_gpu, gpu_device_id)
    model_fun = model.get_model_fun()
    
    successful_slides = []
    failed_slides = []
    
    # Process slides in batches
    for batch_start in range(0, num_slides, slide_batch_size):
        batch_end = min(batch_start + slide_batch_size, num_slides)
        batch_indices = range(batch_start, batch_end)
        
        logger.info(f"Processing slides {batch_start+1}-{batch_end} of {num_slides}")
        
        # Load features and metadata for current batch
        batch_features = []
        batch_coords = []
        batch_patch_sizes = []
        batch_slide_names = []
        
        for i in batch_indices:
            with h5py.File(patch_features_h5_paths[i], "r") as file:
                features = file["features"][:]
                coords = file["coords"][:] if "coords" in file else None
                
                patch_size = None
                if "coords" in file and "patch_size" in file["coords"].attrs:
                    patch_size = file["coords"].attrs["patch_size"]
                
                batch_features.append(features)
                batch_coords.append(coords)
                batch_patch_sizes.append(patch_size)
                batch_slide_names.append(Path(patch_features_h5_paths[i]).stem)
        
        # Process batch based on model type requirements
        with torch.no_grad():
            if model_type == ModelType.TITAN_SLIDE:
                # TITAN requires features, coords, and patch_size per slide
                # Note: We process slides sequentially here because TITAN_SLIDE expects
                # variable-length sequences per slide. The main performance benefit comes
                # from loading the model once rather than N times.
                # Future optimization: Implement true batching with padding if model supports it
                aggregated_batch = []
                for slide_idx, (features, coords, patch_size, slide_name) in enumerate(zip(batch_features, batch_coords, batch_patch_sizes, batch_slide_names)):
                    try:
                        if coords is None:
                            raise ValueError("TITAN_SLIDE requires coordinates")
                        if patch_size is None:
                            patch_size = 256
                            logger.warning(f"patch_size not provided, using default: {patch_size}")
                        
                        features_tensor = torch.from_numpy(features).unsqueeze(0)
                        coords_tensor = torch.from_numpy(coords).long().unsqueeze(0)
                        agg_features = model_fun(features_tensor, coords_tensor, patch_size).cpu().numpy()
                        aggregated_batch.append(agg_features)
                        successful_slides.append(slide_name)
                    except Exception as e:
                        logger.error(f"Failed to process slide {slide_name}: {str(e)}")
                        aggregated_batch.append(None)
                        failed_slides.append(slide_name)
                        # Continue with next slide
                        
            elif model_type == ModelType.GIGAPATH_SLIDE:
                # GIGAPATH requires features and coords per slide
                # Note: We process slides sequentially here because GIGAPATH_SLIDE expects
                # variable-length sequences per slide. The main performance benefit comes
                # from loading the model once rather than N times.
                # Future optimization: Implement true batching with padding if model supports it
                aggregated_batch = []
                for slide_idx, (features, coords, slide_name) in enumerate(zip(batch_features, batch_coords, batch_slide_names)):
                    try:
                        if coords is None:
                            raise ValueError("GIGAPATH_SLIDE requires coordinates")
                        
                        features_tensor = torch.from_numpy(features).unsqueeze(0)  # (1, N, D)
                        coords_tensor = torch.from_numpy(coords).unsqueeze(0)  # (1, N, 2)
                        agg_features = model_fun(features_tensor, coords_tensor).cpu().numpy()
                        aggregated_batch.append(agg_features)
                        successful_slides.append(slide_name)
                    except Exception as e:
                        import traceback
                        logger.error(f"Failed to process slide {slide_name}: {str(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        aggregated_batch.append(None)
                        failed_slides.append(slide_name)
                        # Continue with next slide
                    
            else:
                # Other slide encoders may only need features
                # Stack features from all slides in batch and process together
                try:
                    features_list = [torch.from_numpy(f).unsqueeze(0) for f in batch_features]
                    features_batch = torch.cat(features_list, dim=0)
                    
                    # Process entire batch at once for maximum efficiency
                    aggregated_batch_tensor = model_fun(features_batch).cpu().numpy()
                    aggregated_batch = [aggregated_batch_tensor[i:i+1] for i in range(len(batch_features))]
                    successful_slides.extend(batch_slide_names)
                except Exception as e:
                    logger.error(f"Failed to process batch {batch_start+1}-{batch_end}: {str(e)}")
                    # Mark all slides in batch as failed
                    aggregated_batch = [None] * len(batch_features)
                    failed_slides.extend(batch_slide_names)
        
        # Save results for each slide in batch (skip failed slides)
        for idx, i in enumerate(batch_indices):
            aggregated_features = aggregated_batch[idx]
            slide_name = batch_slide_names[idx]
            
            if aggregated_features is None:
                logger.warning(f"Skipping save for failed slide: {slide_name}")
                continue
            
            try:
                # Save to HDF5 if requested
                if output_h5_paths and output_h5_paths[i] is not None:
                    asset_dict = {"features": aggregated_features}
                    save_hdf5(output_h5_paths[i], asset_dict, attr_h5_path=None, mode="w")
                
                # Save to PyTorch if requested
                if output_pt_paths and output_pt_paths[i] is not None:
                    features_tensor = torch.from_numpy(aggregated_features)
                    from .file import save_torch_tensor
                    save_torch_tensor(output_pt_paths[i], features_tensor)
            except Exception as e:
                logger.error(f"Failed to save results for slide {slide_name}: {str(e)}")
                if slide_name in successful_slides:
                    successful_slides.remove(slide_name)
                if slide_name not in failed_slides:
                    failed_slides.append(slide_name)
    
    # Clean up model to free GPU/CPU memory
    del model
    del model_fun
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Model cleaned up, memory freed")
    
    if failed_slides:
        logger.warning(f"\n=== Batch Processing Summary ===")
        logger.warning(f"Successfully processed: {len(successful_slides)}/{num_slides} slides")
        logger.warning(f"Failed: {len(failed_slides)} slides")
        logger.warning(f"Failed slides: {', '.join(failed_slides)}")
    else:
        logger.info(f"\n=== All {num_slides} slides processed successfully ===")
    
    logger.info(f"Batch aggregation complete")
    return output_h5_paths, output_pt_paths


@timed
def aggregate_slide_features(
    patch_features_h5_path,
    output_h5_path=None,
    output_pt_path=None,
    aggregation_method="identity",
    model_type=None,
    model_path=None,
    use_gpu=True,
    gpu_device_id=None,
    gpu_device_ids=None,
):
    """Aggregate patch-level features to slide-level (Step 2: Slide Encoding).

    This function takes patch-level features and aggregates them into slide-level
    representations. Supports simple aggregation methods (mean, max) or model-based
    aggregation using a slide encoder model (e.g., Prov-GigaPath slide encoder).

    Args:
        patch_features_h5_path: Path to HDF5 file with patch-level features.
        output_h5_path: Optional path to save slide-level features in HDF5 format.
        output_pt_path: Optional path to save slide-level features in PyTorch format.
        aggregation_method: Method for aggregating features (default: "identity").
            - "identity": No aggregation, keeps all patch features (backward compatible)
            - "mean": Mean pooling across patches
            - "max": Max pooling across patches
            - "model": Use a slide encoder model for aggregation
        model_type: Type of slide encoder model to use (only when aggregation_method="model").
        model_path: Optional path to slide encoder model weights.
        use_gpu: Whether to use GPU for model-based aggregation (default: True).
        gpu_device_id: GPU device ID to use.
        gpu_device_ids: List of GPU device IDs for multi-GPU.

    Returns:
        Tuple of (output_h5_path, output_pt_path) if saving, otherwise features tensor.
    """
    logger.info("Step 2: Aggregating patch features to slide level")

    with h5py.File(patch_features_h5_path, "r") as file:
        features = file["features"][:]
        logger.info(f"Loaded patch features with shape: {features.shape}")

        # Load coordinates if available
        coords = file["coords"][:] if "coords" in file else None

        # Load patch_size from coords attributes if available
        patch_size = None
        if "coords" in file and "patch_size" in file["coords"].attrs:
            patch_size = file["coords"].attrs["patch_size"]
            logger.info(f"Loaded patch_size from h5 file: {patch_size}")

        # Apply aggregation using shared helper function
        aggregated_features = _apply_slide_aggregation(
            features,
            aggregation_method=aggregation_method,
            slide_model_type=model_type,
            slide_model_path=model_path,
            use_gpu=use_gpu,
            gpu_device_id=gpu_device_id,
            gpu_device_ids=gpu_device_ids,
            coords=coords,
            patch_size=patch_size,
        )

        # Save to HDF5 if requested
        if output_h5_path is not None:
            logger.info(f"Saving aggregated features to {output_h5_path}")
            asset_dict = {"features": aggregated_features}

            # Copy coordinates if they exist and we're using identity
            if aggregation_method == "identity" and "coords" in file:
                asset_dict["coords"] = file["coords"][:]

            save_hdf5(output_h5_path, asset_dict, attr_h5_path=None, mode="w")

        # Save to PyTorch if requested
        if output_pt_path is not None:
            logger.info(f"Saving aggregated features to {output_pt_path}")
            features_tensor = torch.from_numpy(aggregated_features)
            from .file import save_torch_tensor
            save_torch_tensor(output_pt_path, features_tensor)

    return output_h5_path, output_pt_path


@timed
def save_features(
    patch_h5_path,
    slide_path,
    output_h5_path,
    output_pt_path=None,
    model_type=ModelType.CLIP,
    model_path=None,
    model_save_path=None,
    patch_path=None,
    batch_size=64,
    use_gpu=True,
    gpu_device_id=None,
    gpu_device_ids=None,
    num_workers=16,
    pin_memory=True,
    is_test_run=False,
    intermediate_h5_path=None,
    aggregation_method="identity",
    slide_model_type=None,
    slide_model_path=None,
):
    """Extract features from whole slide image and save to HDF5 and PyTorch formats.

    This function can operate in two modes (automatically determined by aggregation_method):
    1. Single-step mode (aggregation_method="identity"): Direct feature extraction (default, backward compatible)
    2. Two-step mode (aggregation_method in ["mean", "max", "model"]): Separate patch encoding and slide aggregation

    Args:
        patch_h5_path: Path to the h5 file containing patch coordinates.
        slide_path: Path to the whole slide image.
        output_h5_path: Path to save the extracted features in HDF5 format.
        output_pt_path: Optional path to save features in PyTorch format.
        model_type: Type of foundation model to use for patch encoding (default: ModelType.CLIP).
            When using model-based aggregation with a slide encoder, this will be automatically
            set to the required patch encoder if not already specified correctly.
        model_path: Optional path to patch encoder model weights.
        model_save_path: Optional path to save the model.
        patch_path: Optional path to folder with pre-extracted patches.
        batch_size: Batch size for feature extraction (default: 64).
        use_gpu: Whether to use GPU for inference (default: True).
        gpu_device_id: GPU device ID to use.
        gpu_device_ids: List of GPU device IDs for multi-GPU.
        num_workers: Number of worker processes for data loading (default: 16).
        pin_memory: Whether to pin memory for data loading (default: True).
        is_test_run: If True, only process first 3 batches (default: False).
        intermediate_h5_path: Path for intermediate patch features (two-step mode only).
        aggregation_method: Aggregation method (default: "identity").
            Options: "identity" (single-step), "mean" (two-step), "max" (two-step), "model" (two-step).
        slide_model_type: Type of slide encoder model (only when aggregation_method="model").
            The required patch encoder will be automatically inferred and used. For example,
            specifying GIGAPATH_SLIDE will automatically use GIGAPATH as the patch encoder.
        slide_model_path: Optional path to slide encoder model weights.
    """
    # Auto-set aggregation_method to "model" if slide_model_type is specified
    if slide_model_type is not None and aggregation_method == "identity":
        logger.info(
            f"Auto-setting aggregation_method to 'model' since slide_model_type "
            f"({slide_model_type}) is specified"
        )
        aggregation_method = "model"

    # Infer two-step mode from aggregation_method
    use_two_step = aggregation_method != "identity"

    if use_two_step:
        # Two-step process: patch encoding -> slide aggregation
        logger.info("Using two-step feature extraction process")

        # Auto-infer patch encoder from slide encoder if using model-based aggregation
        if aggregation_method == "model" and slide_model_type is not None:
            required_patch_encoder = get_required_patch_encoder(slide_model_type)
            if model_type != required_patch_encoder:
                logger.info(
                    f"Auto-selecting patch encoder {required_patch_encoder} "
                    f"as required by slide encoder {slide_model_type}"
                )
                model_type = required_patch_encoder
            # Validate compatibility
            validate_slide_encoder_compatibility(model_type, slide_model_type)

        # Determine intermediate path
        if intermediate_h5_path is None:
            intermediate_h5_path = str(Path(output_h5_path).with_suffix(".patch.h5"))

        # Step 1: Extract patch-level features
        extract_patch_features(
            patch_h5_path=patch_h5_path,
            slide_path=slide_path,
            output_h5_path=intermediate_h5_path,
            model_type=model_type,
            model_path=model_path,
            model_save_path=model_save_path,
            patch_path=patch_path,
            batch_size=batch_size,
            use_gpu=use_gpu,
            gpu_device_id=gpu_device_id,
            gpu_device_ids=gpu_device_ids,
            num_workers=num_workers,
            pin_memory=pin_memory,
            is_test_run=is_test_run,
        )

        # Step 2: Aggregate to slide level
        aggregate_slide_features(
            patch_features_h5_path=intermediate_h5_path,
            output_h5_path=output_h5_path,
            output_pt_path=output_pt_path,
            aggregation_method=aggregation_method,
            model_type=slide_model_type,
            model_path=slide_model_path,
            use_gpu=use_gpu,
            gpu_device_id=gpu_device_id,
            gpu_device_ids=gpu_device_ids,
        )
    else:
        # Single-step process (backward compatible)
        if gpu_device_ids:
            gpu_device_id = gpu_device_ids

        logger.info("loading model checkpoint")

        model_factory = get_model_factory(model_type)
        if model_factory is None:
            raise ValueError("model not recognized")
        model = model_factory.get_model(model_path, use_gpu, gpu_device_id)
        if model_save_path is not None:
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"saving model to {model_save_path}")
            model.save(model_save_path)
        preprocessing = model.get_preprocessing_fun()

        if patch_path:
            dataset = ImageFolder(
                root=patch_path,
                transform=preprocessing,
            )

            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=None,
                shuffle=False,
                persistent_workers=num_workers > 0,
                prefetch_factor=4 if num_workers > 0 else None,
            )
        elif patch_h5_path:
            dataset = WholeSlideImageH5Dataset(
                h5_path=patch_h5_path,
                slide_path=slide_path,
                preprocess=preprocessing,
                use_imagenet_rgb_dist=preprocessing is None,
                init_wsi_in_worker=num_workers > 0,
            )

            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_features,
                worker_init_fn=dataset.worker_init if num_workers > 0 else None,
                shuffle=False,
                persistent_workers=num_workers > 0,
                prefetch_factor=4 if num_workers > 0 else None,
            )
        else:
            raise ValueError("Either patch_path or patch_h5_path must be provided")

        process_dataset(
            dataset,
            loader,
            model_fun=model.get_model_fun(),
            patch_h5_path=patch_h5_path,
            output_h5_path=output_h5_path,
            is_test_run=is_test_run,
        )

        if output_pt_path is not None:
            with h5py.File(output_h5_path, "r") as file:
                features = file["features"][:]
                logger.info(f"features size: {features.shape} ")
                # logger.info(f'coordinates size: {file["coords"].shape} ')

                features = torch.from_numpy(features)
                from .file import save_torch_tensor
                save_torch_tensor(output_pt_path, features)


@timed
def filter_features(
    features: torch.Tensor,
    coords,
    classifier,
    threshold: float,
):
    """Filter features based on classifier predictions.

    Args:
        features: Feature tensor to filter.
        coords: Coordinate array corresponding to features.
        classifier: Classifier with predict_proba method.
        threshold: Probability threshold for filtering.

    Returns:
        Tuple of (filtered features, filtered coords).
    """
    logger.info("Predicting probabilities...")
    inclusion_mask = classifier.predict_proba(features)[:, 1] > threshold
    logger.info(f"{sum(inclusion_mask)} tiles above {threshold} threshold")
    return features[inclusion_mask], np.array(coords)[inclusion_mask]
