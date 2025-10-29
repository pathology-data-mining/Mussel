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
from mussel.models import ModelType, get_model_factory, validate_slide_encoder_compatibility, get_required_patch_encoder

from .file import save_hdf5
from .ml import collate_features
from .timer import timed

logger = logging.getLogger(__name__)


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
        logger.info(f"Applied mean pooling: {features.shape} -> {aggregated_features.shape}")
        return aggregated_features
    elif aggregation_method == "max":
        # Max pooling across patches
        aggregated_features = np.max(features, axis=0, keepdims=True)
        logger.info(f"Applied max pooling: {features.shape} -> {aggregated_features.shape}")
        return aggregated_features
    elif aggregation_method == "model":
        # Model-based aggregation using a slide encoder
        if slide_model_type is None:
            raise ValueError("slide_model_type must be provided when using model-based aggregation")
        
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
            if slide_model_type == ModelType.TITAN_SLIDE:
                # TITAN requires features, coords, and patch_size
                if coords is None:
                    raise ValueError("TITAN_SLIDE requires coordinates")
                if patch_size is None:
                    # Default patch size at level 0 (commonly 256 for TITAN/CONCH)
                    patch_size = 256
                    logger.warning(f"patch_size not provided, using default: {patch_size}")
                coords_tensor = torch.from_numpy(coords).unsqueeze(0)  # Add batch dimension
                aggregated_features = model_fun(features_tensor, coords_tensor, patch_size).cpu().numpy()
            elif slide_model_type == ModelType.GIGAPATH_SLIDE:
                # GIGAPATH requires features and coords
                if coords is None:
                    raise ValueError("GIGAPATH_SLIDE requires coordinates")
                coords_tensor = torch.from_numpy(coords).unsqueeze(0)  # Add batch dimension
                aggregated_features = model_fun(features_tensor, coords_tensor).cpu().numpy()
            else:
                # Other slide encoders may only need features
                aggregated_features = model_fun(features_tensor).cpu().numpy()
        
        logger.info(f"Applied model aggregation: {features.shape} -> {aggregated_features.shape}")
        return aggregated_features
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")


def get_features(
    coords,
    slide_path,
    attrs,
    model_type=ModelType.CLIP,
    model_path=None,
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
    logger.info("loading model checkpoint")

    if gpu_device_ids:
        gpu_device_id = gpu_device_ids

    # Auto-set aggregation_method to "model" if slide_model_type is specified
    if use_slide_encoder and slide_model_type is not None and aggregation_method != "model":
        logger.info(
            f"Auto-setting aggregation_method to 'model' since slide_model_type "
            f"({slide_model_type}) is specified"
        )
        aggregation_method = "model"

    # Auto-infer patch encoder from slide encoder if using model-based aggregation
    if use_slide_encoder and aggregation_method == "model" and slide_model_type is not None:
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
            torch.save(features_tensor, output_pt_path)
    
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
    use_two_step=False,
    intermediate_h5_path=None,
    aggregation_method="identity",
    slide_model_type=None,
    slide_model_path=None,
):
    """Extract features from whole slide image and save to HDF5 and PyTorch formats.

    This function can operate in two modes:
    1. Single-step mode (use_two_step=False): Direct feature extraction (default, backward compatible)
    2. Two-step mode (use_two_step=True): Separate patch encoding and slide aggregation

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
        use_two_step: If True, use two-step process (patch encoding + aggregation).
        intermediate_h5_path: Path for intermediate patch features (two-step mode only).
        aggregation_method: Aggregation method for two-step mode (default: "identity").
            Options: "identity", "mean", "max", "model".
        slide_model_type: Type of slide encoder model (only when aggregation_method="model").
            The required patch encoder will be automatically inferred and used. For example,
            specifying GIGAPATH_SLIDE will automatically use GIGAPATH as the patch encoder.
        slide_model_path: Optional path to slide encoder model weights.
    """
    if use_two_step:
        # Two-step process: patch encoding -> slide aggregation
        logger.info("Using two-step feature extraction process")
        
        # Auto-set aggregation_method to "model" if slide_model_type is specified
        if slide_model_type is not None and aggregation_method != "model":
            logger.info(
                f"Auto-setting aggregation_method to 'model' since slide_model_type "
                f"({slide_model_type}) is specified"
            )
            aggregation_method = "model"
        
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
                torch.save(features, output_pt_path)


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
