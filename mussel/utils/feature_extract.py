import gc
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from mussel.datasets import (
    WholeSlideImageH5Dataset,
    WholeSlideImageTileCoordDataset,
    FlatImageDataset,
)
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


# =============================================================================
# Feature Extraction Result and Dataset Processors
# =============================================================================


@dataclass
class FeatureExtractionResult:
    """Result of feature extraction from a dataset.

    Attributes:
        features: Extracted feature array of shape (N, D) where N is number of
            samples and D is feature dimension.
        labels: Optional label array of shape (N,) for classification datasets.
        coords: Optional coordinate array of shape (N, 2) for tile-based datasets.
        output_path: Optional path where features were saved to disk.
    """
    features: np.ndarray
    labels: Optional[np.ndarray] = None
    coords: Optional[np.ndarray] = None
    output_path: Optional[str] = None


class DatasetProcessor(ABC):
    """Abstract base class for processing different dataset types.

    This class defines the interface for feature extraction from various
    dataset types. Subclasses implement the specific processing logic for
    each dataset type (H5, ImageFolder, TileCoord, etc.).
    """

    @abstractmethod
    def process(
        self,
        dataset,
        loader: DataLoader,
        model_fun,
        output_h5_path: Optional[str] = None,
        patch_h5_path: Optional[str] = None,
        is_test_run: bool = False,
    ) -> FeatureExtractionResult:
        """Process a dataset and extract features.

        Args:
            dataset: The dataset to process.
            loader: DataLoader for the dataset.
            model_fun: Function to extract features from a batch of images.
            output_h5_path: Optional path to save features to HDF5.
            patch_h5_path: Optional path to source H5 for copying attributes.
            is_test_run: If True, only process first 3 batches.

        Returns:
            FeatureExtractionResult containing features and metadata.
        """
        pass


class TileCoordProcessor(DatasetProcessor):
    """Processor for WholeSlideImageTileCoordDataset.

    This processor extracts features from tile coordinates without saving
    to disk. It returns features and labels directly in memory.
    """

    def process(
        self,
        dataset,
        loader: DataLoader,
        model_fun,
        output_h5_path: Optional[str] = None,
        patch_h5_path: Optional[str] = None,
        is_test_run: bool = False,
    ) -> FeatureExtractionResult:
        """Process WholeSlideImageTileCoordDataset to extract features.

        Args:
            dataset: WholeSlideImageTileCoordDataset instance.
            loader: DataLoader for the dataset.
            model_fun: Function to extract features from a batch.
            output_h5_path: Unused (features returned in memory).
            patch_h5_path: Unused.
            is_test_run: If True, only process first 3 batches.

        Returns:
            FeatureExtractionResult with features and labels arrays.
        """
        all_features = []
        all_labels = []

        for count, (batch, labels) in enumerate(tqdm(loader, desc="Extracting features")):
            if is_test_run and count > 2:
                break

            features = model_fun(batch)
            all_features.append(features.numpy())
            all_labels.append(labels)

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        return FeatureExtractionResult(features=features, labels=labels)


class H5DatasetProcessor(DatasetProcessor):
    """Processor for WholeSlideImageH5Dataset.

    This processor extracts features from an H5-based dataset and saves
    them incrementally to an output H5 file.
    """

    def process(
        self,
        dataset,
        loader: DataLoader,
        model_fun,
        output_h5_path: Optional[str] = None,
        patch_h5_path: Optional[str] = None,
        is_test_run: bool = False,
    ) -> FeatureExtractionResult:
        """Process WholeSlideImageH5Dataset to extract features and save to HDF5.

        Args:
            dataset: WholeSlideImageH5Dataset instance.
            loader: DataLoader for the dataset.
            model_fun: Function to extract features from a batch.
            output_h5_path: Path to save extracted features (required).
            patch_h5_path: Path to source H5 for copying attributes.
            is_test_run: If True, only process first 3 batches.

        Returns:
            FeatureExtractionResult with features, coords, and output path.

        Raises:
            ValueError: If output_h5_path is not provided.
        """
        if output_h5_path is None:
            raise ValueError("output_h5_path is required for H5DatasetProcessor")

        mode = "w"
        all_features = []
        all_coords = []

        for count, (batch, coords) in enumerate(tqdm(loader, desc="Extracting features")):
            if is_test_run and count > 2:
                break

            # Skip empty batches (all tiles failed to load)
            if batch.numel() == 0:
                logger.warning(f"Skipping empty batch {count} (all tiles failed to load)")
                continue

            features = model_fun(batch).numpy()
            all_features.append(features)
            all_coords.append(coords)

            asset_dict = {"features": features, "coords": coords}
            save_hdf5(output_h5_path, asset_dict, attr_h5_path=patch_h5_path, mode=mode)
            mode = "a"

        # Concatenate all results
        features = np.concatenate(all_features, axis=0) if all_features else np.array([])
        coords = np.concatenate(all_coords, axis=0) if all_coords else np.array([])

        return FeatureExtractionResult(
            features=features,
            coords=coords,
            output_path=output_h5_path,
        )


class ImageFolderProcessor(DatasetProcessor):
    """Processor for ImageFolder and FlatImageDataset.

    This processor extracts features from image folder datasets and saves
    them to an output H5 file along with image paths and class labels.
    """

    def process(
        self,
        dataset,
        loader: DataLoader,
        model_fun,
        output_h5_path: Optional[str] = None,
        patch_h5_path: Optional[str] = None,
        is_test_run: bool = False,
    ) -> FeatureExtractionResult:
        """Process ImageFolder or FlatImageDataset to extract features.

        Args:
            dataset: ImageFolder or FlatImageDataset instance.
            loader: DataLoader for the dataset.
            model_fun: Function to extract features from a batch.
            output_h5_path: Path to save extracted features (required).
            patch_h5_path: Unused.
            is_test_run: If True, only process first 3 batches.

        Returns:
            FeatureExtractionResult with features, labels, and output path.

        Raises:
            ValueError: If output_h5_path is not provided.
        """
        if output_h5_path is None:
            raise ValueError("output_h5_path is required for ImageFolderProcessor")

        # Save metadata first based on dataset type
        if hasattr(dataset, 'imgs'):
            # ImageFolder dataset
            asset_dict = {
                "image_paths": np.array([x[0] for x in dataset.imgs]).astype("S"),
                "class_to_idx": np.array(
                    [np.asarray([k, v], dtype="S") for k, v in dataset.class_to_idx.items()]
                ),
            }
        elif hasattr(dataset, 'samples'):
            # FlatImageDataset
            asset_dict = {
                "image_paths": np.array([str(x) for x in dataset.samples]).astype("S"),
            }
        else:
            asset_dict = {}

        save_hdf5(output_h5_path, asset_dict, attr_h5_path=None, mode="w")

        all_features = []
        all_labels = []

        for count, (batch, labels) in enumerate(tqdm(loader, desc="Extracting features")):
            if is_test_run and count > 2:
                break

            labels_np = labels.numpy()
            features = model_fun(batch).numpy()

            all_features.append(features)
            all_labels.append(labels_np)

            asset_dict = {"features": features, "class": labels_np}
            save_hdf5(output_h5_path, asset_dict, attr_h5_path=None, mode="a")

        # Concatenate all results
        features = np.concatenate(all_features, axis=0) if all_features else np.array([])
        labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])

        return FeatureExtractionResult(
            features=features,
            labels=labels,
            output_path=output_h5_path,
        )


def get_dataset_processor(dataset) -> DatasetProcessor:
    """Get the appropriate processor for a dataset type.

    This factory function returns the correct DatasetProcessor subclass
    based on the type of dataset provided. Uses isinstance checks first,
    then falls back to class name matching for compatibility with mocked
    or dynamically-created dataset types.

    Args:
        dataset: The dataset instance to process.

    Returns:
        DatasetProcessor instance appropriate for the dataset type.

    Raises:
        ValueError: If dataset type is not supported.
    """
    # Map class names to processors as a fallback for when isinstance fails
    # (e.g., when classes are mocked in tests)
    _name_to_processor = {
        "WholeSlideImageTileCoordDataset": TileCoordProcessor,
        "WholeSlideImageH5Dataset": H5DatasetProcessor,
        "ImageFolder": ImageFolderProcessor,
        "FlatImageDataset": ImageFolderProcessor,
    }

    try:
        if isinstance(dataset, WholeSlideImageTileCoordDataset):
            return TileCoordProcessor()
        elif isinstance(dataset, WholeSlideImageH5Dataset):
            return H5DatasetProcessor()
        elif isinstance(dataset, (ImageFolder, FlatImageDataset)):
            return ImageFolderProcessor()
    except TypeError:
        # isinstance can fail if the class was replaced by a mock
        pass

    # Fallback: match by class name
    class_name = type(dataset).__name__
    if class_name in _name_to_processor:
        return _name_to_processor[class_name]()

    raise ValueError(
        f"Unsupported dataset type: {class_name}. "
        f"Supported types: WholeSlideImageTileCoordDataset, "
        f"WholeSlideImageH5Dataset, ImageFolder, FlatImageDataset"
    )


def process_dataset(
    dataset,
    loader: DataLoader,
    model_fun,
    output_h5_path: Optional[str] = None,
    patch_h5_path: Optional[str] = None,
    is_test_run: bool = False,
) -> FeatureExtractionResult:
    """Process a dataset to extract features.

    This function automatically selects the appropriate processor based on
    the dataset type and runs feature extraction. It provides a unified
    interface for all supported dataset types.

    Args:
        dataset: Dataset to process. Supported types:
            - WholeSlideImageTileCoordDataset: Returns features in memory
            - WholeSlideImageH5Dataset: Saves to H5 file
            - ImageFolder: Saves to H5 file with class labels
            - FlatImageDataset: Saves to H5 file
        loader: DataLoader for the dataset.
        model_fun: Function to extract features from a batch of images.
        output_h5_path: Optional path to save features to HDF5.
            Required for H5Dataset, ImageFolder, and FlatImageDataset.
        patch_h5_path: Optional path to source H5 for copying attributes.
            Used by H5DatasetProcessor.
        is_test_run: If True, only process first 3 batches (default: False).

    Returns:
        FeatureExtractionResult containing:
            - features: Extracted feature array
            - labels: Label array (for TileCoord and ImageFolder datasets)
            - coords: Coordinate array (for H5Dataset)
            - output_path: Path to saved file (if applicable)

    Raises:
        ValueError: If dataset type is not supported or required parameters
            are missing.
    """
    processor = get_dataset_processor(dataset)
    return processor.process(
        dataset=dataset,
        loader=loader,
        model_fun=model_fun,
        output_h5_path=output_h5_path,
        patch_h5_path=patch_h5_path,
        is_test_run=is_test_run,
    )


# =============================================================================
# Model Path Resolution
# =============================================================================


def get_model_path_from_dir(
    model_dir: Optional[str], model_type: Optional[ModelType]
) -> Optional[str]:
    """Get model path from model_dir if available.

    Searches for pre-downloaded models in the specified directory. Supports
    multiple naming conventions and model formats.

    Args:
        model_dir: Directory containing pre-downloaded models (should be local).
        model_type: ModelType enum indicating which model to find.

    Returns:
        Path to model if found in model_dir, None otherwise.

    Search order:
        1. Special case for GIGAPATH_SLIDE: returns HF hub path
        2. Special case for CONCH1_5: prefers TITAN_SLIDE directory
        3. Directory named after model type (uppercase)
        4. Directory named after model type (lowercase)
        5. .pth file named after model type (uppercase)
        6. .pth file named after model type (lowercase)
    """
    if model_dir is None or model_type is None:
        return None

    # Special case for GIGAPATH_SLIDE: return "hf-hub:" format to trigger HF caching
    # The GigapathSlideEncoderModel only accepts hf-hub: format
    # Must check this BEFORE checking for directories to avoid returning directory paths
    if model_type == ModelType.GIGAPATH_SLIDE:
        # Always use the default HF hub path, which will be cached automatically
        return ModelType.GIGAPATH_SLIDE.path

    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        return None

    # Special case for CONCH1_5: prefer TITAN_SLIDE directory if available
    # CONCH can be extracted from TITAN model, avoiding pickle issues
    if model_type == ModelType.CONCH1_5:
        titan_dir = model_dir_path / "TITAN_SLIDE"
        if titan_dir.exists() and titan_dir.is_dir():
            logger.info(f"Using local model file: {model_type.name} -> {titan_dir}")
            return str(titan_dir)
        titan_dir_lower = model_dir_path / "titan_slide"
        if titan_dir_lower.exists() and titan_dir_lower.is_dir():
            logger.info(f"Using local model file: {model_type.name} -> {titan_dir_lower}")
            return str(titan_dir_lower)

    # Check for model directory named after the model type
    model_subdir = model_dir_path / model_type.name
    if model_subdir.exists() and model_subdir.is_dir():
        # For slide encoders, look for slide_encoder.pth inside the directory
        # If it doesn't exist but pytorch_model.bin does, return the directory path
        # The GigapathSlideEncoderModel can handle directories with pytorch_model.bin
        if model_type in [ModelType.GIGAPATH_SLIDE, ModelType.TITAN_SLIDE]:
            slide_encoder_pth = model_subdir / "slide_encoder.pth"
            if slide_encoder_pth.exists():
                logger.info(f"Using local model file: {model_type.name} -> {slide_encoder_pth}")
                return str(slide_encoder_pth)
            # Check if directory has pytorch_model.bin (HuggingFace cache format)
            pytorch_model_bin = model_subdir / "pytorch_model.bin"
            if pytorch_model_bin.exists():
                logger.info(f"Using local model directory: {model_type.name} -> {model_subdir}")
                return str(model_subdir)
        logger.info(f"Using local model file: {model_type.name} -> {model_subdir}")
        return str(model_subdir)

    # Check for model directory named with lowercase
    model_subdir_lower = model_dir_path / model_type.name.lower()
    if model_subdir_lower.exists() and model_subdir_lower.is_dir():
        # For slide encoders, look for slide_encoder.pth inside the directory
        # If it doesn't exist but pytorch_model.bin does, return the directory path
        # The GigapathSlideEncoderModel can handle directories with pytorch_model.bin
        if model_type in [ModelType.GIGAPATH_SLIDE, ModelType.TITAN_SLIDE]:
            slide_encoder_pth = model_subdir_lower / "slide_encoder.pth"
            if slide_encoder_pth.exists():
                logger.info(f"Using local model file: {model_type.name} -> {slide_encoder_pth}")
                return str(slide_encoder_pth)
            # Check if directory has pytorch_model.bin (HuggingFace cache format)
            pytorch_model_bin = model_subdir_lower / "pytorch_model.bin"
            if pytorch_model_bin.exists():
                logger.info(f"Using local model directory: {model_type.name} -> {model_subdir_lower}")
                return str(model_subdir_lower)
        logger.info(f"Using local model file: {model_type.name} -> {model_subdir_lower}")
        return str(model_subdir_lower)

    # Check for .pth file (pickled models) - uppercase
    model_file_pth = model_dir_path / f"{model_type.name}.pth"
    if model_file_pth.exists() and model_file_pth.is_file():
        logger.info(f"Using local model file: {model_type.name} -> {model_file_pth}")
        return str(model_file_pth)

    # Check for .pth file (pickled models) - lowercase
    model_file_pth_lower = model_dir_path / f"{model_type.name.lower()}.pth"
    if model_file_pth_lower.exists() and model_file_pth_lower.is_file():
        logger.info(f"Using local model file: {model_type.name} -> {model_file_pth_lower}")
        return str(model_file_pth_lower)

    return None


def get_classifier_pkl_from_model_dir(
    model_dir: Optional[str], classifier_pkl: Optional[str]
) -> Optional[str]:
    """Get classifier pkl path from model_dir if available.
    
    If classifier_pkl is provided directly, returns it. Otherwise, looks for
    classifier.pkl in the model_dir directory.
    
    Args:
        model_dir: Directory containing pre-downloaded models and classifiers.
        classifier_pkl: Direct path to classifier pkl file.
    
    Returns:
        Path to classifier pkl if found, None otherwise.
    """
    if classifier_pkl:
        return classifier_pkl

    if not model_dir:
        return None

    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        return None

    # Check for classifier.pkl in model_dir
    classifier_file = model_dir_path / "classifier.pkl"
    if classifier_file.exists() and classifier_file.is_file():
        logger.info(f"✓ Using local classifier file: {classifier_file}")
        return str(classifier_file)

    return None


def get_batch_size_for_model(cfg, model_type) -> int:
    """Get the appropriate batch size for a given model type.
    
    Looks up model-specific batch sizes from config, falling back to default.
    
    Args:
        cfg: Configuration object with batch_size and model_batch_sizes attributes.
        model_type: ModelType enum or string name.
    
    Returns:
        Batch size to use for this model (from model_batch_sizes if defined,
        else default batch_size).
    
    Examples:
        >>> cfg.batch_size = 64
        >>> cfg.model_batch_sizes = {"VIRCHOW": 32}
        >>> get_batch_size_for_model(cfg, ModelType.VIRCHOW)
        32
        >>> get_batch_size_for_model(cfg, ModelType.CLIP)
        64
    """
    # Get model name
    if hasattr(model_type, "name"):
        model_name = model_type.name
    else:
        model_name = str(model_type)

    # Return per-model batch size if defined, else default
    return cfg.model_batch_sizes.get(model_name, cfg.batch_size)


def resolve_aggregation_method(
    aggregation_method: str, slide_model_type: Optional
) -> str:
    """Auto-set aggregation method based on slide model type.
    
    If slide_model_type is specified and aggregation_method is "identity",
    automatically switches to "model" aggregation.
    
    Args:
        aggregation_method: Current aggregation method setting
        slide_model_type: Optional slide encoder model type
    
    Returns:
        Resolved aggregation method (may be changed to "model")
    
    Examples:
        >>> resolve_aggregation_method("identity", ModelType.GIGAPATH_SLIDE)
        'model'
        >>> resolve_aggregation_method("mean", ModelType.GIGAPATH_SLIDE)
        'mean'
        >>> resolve_aggregation_method("identity", None)
        'identity'
    """
    if slide_model_type is not None and aggregation_method == "identity":
        logger.info(
            f"Auto-setting aggregation_method to 'model' since "
            f"slide_model_type={slide_model_type}"
        )
        return "model"
    return aggregation_method


def resolve_patch_encoder(
    model_type: Optional, slide_model_type: Optional
):
    """Auto-infer patch encoder from slide model type if not specified.
    
    If model_type is None and slide_model_type is provided, automatically
    determines the required patch encoder for that slide model.
    
    Args:
        model_type: Optional explicit patch encoder model type
        slide_model_type: Optional slide encoder model type
    
    Returns:
        Resolved model type (may be auto-inferred from slide model)
    
    Examples:
        >>> resolve_patch_encoder(None, ModelType.GIGAPATH_SLIDE)
        ModelType.GIGAPATH
        >>> resolve_patch_encoder(ModelType.CLIP, ModelType.GIGAPATH_SLIDE)
        ModelType.CLIP
    """
    if model_type is None and slide_model_type is not None:
        from mussel.models import get_required_patch_encoder
        model_type = get_required_patch_encoder(slide_model_type)
        logger.info(
            f"Auto-inferring model_type={model_type.name} "
            f"from slide_model_type={slide_model_type.name}"
        )
    return model_type


def _apply_slide_aggregation(
    features: np.ndarray,
    aggregation_method: str = "identity",
    slide_model_type: Optional[ModelType] = None,
    slide_model_path: Optional[str] = None,
    use_gpu: bool = True,
    gpu_device_id: Optional[Union[int, List[int]]] = None,
    gpu_device_ids: Optional[List[int]] = None,
    coords: Optional[np.ndarray] = None,
    patch_size: Optional[int] = None,
) -> np.ndarray:
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
    coords: np.ndarray,
    slide_path: str,
    attrs: dict,
    model_type: ModelType = ModelType.CLIP,
    model_path: Optional[str] = None,
    batch_size: int = 64,
    use_gpu: bool = True,
    gpu_device_id: Optional[Union[int, List[int]]] = None,
    gpu_device_ids: Optional[List[int]] = None,
    pin_memory: bool = True,
    num_workers: int = 16,
    is_test_run: bool = False,
    use_slide_encoder: bool = False,
    slide_model_type: Optional[ModelType] = None,
    slide_model_path: Optional[str] = None,
    aggregation_method: str = "identity",
) -> tuple[np.ndarray, np.ndarray]:
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

    result = process_dataset(
        dataset, loader, model_fun=model.get_model_fun(), is_test_run=is_test_run
    )
    features, labels = result.features, result.labels

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
    patch_h5_path: str,
    slide_path: str,
    output_h5_path: str,
    model_type: ModelType = ModelType.CLIP,
    model_path: Optional[str] = None,
    model_save_path: Optional[str] = None,
    patch_path: Optional[str] = None,
    batch_size: int = 64,
    use_gpu: bool = True,
    gpu_device_id: Optional[Union[int, List[int]]] = None,
    gpu_device_ids: Optional[List[int]] = None,
    num_workers: int = 16,
    pin_memory: bool = True,
    is_test_run: bool = False,
) -> str:
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
        # Try ImageFolder first (for class-based structure)
        # Fall back to FlatImageDataset if no class folders found
        try:
            dataset = ImageFolder(
                root=patch_path,
                transform=preprocessing,
            )
        except FileNotFoundError:
            # No class folders found, use flat directory structure
            logger.info(f"No class folders found in {patch_path}, using flat directory structure")
            dataset = FlatImageDataset(
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
    
    # If still None, use the default HF hub path from ModelType
    if model_path is None:
        model_path = model_type.path
        logger.info(f"Using default HuggingFace path for {model_type.name}: {model_path}")
    
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
                # Remove from successful list if present
                try:
                    successful_slides.remove(slide_name)
                except ValueError:
                    pass  # Slide wasn't in successful list
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
    patch_features_h5_path: str,
    output_h5_path: Optional[str] = None,
    output_pt_path: Optional[str] = None,
    aggregation_method: str = "identity",
    model_type: Optional[ModelType] = None,
    model_path: Optional[str] = None,
    use_gpu: bool = True,
    gpu_device_id: Optional[Union[int, List[int]]] = None,
    gpu_device_ids: Optional[List[int]] = None,
) -> Union[tuple[Optional[str], Optional[str]], np.ndarray]:
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
    patch_h5_path: str,
    slide_path: str,
    output_h5_path: str,
    output_pt_path: Optional[str] = None,
    model_type: ModelType = ModelType.CLIP,
    model_path: Optional[str] = None,
    model_save_path: Optional[str] = None,
    patch_path: Optional[str] = None,
    batch_size: int = 64,
    use_gpu: bool = True,
    gpu_device_id: Optional[Union[int, List[int]]] = None,
    gpu_device_ids: Optional[List[int]] = None,
    num_workers: int = 16,
    pin_memory: bool = True,
    is_test_run: bool = False,
    intermediate_h5_path: Optional[str] = None,
    aggregation_method: str = "identity",
    slide_model_type: Optional[ModelType] = None,
    slide_model_path: Optional[str] = None,
) -> tuple[str, Optional[str]]:
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
            # Try ImageFolder first (for class-based structure)
            # Fall back to FlatImageDataset if no class folders found
            try:
                dataset = ImageFolder(
                    root=patch_path,
                    transform=preprocessing,
                )
            except FileNotFoundError:
                # No class folders found, use flat directory structure
                logger.info(f"No class folders found in {patch_path}, using flat directory structure")
                dataset = FlatImageDataset(
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


def subsample_tiles(
    features: np.ndarray,
    coords: np.ndarray,
    max_tiles: int,
    strategy: str,
    slide_sizes: List[int],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample tile features and coordinates to at most *max_tiles* rows.

    Args:
        features: Array of shape ``(N, D)`` with all concatenated tile features.
        coords:   Array of shape ``(N, 2)`` with corresponding tile coordinates.
        max_tiles: Maximum number of tiles to keep.
        strategy: One of ``"random"``, ``"proportional"``, or ``"equal"``.
            - ``"random"``: uniformly sample *max_tiles* from the full pool.
            - ``"proportional"``: sample from each slide in proportion to its size.
            - ``"equal"``: sample an equal number of tiles from each slide.
        slide_sizes: List of per-slide tile counts ``[N_1, N_2, ...]`` whose sum
            equals ``len(features)``.  Required for ``"proportional"`` and
            ``"equal"`` strategies; ignored by ``"random"``.
        seed: Random seed for reproducibility.

    Returns:
        ``(features_sub, coords_sub)`` with at most *max_tiles* rows.
    """
    n_total = len(features)
    if n_total <= max_tiles:
        return features, coords

    rng = np.random.default_rng(seed)

    if strategy == "random":
        idx = rng.choice(n_total, size=max_tiles, replace=False)
        idx.sort()

    elif strategy == "proportional":
        idx_parts = []
        start = 0
        remaining = max_tiles
        for i, size in enumerate(slide_sizes):
            if i == len(slide_sizes) - 1:
                n_i = remaining
            else:
                n_i = round(max_tiles * size / n_total)
                n_i = min(n_i, size)
                remaining -= n_i
            slide_idx = np.arange(start, start + size)
            chosen = rng.choice(slide_idx, size=n_i, replace=False)
            idx_parts.append(chosen)
            start += size
        idx = np.sort(np.concatenate(idx_parts))

    elif strategy == "equal":
        n_slides = len(slide_sizes)
        base = max_tiles // n_slides
        idx_parts = []
        start = 0
        for i, size in enumerate(slide_sizes):
            n_i = base + (max_tiles % n_slides if i == n_slides - 1 else 0)
            n_i = min(n_i, size)
            slide_idx = np.arange(start, start + size)
            chosen = rng.choice(slide_idx, size=n_i, replace=False)
            idx_parts.append(chosen)
            start += size
        idx = np.sort(np.concatenate(idx_parts))

    else:
        raise ValueError(
            f"Unknown subsampling strategy {strategy!r}. "
            "Choose from 'random', 'proportional', or 'equal'."
        )

    return features[idx], coords[idx]


def aggregate_sample_features(
    patch_features_h5_paths: List[str],
    sample_ids: List[str],
    output_dir: str,
    output_h5_suffix: str = "features.h5",
    max_tiles: Optional[int] = None,
    subsampling_strategy: str = "random",
    seed: int = 42,
) -> None:
    """Concatenate per-slide patch features into one H5 per sample.

    Reads per-slide feature H5 files (each with ``features`` (N_i, D) and
    ``coords`` (N_i, 2) datasets), groups them by ``sample_id``, concatenates
    on the tile axis, optionally subsamples to ``max_tiles``, and writes one
    output H5 per unique sample.

    Args:
        patch_features_h5_paths: Paths to per-slide feature H5 files
            (produced by ``extract_features``).
        sample_ids: Sample identifier for each slide (same length as
            ``patch_features_h5_paths``).  Slides with the same ``sample_id``
            are concatenated together.
        output_dir: Directory where one ``{sample_id}.{output_h5_suffix}`` file
            is written per unique sample.
        output_h5_suffix: Filename suffix for output files (default
            ``"features.h5"``).
        max_tiles: If set, subsample each sample to at most this many tiles
            after concatenation.  ``None`` keeps all tiles.
        subsampling_strategy: Strategy when subsampling — ``"random"``,
            ``"proportional"``, or ``"equal"``.  Ignored when ``max_tiles``
            is ``None`` or total tiles ≤ ``max_tiles``.
        seed: Random seed for subsampling reproducibility (default ``42``).
    """
    import collections
    import os

    if len(patch_features_h5_paths) != len(sample_ids):
        raise ValueError(
            f"patch_features_h5_paths ({len(patch_features_h5_paths)}) and "
            f"sample_ids ({len(sample_ids)}) must have the same length."
        )

    os.makedirs(output_dir, exist_ok=True)

    groups: dict = collections.OrderedDict()
    for idx, sid in enumerate(sample_ids):
        groups.setdefault(sid, []).append(idx)

    for sample_id, indices in groups.items():
        logger.info("Aggregating sample %s from %d slide(s)", sample_id, len(indices))

        all_features = []
        all_coords = []
        slide_sizes = []

        for i in indices:
            h5_path = patch_features_h5_paths[i]
            with h5py.File(h5_path, "r") as h5:
                feats = np.array(h5["features"])
                coords = h5["coords"][:]
            all_features.append(feats)
            all_coords.append(coords)
            slide_sizes.append(len(feats))
            logger.debug("  slide %d: %d tiles from %s", i, len(feats), h5_path)

        features = np.concatenate(all_features, axis=0)
        coords = np.concatenate(all_coords, axis=0)

        if max_tiles is not None:
            features, coords = subsample_tiles(
                features,
                coords,
                max_tiles=max_tiles,
                strategy=subsampling_strategy,
                slide_sizes=slide_sizes,
                seed=seed,
            )
            logger.info(
                "  subsampled to %d tiles (strategy=%s, max_tiles=%d)",
                len(features),
                subsampling_strategy,
                max_tiles,
            )

        out_path = os.path.join(output_dir, f"{sample_id}.{output_h5_suffix}")
        save_hdf5(out_path, {"features": features, "coords": coords}, mode="w")
        logger.info("Wrote %s (%d tiles, dim=%d)", out_path, len(features), features.shape[1])


@timed
def filter_features(
    features: torch.Tensor,
    coords: np.ndarray,
    classifier: Any,
    threshold: float,
) -> tuple[torch.Tensor, np.ndarray]:
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
