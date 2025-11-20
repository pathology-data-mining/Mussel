import json
import logging
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable, List

import open_clip
import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torchvision import transforms
from transformers import AutoModel

# Import model cache utilities for file locking
try:
    from mussel.utils.model_cache import model_download_lock
except ImportError:
    # Fallback if module not available
    from contextlib import contextmanager
    @contextmanager
    def model_download_lock(model_name, **kwargs):
        yield True  # No locking, always download

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGENET_STD = [0.229, 0.224, 0.225]

# Enable TF32 for Ampere+ GPUs for faster matmul operations
# This provides significant speedups on A100, A10, etc.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable optimized SDPA (Scaled Dot Product Attention) for transformer models
# This uses Flash Attention 2 if available, otherwise falls back to xformers or efficient attention
# All timm models (UNI, UNI2, Virchow, Virchow2, OPTIMUS) automatically use this
# HuggingFace models (CONCH1.5) use attn_implementation parameter for flash attention
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


def get_best_attn_implementation():
    """Determine the best attention implementation available.
    
    Returns:
        str: One of "flash_attention_2", "sdpa", or "eager"
        
    Priority:
        1. flash_attention_2 - Fastest, requires flash-attn package and CUDA >= 8.0
        2. sdpa - PyTorch 2.0+ built-in, uses xformers if available
        3. eager - Fallback to standard PyTorch attention
        
    Note:
        - timm models automatically use SDPA when available
        - This function is for HuggingFace transformers models (e.g., CONCH1.5)
    """
    # Check for flash_attn package
    try:
        import flash_attn
        # Check CUDA capability (flash_attn requires compute capability >= 8.0)
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            logger.info("Using flash_attention_2 for HuggingFace models")
            return "flash_attention_2"
    except (ImportError, AttributeError):
        pass
    
    # SDPA is available in PyTorch 2.0+ and will use xformers if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        logger.info("Using sdpa (scaled_dot_product_attention) for HuggingFace models")
        return "sdpa"
    
    # Fallback to eager
    logger.warning("Using eager attention (no acceleration available)")
    return "eager"


class ModelType(Enum):
    def __init__(self, id, code, path):
        """Initialize a ModelType enum value.

        Args:
            id: Unique integer identifier for the model type.
            code: String code for the model type.
            path: Path or identifier to load the model from.
        """
        self.id = id
        self.code = code
        self.path = path

    RESNET50 = 1, "resnet50", ""
    CTRANSPATH = 2, "ctranspath", ""
    GIGAPATH = 3, "gigapath", "hf-hub:prov-gigapath/prov-gigapath"
    VIRCHOW = 4, "virchow", "hf-hub:paige-ai/Virchow"
    OPTIMUS = 5, "optimus", "hf-hub:bioptimus/H-optimus-0"
    CLIP = 6, "clip", "hf-hub:wisdomik/QuiltNet-B-16-PMB"
    GOOGLEPATH = 7, "googlepath", "google/path-foundation"
    CONCH1_5 = 8, "conch1_5", "MahmoodLab/TITAN"
    VIRCHOW2 = 9, "virchow2", "hf-hub:paige-ai/Virchow2"
    UNI2 = 10, "uni2h", "hf-hub:MahmoodLab/UNI2-h"
    UNI = 11, "uni", "hf-hub:MahmoodLab/UNI"
    GIGAPATH_SLIDE = 12, "gigapath_slide", "hf-hub:prov-gigapath/prov-gigapath"
    TITAN_SLIDE = 13, "titan_slide", "MahmoodLab/TITAN"


# Mapping of slide encoder models to their compatible patch encoder models
SLIDE_ENCODER_COMPATIBILITY = {
    ModelType.GIGAPATH_SLIDE: ModelType.GIGAPATH,
    ModelType.TITAN_SLIDE: ModelType.CONCH1_5,
    # Add more slide encoder -> patch encoder mappings as they become available
}


# Recommended patch sizes for each model type
MODEL_PATCH_SIZES = {
    ModelType.RESNET50: 256,
    ModelType.CTRANSPATH: 224,
    ModelType.GIGAPATH: 256,
    ModelType.VIRCHOW: 224,
    ModelType.VIRCHOW2: 224,
    ModelType.OPTIMUS: 224,
    ModelType.CLIP: 224,  # QuiltNet
    ModelType.GOOGLEPATH: 224,  # Google Path Foundation
    ModelType.CONCH1_5: 512,
    ModelType.UNI: 256,  # TRIDENT default patch size for UNI
    ModelType.UNI2: 256,  # TRIDENT default patch size for UNI2
    # Slide encoders inherit from their patch encoders
    ModelType.GIGAPATH_SLIDE: 256,
    ModelType.TITAN_SLIDE: 512,
}


def get_required_patch_encoder(slide_encoder: ModelType) -> ModelType:
    """Get the required patch encoder for a given slide encoder.

    Each slide encoder model is designed to work with features from a specific
    patch encoder model. This function returns the required patch encoder.

    Args:
        slide_encoder: The slide-level encoder model type.

    Returns:
        The required patch encoder model type.

    Raises:
        ValueError: If the slide encoder is not recognized.
    """
    if slide_encoder not in SLIDE_ENCODER_COMPATIBILITY:
        raise ValueError(
            f"Unknown slide encoder: {slide_encoder}. "
            f"Available slide encoders: {list(SLIDE_ENCODER_COMPATIBILITY.keys())}"
        )

    return SLIDE_ENCODER_COMPATIBILITY[slide_encoder]


def get_default_patch_size(model_type: ModelType) -> int:
    """Get the recommended default patch size for a model type.

    Args:
        model_type: The model type to get the patch size for.

    Returns:
        The recommended patch size in pixels.

    Raises:
        ValueError: If the model type is not recognized.
    """
    if model_type not in MODEL_PATCH_SIZES:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available model types: {list(MODEL_PATCH_SIZES.keys())}"
        )

    return MODEL_PATCH_SIZES[model_type]


def validate_slide_encoder_compatibility(
    patch_encoder: ModelType, slide_encoder: ModelType
) -> bool:
    """Validate that a slide encoder is compatible with a patch encoder.

    Each slide encoder model is designed to work with features from a specific
    patch encoder model. This function validates that the combination is valid.

    Args:
        patch_encoder: The patch-level encoder model type.
        slide_encoder: The slide-level encoder model type.

    Returns:
        True if the slide encoder is compatible with the patch encoder.

    Raises:
        ValueError: If the slide encoder is not compatible with the patch encoder.
    """
    if slide_encoder not in SLIDE_ENCODER_COMPATIBILITY:
        raise ValueError(
            f"Unknown slide encoder: {slide_encoder}. "
            f"Available slide encoders: {list(SLIDE_ENCODER_COMPATIBILITY.keys())}"
        )

    expected_patch_encoder = SLIDE_ENCODER_COMPATIBILITY[slide_encoder]
    if patch_encoder != expected_patch_encoder:
        raise ValueError(
            f"Slide encoder {slide_encoder} requires patch encoder {expected_patch_encoder}, "
            f"but {patch_encoder} was provided. Each slide encoder is tied to a specific "
            f"patch encoder model."
        )

    return True


class Model:
    def __init__(
        self,
        model_obj,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize base Model wrapper.

        Args:
            model_obj: The underlying model object.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        self.obj = model_obj

    def get_model_fun(self) -> Callable:
        """Get a callable function for model inference.

        Returns:
            Callable that takes input and returns model output.
        """
        return self.obj

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing function for input data.

        Returns:
            Callable for preprocessing or None if no preprocessing needed.
        """
        return None

    def save(self, save_path: str):
        """Save the model to disk.

        Args:
            save_path: Path to save the model.
        """
        pass


class GooglePathModel(Model):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize GooglePath model.

        Args:
            model_path: Path to the model or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        import tensorflow as tf
        from huggingface_hub import from_pretrained_keras

        if model_path is None:
            model_path = ModelType.GOOGLEPATH.path

        if use_gpu and len(tf.config.list_physical_devices("GPU")) == 0:
            raise OSError("cuda not available")
        if use_gpu and gpu_device_id:
            if isinstance(gpu_device_id, list):
                devices = [
                    tf.config.list_physical_devices("GPU")[i] for i in gpu_device_id
                ]
            else:
                devices = tf.config.list_physical_devices("GPU")[gpu_device_id]
            tf.config.set_visible_devices(devices)

        model_obj = from_pretrained_keras(model_path)

        super().__init__(model_obj)

    def get_model_fun(self) -> Callable:
        """Get model inference function for GooglePath.

        Returns:
            Callable that preprocesses input and returns model output.
        """
        import tensorflow as tf

        def model_fun(x) -> Callable:
            """Preprocess and run inference for GooglePath model."""
            tensor = tf.cast(x, tf.float32) / 255.0
            tensor = tf.transpose(tensor, [0, 2, 3, 1])
            tensor = tf.image.resize(
                tensor, size=(224, 224), method=tf.image.ResizeMethod.BICUBIC
            )
            return self.obj(tensor)

        return model_fun

    def save(self, save_path: str):
        """Save GooglePath model (not implemented).

        Args:
            save_path: Path to save the model.

        Raises:
            NotImplementedError: GooglePath model saving is not yet implemented.
        """
        raise NotImplementedError("GooglePath model saving not implemented yet")


class TorchModel(Model):
    def __init__(
        self,
        model_path,
        model_obj=None,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize PyTorch model wrapper.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            model_obj: Optional pre-loaded model object (default: None).
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        self.use_gpu = use_gpu
        if model_obj is None:
            if model_path.startswith("hf-hub:"):
                repo_id = model_path.replace("hf-hub:", "")
                # Use locking when downloading from HuggingFace
                with model_download_lock(repo_id) as should_download:
                    config_file = hf_hub_download(
                        repo_id=repo_id,
                        filename="config.json",
                    )
                    with open(config_file, "r") as f:
                        config = json.load(f)
                    model_name = config.get("model_name", None)
                    if not model_name:
                        raise ValueError(
                            f"model_name not found in config.json from {repo_id}"
                        )
                    model_obj = timm.create_model(model_name, pretrained=True)
            elif Path(model_path).is_file():
                with open(model_path, "rb") as f:
                    model_obj = pickle.load(f)
            else:
                raise ValueError(f"invalid model_path: {model_path}")
        super().__init__(model_obj)
        if use_gpu and not torch.cuda.is_available():
            raise OSError("cuda not available")

        device_type = "cuda" if use_gpu else "cpu"
        device_id = (
            gpu_device_id[0]
            if isinstance(gpu_device_id, list) and len(gpu_device_id) > 0
            else gpu_device_id
        )
        self.device = (
            torch.device(device_type, device_id)
            if device_id is not None
            else torch.device(device_type)
        )

        if isinstance(gpu_device_id, list) and len(gpu_device_id) > 1:
            self.obj = nn.DataParallel(self.obj, device_ids=gpu_device_id)
        
        # Apply performance optimizations
        if use_gpu:
            # Enable cuDNN benchmarking for optimized convolution algorithms
            torch.backends.cudnn.benchmark = True
            # Set matmul precision for faster operations on Ampere+ GPUs
            torch.set_float32_matmul_precision('high')
            # Convert to channels_last memory format for better GPU utilization
            try:
                self.obj = self.obj.to(memory_format=torch.channels_last)
            except:
                pass  # Some models may not support channels_last
        
        self.obj = self.obj.to(self.device)
        self.obj.eval()
        
        # Disable torch.compile for now due to compatibility issues
        # Will re-enable after testing individual models
        # if hasattr(torch, 'compile') and use_gpu:
        #     try:
        #         self.obj = torch.compile(self.obj, mode='max-autotune')
        #     except Exception as e:
        #         # Fall back if compilation fails
        #         import logging
        #         logging.getLogger(__name__).warning(f"Failed to compile model: {e}")

    def get_model_fun(self) -> Callable:
        """Get model inference function with automatic mixed precision.

        Returns:
            Callable that runs inference on GPU or CPU with autocast.
        """

        def model_fun(x):
            """Run inference with mixed precision."""
            with (
                torch.no_grad(),
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type, dtype=torch.float16),
            ):
                x = x.to(self.device, non_blocking=True)
                # Convert to channels_last for better GPU utilization if possible
                if self.use_gpu:
                    try:
                        x = x.to(memory_format=torch.channels_last)
                    except:
                        pass
                return self.obj(x).cpu()

        return model_fun

    def save(self, save_path: str):
        """Save PyTorch model to disk using pickle.

        Args:
            save_path: Path to save the model.
        """
        with open(save_path, "wb") as f:
            pickle.dump(self.obj, f)


class Conch15Model(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize CONCH v1.5 model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.CONCH1_5.path
        model_obj = None
        if not Path(model_path).is_file():
            attn_impl = get_best_attn_implementation()
            # Use locking when downloading from HuggingFace
            with model_download_lock(model_path) as should_download:
                try:
                    titan = AutoModel.from_pretrained(
                        model_path, 
                        trust_remote_code=True,
                        attn_implementation=attn_impl
                    )
                except (ValueError, NotImplementedError) as e:
                    # TITAN model doesn't support Flash Attention 2.0 or SDPA yet, fallback to eager
                    if "does not support an attention implementation" in str(e) or "does not support Flash Attention" in str(e):
                        logger.info("TITAN model doesn't support optimized attention, using eager mode")
                        titan = AutoModel.from_pretrained(
                            model_path, 
                            trust_remote_code=True,
                            attn_implementation="eager"
                        )
                    else:
                        raise
            model_obj, _ = titan.return_conch()
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for CONCH v1.5.

        Returns:
            Composed transforms for CONCH v1.5 input preprocessing.
        """
        preprocessing = transforms.Compose(
            [
                transforms.Resize(
                    448,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return preprocessing

    def save(self, save_path: str):
        """CONCH1_5 model saving is disabled.
        
        The CONCH model is extracted from TITAN and contains dynamic modules 
        that cannot be reliably pickled/unpickled. Use TITAN_SLIDE directory instead.
        
        Args:
            save_path: Path where model would be saved (ignored).
        
        Raises:
            NotImplementedError: Always raised to prevent saving.
        """
        raise NotImplementedError(
            "Saving CONCH1_5 model is not supported. "
            "CONCH is extracted from TITAN model and contains dynamic modules that "
            "cannot be reliably pickled. To cache models, save the TITAN_SLIDE directory instead."
        )


class TitanSlideEncoderModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize TITAN slide encoder model.

        This is the slide-level encoder from MahmoodLab/TITAN that aggregates
        CONCH patch-level features into slide-level representations.

        Args:
            model_path: Path to slide encoder model directory or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.TITAN_SLIDE.path
        
        # Validate that model_path is appropriate for TITAN slide encoder
        if Path(model_path).is_file():
            raise ValueError(
                f"TITAN_SLIDE requires a directory path or HuggingFace repo ID, not a file. "
                f"Got: {model_path}\n"
                f"Hint: Use slide_model_type=TITAN_SLIDE (not model_type) and "
                f"slide_model_path='path/to/TITAN_SLIDE/'"
            )
        
        model_obj = None
        # Load the TITAN model from HuggingFace or saved directory
        # This handles both MahmoodLab/TITAN and locally saved directories
        # TITAN doesn't support Flash Attention 2.0, so we use eager mode
        # Use locking when downloading from HuggingFace
        with model_download_lock(model_path) as should_download:
            try:
                model_obj = AutoModel.from_pretrained(
                    model_path, 
                    trust_remote_code=True,
                    attn_implementation="eager"
                )
            except TypeError:
                # Fallback for older transformers that don't support attn_implementation
                model_obj = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_model_fun(self) -> Callable:
        """Get model inference function for TITAN slide encoder.

        The TITAN slide encoder uses encode_slide_from_patch_features method
        which requires patch features, coordinates, and patch size at level 0.

        Returns:
            Callable that takes patch features, coords, and patch_size, returns slide-level features
            with shape (768,) matching the GIGAPATH slide encoder output format.
        """

        def model_fun(patch_features, coords, patch_size):
            """Run TITAN slide encoder on patch features with coordinates and patch size."""
            with (
                torch.no_grad(),
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type, dtype=torch.float16),
            ):
                patch_features = patch_features.to(self.device, non_blocking=True)
                coords = coords.to(self.device, non_blocking=True)
                return self.obj.encode_slide_from_patch_features(
                    patch_features, coords, patch_size
                ).squeeze().cpu()

        return model_fun

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing function for slide encoder.

        Slide encoders work on patch features, not images, so no preprocessing needed.

        Returns:
            None, as slide encoders don't preprocess images.
        """
        return None

    def save(self, save_path: str):
        """Save TITAN slide encoder model to disk.

        TITAN models loaded from HuggingFace cannot be pickled directly,
        so we save using HuggingFace's save_pretrained method to a directory.

        Args:
            save_path: Path to save the model (must be a directory, not a file).
            
        Raises:
            ValueError: If save_path has a file extension (like .pth or .pkl).
        """
        # Check if save_path looks like a file (has extension)
        if Path(save_path).suffix:
            raise ValueError(
                f"TITAN_SLIDE model cannot be saved to a file ({save_path}). "
                f"It must be saved as a directory using HuggingFace's format. "
                f"Remove the file extension and try again."
            )
        
        # Create directory if it doesn't exist
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the base model (unwrap DataParallel if needed)
        model_to_save = self.obj.module if hasattr(self.obj, 'module') else self.obj
        
        # Save using HuggingFace's save_pretrained method
        # This saves the model weights and configuration
        try:
            # For HuggingFace models, use save_pretrained
            model_to_save.save_pretrained(save_path)
            logger.info(f"Saved TITAN slide encoder to {save_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to save TITAN_SLIDE model to {save_path}: {e}"
            ) from e


class GigapathModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize Prov-GigaPath model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.GIGAPATH.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj = timm.create_model(model_path, pretrained=True)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for Prov-GigaPath.

        Returns:
            Composed transforms for Prov-GigaPath input preprocessing.
        """
        preprocessing = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return preprocessing


class GigapathSlideEncoderModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize Prov-GigaPath slide encoder model.

        This is the slide-level encoder from Prov-GigaPath that aggregates
        patch-level features into slide-level representations.

        Args:
            model_path: Path to slide encoder model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.GIGAPATH_SLIDE.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            # Load GigaPath slide encoder using their official API
            # See: https://github.com/prov-gigapath/prov-gigapath#inference-with-the-slide-encoder
            import gigapath.slide_encoder
            
            # gigapath expects "hf_hub:" format (underscore, not hyphen)
            model_path_fixed = model_path.replace("hf-hub:", "hf_hub:")
            # Use locking when downloading from HuggingFace
            with model_download_lock(model_path) as should_download:
                # create_model(repo_id, model_name, in_chans)
                # in_chans=1536 for GigaPath tile embeddings
                model_obj = gigapath.slide_encoder.create_model(
                    model_path_fixed, 
                    "gigapath_slide_enc12l768d", 
                    1536
                )
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_model_fun(self) -> Callable:
        """Get model inference function for GigaPath slide encoder.

        The GigaPath slide encoder uses the slide_encoder method which
        requires both tile embeddings and coordinates as arguments.

        Returns:
            Callable that takes tile embeddings and coordinates, returns slide-level features
            with shape (768,).
        """

        def model_fun(features, coords):
            """Run inference with mixed precision."""
            with (
                torch.no_grad(),
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type, dtype=torch.float16),
            ):
                features = features.to(self.device, non_blocking=True)
                coords = coords.to(self.device, non_blocking=True)
                # GigaPath slide encoder API: output = model(tile_embed, coordinates)
                # Returns a list with the tensor as first element
                result = self.obj(features, coords)
                if isinstance(result, list):
                    result = result[0]
                return result.squeeze(0).cpu()

        return model_fun

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing function for slide encoder.

        Slide encoders work on patch features, not images, so no preprocessing needed.

        Returns:
            None, as slide encoders don't preprocess images.
        """
        return None


class OptimusModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize H-Optimus-0 model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.OPTIMUS.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj = timm.create_model(
                model_path,
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False,
            )
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for H-Optimus-0.

        Returns:
            Composed transforms for H-Optimus-0 input preprocessing.
        """
        preprocessing = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )
        return preprocessing


class VirchowModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize Virchow model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.VIRCHOW.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj = timm.create_model(
                model_path,
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for Virchow.

        Returns:
            Preprocessing transforms resolved from model config.
        """
        preprocessing = create_transform(
            **resolve_data_config(self.obj.pretrained_cfg, model=self.obj)
        )
        return preprocessing

    def get_model_fun(self) -> Callable:
        """Get model inference function that concatenates class token with average pooled patch tokens.

        For Virchow, we concatenate the class token (CLS) with the average of patch tokens
        as recommended in: https://huggingface.co/paige-ai/Virchow#image-embeddings

        Returns:
            Callable that runs inference and returns concatenated embeddings.
        """

        def model_fun(x):
            """Run inference with mixed precision and concatenate class + avg patch tokens."""
            with (
                torch.no_grad(),
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type, dtype=torch.float16),
            ):
                x = x.to(self.device, non_blocking=True)
                if self.use_gpu:
                    try:
                        x = x.to(memory_format=torch.channels_last)
                    except:
                        pass
                
                output = self.obj(x)
                
                # Virchow returns [batch, num_tokens, embed_dim]
                # First token is CLS, rest are patch tokens
                class_token = output[:, 0]  # [batch, embed_dim]
                patch_tokens = output[:, 1:]  # [batch, num_patches, embed_dim]
                
                # Average pool the patch tokens
                avg_patch_tokens = patch_tokens.mean(dim=1)  # [batch, embed_dim]
                
                # Concatenate class token with averaged patch tokens
                concatenated = torch.cat([class_token, avg_patch_tokens], dim=1)  # [batch, embed_dim * 2]
                
                return concatenated.cpu()

        return model_fun


class Virchow2Model(VirchowModel):
    """Virchow2 model - uses same architecture and feature extraction as Virchow."""
    
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize Virchow2 model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.VIRCHOW2.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj = timm.create_model(
                model_path,
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        # Call TorchModel.__init__ directly to avoid calling VirchowModel.__init__
        TorchModel.__init__(self, model_path, model_obj, use_gpu, gpu_device_id)


class UniModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize UNI model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.UNI.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj = timm.create_model(
                model_path,
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
            )
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for UNI.

        Returns:
            Preprocessing transforms resolved from model config.
        """
        preprocessing = create_transform(
            **resolve_data_config(self.obj.pretrained_cfg, model=self.obj)
        )
        return preprocessing


class Uni2Model(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize UNI2 model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.UNI2.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            # UNI2 requires specific architecture parameters
            # See: https://huggingface.co/MahmoodLab/UNI2-h
            timm_kwargs = {
                'img_size': 224,
                'patch_size': 14,
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5,
                'embed_dim': 1536,
                'mlp_ratio': 2.66667 * 2,
                'num_classes': 0,
                'no_embed_class': True,
                'mlp_layer': SwiGLUPacked,
                'act_layer': torch.nn.SiLU,
                'reg_tokens': 8,
                'dynamic_img_size': True,
            }
            model_obj = timm.create_model(
                model_path,
                pretrained=True,
                **timm_kwargs
            )
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for UNI2.

        Returns:
            Preprocessing transforms resolved from model config.
        """
        preprocessing = create_transform(
            **resolve_data_config(self.obj.pretrained_cfg, model=self.obj)
        )
        return preprocessing


class ClipModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize CLIP (QuiltNet) model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.CLIP.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj, _, self.preprocessing = open_clip.create_model_and_transforms(
                model_path,
            )
            model_obj.forward = partial(model_obj.encode_image)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for CLIP.

        Returns:
            Preprocessing transforms from open_clip.
        """
        return self.preprocessing


class TransPathModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize TransPath model.

        Args:
            model_path: Path to model weights file.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).

        Raises:
            ValueError: If model_path is not provided.
        """
        if model_path is None:
            raise ValueError("model_path must be provided for TransPath model")
        from transpath.ctran import ctranspath

        model_obj = ctranspath()
        model_obj.head = nn.Identity()
        td = torch.load(model_path, weights_only=True)
        model_obj.load_state_dict(td["model"], strict=True)
        # ctranspath() module has required torch transforms built in so
        # preprocessing should be None here
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)


class ResnetModel(TorchModel):
    def __init__(
        self, use_gpu: bool = True, gpu_device_id: int | List[int] | None = None
    ):
        """Initialize ResNet-50 model.

        Args:
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        from mussel.models.resnet_custom import resnet50_baseline

        model_obj = resnet50_baseline(pretrained=True)
        super().__init__(None, model_obj, use_gpu, gpu_device_id)


MODEL_FACTORIES = {}


def register_model_factory(model_type: ModelType):
    """Decorator to register a model factory for a given model type.

    Args:
        model_type: The ModelType to register the factory for.

    Returns:
        Decorator function that registers the factory.
    """

    def decorator(fn):
        """Register factory function."""
        MODEL_FACTORIES[model_type] = fn
        return fn

    return decorator


class ModelFactory(ABC):
    @abstractmethod
    def get_model(self, model_path, use_gpu, gpu_device_id) -> Model:
        """Get a model instance.

        Args:
            model_path: Path to model weights or config.
            use_gpu: Whether to use GPU.
            gpu_device_id: GPU device ID or list of IDs.

        Returns:
            Model instance.
        """
        pass


@register_model_factory(ModelType.GOOGLEPATH)
class GooglePathModelFactory(ModelFactory):
    def get_model(self, model_path, use_gpu=True, gpu_device_id=None) -> Model:
        """Create GooglePath model instance."""
        return GooglePathModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.RESNET50)
class Resnet50ModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create ResNet-50 model instance."""
        return ResnetModel(use_gpu, gpu_device_id)


@register_model_factory(ModelType.CTRANSPATH)
class CTransPathModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create TransPath model instance."""
        return TransPathModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.GIGAPATH)
class GigapathModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create Prov-GigaPath model instance."""
        return GigapathModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.GIGAPATH_SLIDE)
class GigapathSlideEncoderModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create Prov-GigaPath slide encoder model instance."""
        return GigapathSlideEncoderModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.VIRCHOW)
class VirchowModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create Virchow model instance."""
        return VirchowModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.VIRCHOW2)
class Virchow2ModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create Virchow2 model instance."""
        return Virchow2Model(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.UNI)
class UniModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create UNI model instance."""
        return UniModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.UNI2)
class Uni2ModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create UNI2 model instance."""
        return Uni2Model(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.CONCH1_5)
class Conch15ModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create CONCH v1.5 model instance."""
        return Conch15Model(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.TITAN_SLIDE)
class TitanSlideEncoderModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create TITAN slide encoder model instance."""
        return TitanSlideEncoderModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.OPTIMUS)
class OptimusModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create H-Optimus-0 model instance."""
        return OptimusModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.CLIP)
class ClipModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create CLIP (QuiltNet) model instance."""
        return ClipModel(model_path, use_gpu, gpu_device_id)


def get_model_factory(
    model_type: ModelType | str = ModelType.CTRANSPATH,
) -> ModelFactory:
    """Get the model factory for a given model type.

    Args:
        model_type: ModelType enum or string name of the model (default: ModelType.CTRANSPATH).

    Returns:
        ModelFactory instance for the specified model type.

    Raises:
        ValueError: If model_type string is not recognized.
    """
    if isinstance(model_type, str):
        try:
            model_type = ModelType[model_type.upper()]
        except KeyError:
            raise ValueError(f"unknown model type: {model_type}")
    return MODEL_FACTORIES.get(model_type)()
