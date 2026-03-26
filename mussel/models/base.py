"""Base model classes and shared utilities for all model implementations."""

import json
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Callable, List

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

try:
    from mussel.utils.model_cache import model_download_lock
except ImportError:
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
    try:
        import flash_attn

        # Check CUDA capability (flash_attn requires compute capability >= 8.0)
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            logger.info("Using flash_attention_2 for HuggingFace models")
            return "flash_attention_2"
    except (ImportError, AttributeError) as exc:
        # flash_attn is optional; if unavailable or incompatible we fall back to SDPA/eager
        logger.debug(
            "flash_attn not available or incompatible; falling back to other attention implementations: %s",
            exc,
        )

    # SDPA is available in PyTorch 2.0+ and will use xformers if available
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        logger.info("Using sdpa (scaled_dot_product_attention) for HuggingFace models")
        return "sdpa"

    # Fallback to eager
    logger.warning("Using eager attention (no acceleration available)")
    return "eager"


def _timm_preprocessing(model_obj: nn.Module) -> Callable:
    """Get preprocessing transforms from a timm model's pretrained config.

    Used by models that don't need custom preprocessing (UNI, UNI2, Virchow, Virchow2).

    Args:
        model_obj: A timm model with a pretrained_cfg attribute.

    Returns:
        Callable preprocessing transform.
    """
    return create_transform(**resolve_data_config(model_obj.pretrained_cfg, model=model_obj))


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
            torch.set_float32_matmul_precision("high")
            # Convert to channels_last memory format for better GPU utilization
            try:
                self.obj = self.obj.to(memory_format=torch.channels_last)
            except Exception:
                pass  # Some models may not support channels_last

        self.obj = self.obj.to(self.device)
        self.obj.eval()

        # torch.compile is disabled for now due to compatibility issues.
        # Consider re-enabling after testing individual models with it.

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
                    except Exception:
                        pass  # Some tensor shapes may not support channels_last
                return self.obj(x).cpu()

        return model_fun

    def save(self, save_path: str):
        """Save PyTorch model to disk using pickle.

        Args:
            save_path: Path to save the model.
        """
        with open(save_path, "wb") as f:
            pickle.dump(self.obj, f)
