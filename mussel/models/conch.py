"""CONCH v1.5 patch encoder and TITAN slide encoder from MahmoodLab."""

import logging
from pathlib import Path
from typing import Callable, List

import torch
from torchvision import transforms
from transformers import AutoModel

from mussel.models.base import (IMAGENET_MEAN, IMAGENET_STD, TorchModel,
                                get_best_attn_implementation)

try:
    from mussel.utils.model_cache import model_download_lock
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def model_download_lock(model_name, **kwargs):
        yield True


from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


@register_model(ModelType.CONCH1_5)
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
                        attn_implementation=attn_impl,
                    )
                except (ValueError, NotImplementedError) as e:
                    # TITAN model doesn't support Flash Attention 2.0 or SDPA yet, fallback to eager
                    if "does not support an attention implementation" in str(
                        e
                    ) or "does not support Flash Attention" in str(e):
                        logger.info(
                            "TITAN model doesn't support optimized attention, using eager mode"
                        )
                        titan = AutoModel.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                            attn_implementation="eager",
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
        return transforms.Compose(
            [
                transforms.Resize(448),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

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


@register_model(ModelType.TITAN_SLIDE)
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
        # TITAN doesn't support Flash Attention 2.0, so we use eager mode
        # Use locking when downloading from HuggingFace
        with model_download_lock(model_path) as should_download:
            try:
                model_obj = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    attn_implementation="eager",
                )
            except TypeError:
                # Fallback for older transformers that don't support attn_implementation
                model_obj = AutoModel.from_pretrained(
                    model_path, trust_remote_code=True
                )
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
                return (
                    self.obj.encode_slide_from_patch_features(
                        patch_features, coords, patch_size
                    )
                    .squeeze()
                    .float()
                    .cpu()
                )

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
        model_to_save = self.obj.module if hasattr(self.obj, "module") else self.obj

        # Save using HuggingFace's save_pretrained method
        try:
            model_to_save.save_pretrained(save_path)
            logger.info(f"Saved TITAN slide encoder to {save_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to save TITAN_SLIDE model to {save_path}: {e}"
            ) from e
