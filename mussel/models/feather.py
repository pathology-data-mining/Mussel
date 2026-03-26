"""Feather slide encoder from MahmoodLab."""

import logging
import os
from pathlib import Path
from typing import Callable, List

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel

try:
    from mussel.utils.model_cache import model_download_lock
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def model_download_lock(model_name, **kwargs):
        yield True


from mussel.models.base import TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


@register_model(ModelType.FEATHER_SLIDE)
class FeatherSlideEncoderModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize Feather slide encoder model.

        Feather (MahmoodLab/FEATHER) is an ABMIL-based slide encoder that aggregates
        CONCH 1.5 patch features into slide-level representations.

        Args:
            model_path: HuggingFace repo ID or local directory.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.FEATHER_SLIDE.path
        hf_token = os.environ.get("HF_TOKEN")
        with model_download_lock(model_path) as _:
            local_path = snapshot_download(model_path, token=hf_token)
            model_obj = AutoModel.from_pretrained(local_path, trust_remote_code=True)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_model_fun(self) -> Callable:
        """Get model inference function for Feather slide encoder.

        Returns:
            Callable that takes patch_features [B, N, D] and returns slide-level features.
        """

        def model_fun(patch_features):
            with torch.no_grad(), torch.inference_mode():
                patch_features = patch_features.to(self.device, non_blocking=True)
                h, _ = self.obj.forward_features(patch_features, return_attention=False)
                return h.squeeze().cpu()

        return model_fun

    def get_preprocessing_fun(self) -> Callable:
        """Slide encoders work on patch features; no image preprocessing needed."""
        return None

    def save(self, save_path: str):
        """Save Feather slide encoder model to disk using HuggingFace's save_pretrained.

        Args:
            save_path: Path to save the model (must be a directory, not a file).

        Raises:
            ValueError: If save_path has a file extension.
        """
        if Path(save_path).suffix:
            raise ValueError(
                f"FEATHER_SLIDE model must be saved to a directory, not a file ({save_path})."
            )
        Path(save_path).mkdir(parents=True, exist_ok=True)
        model_to_save = self.obj.module if hasattr(self.obj, "module") else self.obj
        model_to_save.save_pretrained(save_path)
        logger.info(f"Saved Feather slide encoder to {save_path}")
