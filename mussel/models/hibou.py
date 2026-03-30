"""Hibou-L model from histai."""

import logging
from typing import Callable, List

import torch
from transformers import AutoImageProcessor, AutoModel

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


@register_model(ModelType.HIBOU_L)
class HibouLModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize Hibou-L model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.HIBOU_L.path

        with model_download_lock(model_path) as _:
            model_obj = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            self._processor = AutoImageProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing function for Hibou-L.

        The AutoImageProcessor returns a dict with a 'pixel_values' key.
        This wrapper extracts the tensor for compatibility with Mussel's pipeline.

        Returns:
            Callable that preprocesses a PIL image into a single image tensor.
        """
        processor = self._processor

        def preprocess(img):
            """Preprocess a PIL image using the AutoImageProcessor."""
            return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        return preprocess

    def get_model_fun(self) -> Callable:
        """Get model inference function for Hibou-L.

        Extracts the CLS token from the last hidden state.

        Returns:
            Callable that runs inference and returns CLS token embeddings.
        """

        def model_fun(x):
            with torch.no_grad(), torch.inference_mode():
                x = x.to(self.device, non_blocking=True)
                output = self.obj(pixel_values=x)
                return output.last_hidden_state[:, 0].float().cpu()

        return model_fun
