"""Virchow and Virchow2 models from Paige AI."""

import logging
from typing import Callable, List

import torch
import timm
from timm.layers import SwiGLUPacked

from mussel.models.base import TorchModel, _timm_preprocessing
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


@register_model(ModelType.VIRCHOW)
class VirchowModel(TorchModel):
    """Virchow model base class with shared preprocessing and inference logic."""

    # Default model type - subclasses can override this
    _default_model_type = ModelType.VIRCHOW

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
            model_path = self._default_model_type.path
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
        return _timm_preprocessing(self.obj)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate CLS token with average-pooled patch tokens.

        As recommended in https://huggingface.co/paige-ai/Virchow#image-embeddings,
        the embedding is CLS ++ mean(patch_tokens), doubling the embed dim.
        """
        output = self.obj(x)  # (N, num_tokens, embed_dim)
        class_token = output[:, 0]           # (N, embed_dim)
        avg_patch_tokens = output[:, 1:].mean(dim=1)  # (N, embed_dim)
        return torch.cat([class_token, avg_patch_tokens], dim=1)  # (N, embed_dim*2)


@register_model(ModelType.VIRCHOW2)
class Virchow2Model(VirchowModel):
    """Virchow2 model - uses same architecture and feature extraction as Virchow."""

    # Override default model type for Virchow2
    _default_model_type = ModelType.VIRCHOW2
