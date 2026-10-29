"""Virchow and Virchow2 models from Paige AI."""

import logging
from typing import Callable, List

import timm
import torch
import torch.nn as nn
from timm.layers import SwiGLUPacked

from mussel.models.base import TorchModel, _timm_preprocessing
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


@register_model(ModelType.VIRCHOW)
class VirchowModel(TorchModel):
    """Virchow model base class with shared preprocessing and inference logic."""

    # Default model type - subclasses can override this
    _default_model_type = ModelType.VIRCHOW
    # Register tokens between the CLS token and the patch tokens. Used only when
    # the loaded model does not report ``num_reg_tokens`` itself.
    _num_register_tokens = 0

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
        class_token = output[:, 0]  # (N, embed_dim)
        start = self._patch_token_start()
        avg_patch_tokens = output[:, start:].mean(dim=1)  # (N, embed_dim)
        return torch.cat([class_token, avg_patch_tokens], dim=1)  # (N, embed_dim*2)

    def _patch_token_start(self) -> int:
        """Index of the first patch token: after the CLS token and any register tokens.

        Virchow2 has 4 register tokens, so its card averages ``output[:, 5:]``;
        Virchow has none and averages ``output[:, 1:]``.
        """
        model = self.obj.module if isinstance(self.obj, nn.DataParallel) else self.obj
        num_reg = getattr(model, "num_reg_tokens", None)
        if not isinstance(num_reg, int):
            num_reg = self._num_register_tokens
        return 1 + num_reg


@register_model(ModelType.VIRCHOW2)
class Virchow2Model(VirchowModel):
    """Virchow2 model - same embedding as Virchow, skipping its 4 register tokens."""

    # Override default model type for Virchow2
    _default_model_type = ModelType.VIRCHOW2
    # paige-ai/Virchow2 card: reg_tokens=4, so patch tokens are output[:, 5:]
    _num_register_tokens = 4
