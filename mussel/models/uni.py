"""UNI and UNI2-h models from MahmoodLab."""

import logging
from typing import Callable, List

import timm
import torch
from timm.layers import SwiGLUPacked

from mussel.models.base import TorchModel, _timm_preprocessing
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


@register_model(ModelType.UNI)
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
        return _timm_preprocessing(self.obj)


@register_model(ModelType.UNI2H)
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
            model_path = ModelType.UNI2H.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            # UNI2 requires specific architecture parameters
            # See: https://huggingface.co/MahmoodLab/UNI2-h
            timm_kwargs = {
                "img_size": 224,
                "patch_size": 14,
                "depth": 24,
                "num_heads": 24,
                "init_values": 1e-5,
                "embed_dim": 1536,
                "mlp_ratio": 2.66667 * 2,
                "num_classes": 0,
                "no_embed_class": True,
                "mlp_layer": SwiGLUPacked,
                "act_layer": torch.nn.SiLU,
                "reg_tokens": 8,
                "dynamic_img_size": True,
            }
            model_obj = timm.create_model(model_path, pretrained=True, **timm_kwargs)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for UNI2.

        Returns:
            Preprocessing transforms resolved from model config.
        """
        return _timm_preprocessing(self.obj)
