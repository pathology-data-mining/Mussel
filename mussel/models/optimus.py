"""H-Optimus-0, H-Optimus-1, and H0-mini models from bioptimus."""

import logging
from typing import Callable, List

import timm
import torch
from torchvision import transforms

from mussel.models.base import TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)

_BIOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
_BIOPTIMUS_STD = (0.211883, 0.230117, 0.177517)


def _bioptimus_preprocessing() -> Callable:
    """Standard 224px preprocessing for all bioptimus H-Optimus models."""
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=_BIOPTIMUS_MEAN, std=_BIOPTIMUS_STD),
        ]
    )


@register_model(ModelType.OPTIMUS)
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
        return _bioptimus_preprocessing()


@register_model(ModelType.H_OPTIMUS_1)
class HOptimus1Model(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize H-Optimus-1 model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.H_OPTIMUS_1.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj = timm.create_model(
                model_path, pretrained=True, init_values=1e-5, dynamic_img_size=False
            )
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        return _bioptimus_preprocessing()


@register_model(ModelType.H0_MINI)
class H0MiniModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize H0-mini model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.H0_MINI.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj = timm.create_model(
                model_path,
                pretrained=True,
                mlp_layer=timm.layers.SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        return _bioptimus_preprocessing()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.obj(x)  # (N, num_tokens, embed_dim) — global_pool='' returns all tokens
        return out[:, 0]   # CLS token -> (N, embed_dim)
