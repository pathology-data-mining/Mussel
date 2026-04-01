"""Kaiko AI pathology foundation models (ViT-S/8, ViT-S/16, ViT-B/8, ViT-B/16, ViT-L/14).

All five variants are published as standard timm safetensors models by the
``1aurent`` HuggingFace user from the Kaiko AI "Towards Large Pathology FMs"
paper (https://arxiv.org/abs/2404.15217).  They are loaded via
``timm.create_model("hf-hub:…", pretrained=True)`` and preprocessing is
extracted automatically from the model's ``pretrained_cfg``.

Feature dimensions:
  ViT-S/8, ViT-S/16 → 384
  ViT-B/8, ViT-B/16 → 768
  ViT-L/14           → 1024
"""

import logging
from typing import Callable, List

import timm

from mussel.models.base import TorchModel, _timm_preprocessing
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


class _KaikoBase(TorchModel):
    """Shared base for all Kaiko timm models."""

    _default_model_type: ModelType

    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        if model_path is None:
            model_path = self._default_model_type.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj = timm.create_model(model_path, pretrained=True, num_classes=0)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        return _timm_preprocessing(self.obj)


@register_model(ModelType.KAIKO_VITS8)
class KaikoViTS8Model(_KaikoBase):
    """Kaiko ViT-S/8 — 384-dim, 224px input."""

    _default_model_type = ModelType.KAIKO_VITS8


@register_model(ModelType.KAIKO_VITS16)
class KaikoViTS16Model(_KaikoBase):
    """Kaiko ViT-S/16 — 384-dim, 224px input."""

    _default_model_type = ModelType.KAIKO_VITS16


@register_model(ModelType.KAIKO_VITB8)
class KaikoViTB8Model(_KaikoBase):
    """Kaiko ViT-B/8 — 768-dim, 224px input."""

    _default_model_type = ModelType.KAIKO_VITB8


@register_model(ModelType.KAIKO_VITB16)
class KaikoViTB16Model(_KaikoBase):
    """Kaiko ViT-B/16 — 768-dim, 224px input."""

    _default_model_type = ModelType.KAIKO_VITB16


@register_model(ModelType.KAIKO_VITL14)
class KaikoViTL14Model(_KaikoBase):
    """Kaiko ViT-L/14 — 1024-dim, 224px input."""

    _default_model_type = ModelType.KAIKO_VITL14
