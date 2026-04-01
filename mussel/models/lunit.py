"""Lunit DINO pathology models (ViT-S/8, ViT-S/16).

Weights are published as timm safetensors by the ``1aurent`` HuggingFace user,
re-hosting the Lunit DINO self-supervised models from
https://github.com/lunit-io/benchmark-ssl-pathology.

Feature dimension: 384 (ViT-Small) for both variants.
Input: 224×224, ImageNet normalisation (extracted from pretrained_cfg).
"""

import logging
from typing import Callable, List

import timm

from mussel.models.base import TorchModel, _timm_preprocessing
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


class _LunitBase(TorchModel):
    """Shared base for Lunit DINO timm models."""

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


@register_model(ModelType.LUNIT_VITS8)
class LunitViTS8Model(_LunitBase):
    """Lunit DINO ViT-S/8 — 384-dim, 224px input."""

    _default_model_type = ModelType.LUNIT_VITS8


@register_model(ModelType.LUNIT_VITS16)
class LunitViTS16Model(_LunitBase):
    """Lunit DINO ViT-S/16 — 384-dim, 224px input."""

    _default_model_type = ModelType.LUNIT_VITS16
