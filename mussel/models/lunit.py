"""Lunit DINO pathology models (ViT-S/8, ViT-S/16).

Weights are published as timm safetensors by the ``1aurent`` HuggingFace user,
re-hosting the Lunit DINO self-supervised models from
https://github.com/lunit-io/benchmark-ssl-pathology.

Feature dimension: 384 (ViT-Small) for both variants.
Input: 224×224, ImageNet normalisation (extracted from pretrained_cfg).
"""

from mussel.models.base import _TimmHfHubBase
from mussel.models.model_factory import ModelType, register_model


@register_model(ModelType.LUNIT_VITS8)
class LunitViTS8Model(_TimmHfHubBase):
    """Lunit DINO ViT-S/8 — 384-dim, 224px input."""

    _default_model_type = ModelType.LUNIT_VITS8


@register_model(ModelType.LUNIT_VITS16)
class LunitViTS16Model(_TimmHfHubBase):
    """Lunit DINO ViT-S/16 — 384-dim, 224px input."""

    _default_model_type = ModelType.LUNIT_VITS16

