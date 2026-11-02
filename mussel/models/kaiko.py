"""Kaiko AI pathology foundation models (ViT-S/8, ViT-S/16, ViT-B/8, ViT-B/16, ViT-L/14).

All five variants are published as standard timm safetensors models by the
``1aurent`` HuggingFace user from the Kaiko AI "Towards Large Pathology FMs"
paper (https://arxiv.org/abs/2404.15217).  They are loaded via
``timm.create_model("hf-hub:…", pretrained=True)``.

Preprocessing follows the model cards rather than the timm ``pretrained_cfg``
(which carries ImageNet mean/std, and 518 px input for ViT-L/14): Resize(224),
CenterCrop(224), Normalize(mean=0.5, std=0.5).  ViT-L/14 is created with
``dynamic_img_size=True`` so it runs at 224 px, as its card does.

Feature dimensions:
  ViT-S/8, ViT-S/16 → 384
  ViT-B/8, ViT-B/16 → 768
  ViT-L/14           → 1024
"""

from typing import Callable

from torchvision import transforms

from mussel.models.base import _TimmHfHubBase
from mussel.models.model_factory import ModelType, register_model

_KAIKO_MEAN = (0.5, 0.5, 0.5)
_KAIKO_STD = (0.5, 0.5, 0.5)


class _KaikoBase(_TimmHfHubBase):
    """Shared Kaiko preprocessing from the model cards."""

    def get_preprocessing_fun(self) -> Callable:
        """Resize(224), CenterCrop(224), ToTensor, Normalize(0.5, 0.5)."""
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=_KAIKO_MEAN, std=_KAIKO_STD),
            ]
        )


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
    """Kaiko ViT-L/14 — 1024-dim, 224px input.

    The timm config fixes the input at 518 px; ``dynamic_img_size=True`` lets it
    run at 224 px as in the model card.
    """

    _default_model_type = ModelType.KAIKO_VITL14
    _create_model_kwargs = {"dynamic_img_size": True}
