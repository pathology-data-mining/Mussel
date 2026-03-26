"""Phikon and Phikon-v2 models from Owkin."""

import logging
from typing import Callable, List

import torch
from torchvision import transforms
from transformers import AutoModel

from mussel.models.base import IMAGENET_MEAN, IMAGENET_STD, TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


@register_model(ModelType.PHIKON)
class PhikonModel(TorchModel):
    """Phikon model base class for owkin/phikon and owkin/phikon-v2.

    These models use the HuggingFace Transformers format (ViT/DINOv2) and are
    loaded via ``transformers.AutoModel`` rather than timm, since their
    ``config.json`` uses the ``architectures`` key (Transformers style) which
    timm 1.0.x cannot parse.
    """

    _default_model_type = ModelType.PHIKON

    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize Phikon model.

        Args:
            model_path: HuggingFace repo ID (``hf-hub:owkin/phikon``) or local path.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = self._default_model_type.path
        repo_id = model_path.replace("hf-hub:", "")
        # ViTModel accepts add_pooling_layer; DINOv2/other models do not.
        try:
            model_obj = AutoModel.from_pretrained(repo_id, add_pooling_layer=False)
        except TypeError:
            model_obj = AutoModel.from_pretrained(repo_id)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """ImageNet-normalised 224×224 transforms (standard for Phikon models)."""
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

    def get_model_fun(self) -> Callable:
        """Return the CLS-token embedding from the transformer output."""
        autocast_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16

        def model_fun(x):
            with (
                torch.no_grad(),
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type, dtype=autocast_dtype),
            ):
                x = x.to(self.device, non_blocking=True)
                out = self.obj(pixel_values=x)
                return out.last_hidden_state[:, 0].cpu()

        return model_fun


@register_model(ModelType.PHIKON_V2)
class PhikonV2Model(PhikonModel):
    """Phikon-v2 model (owkin/phikon-v2, DINOv2 ViT-L/14)."""

    _default_model_type = ModelType.PHIKON_V2
