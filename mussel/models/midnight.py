"""Midnight-12k model from Kaiko AI."""

import logging
from typing import Callable, List

import torch
from torchvision import transforms
from transformers import AutoModel

from mussel.models.base import IMAGENET_MEAN, IMAGENET_STD, TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


@register_model(ModelType.MIDNIGHT12K)
class Midnight12kModel(TorchModel):
    """Midnight-12k (kaiko-ai/midnight) — DINOv2 ViT-g/14 trained on 12k pathology slides.

    Loaded via ``transformers.AutoModel`` (no timm preprocessor_config available).
    Input size: 518×518, ImageNet normalisation.
    Output: CLS token, dim=1536.
    """

    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize Midnight-12k model.

        Args:
            model_path: HuggingFace repo ID (``hf-hub:kaiko-ai/midnight``) or local path.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.MIDNIGHT12K.path
        repo_id = model_path.replace("hf-hub:", "")
        model_obj = AutoModel.from_pretrained(repo_id)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """518×518 resize + ImageNet normalisation (DINOv2 ViT-g/14 native input size)."""
        return transforms.Compose(
            [
                transforms.Resize(518),
                transforms.CenterCrop(518),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

    def get_model_fun(self) -> Callable:
        """Return the CLS-token embedding from the DINOv2 output."""
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
