"""Midnight-12k model from Kaiko AI."""

import logging
from typing import Callable, List

import torch
from torchvision import transforms
from transformers import AutoModel

from mussel.models.base import TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


@register_model(ModelType.MIDNIGHT12K)
class Midnight12kModel(TorchModel):
    """Midnight-12k (kaiko-ai/midnight) — DINOv2 ViT-g/14 trained on 12k pathology slides.

    Loaded via ``transformers.AutoModel`` (no timm preprocessor_config available).
    As on the kaiko-ai/midnight model card: 224×224 input normalised with mean and
    std 0.5 ("Our models are trained on 224x224 images normalized with a mean of
    (0.5, 0.5, 0.5) and a standard deviation of (0.5, 0.5, 0.5)"), and the
    classification embedding is the CLS token concatenated with the mean of the
    patch tokens, dim=3072.
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
        """224×224 resize/crop, mean/std 0.5 normalisation (model card)."""
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    @property
    def autocast_dtype(self) -> torch.dtype:
        return torch.float16 if self.device.type == "cuda" else torch.bfloat16

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # CLS token + mean of the patch tokens (the card's extract_classification_embedding).
        tokens = self.obj(pixel_values=x).last_hidden_state
        return torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=-1)
