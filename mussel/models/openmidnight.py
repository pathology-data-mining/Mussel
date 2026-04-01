"""OpenMidnight model from SophontAI.

OpenMidnight is an open-weights pathology foundation model based on
DINOv2 ViT-G/14 trained on ~30k pathology slides.  The checkpoint is stored
on HuggingFace as ``SophontAI/OpenMidnight`` (``teacher_checkpoint_load.pt``).

Loading procedure (from the model card):
  1. Instantiate ``dinov2_vitg14_reg`` via ``torch.hub.load``
  2. Patch the positional embedding from the checkpoint (model was trained at
     224px resolution, whereas the base DINOv2 uses 392px)
  3. Load the state dict

Reference: https://huggingface.co/SophontAI/OpenMidnight

Feature dimension: 1536 (ViT-Giant)
Input: 224×224, ImageNet normalisation.
"""

import logging
from typing import Callable, List

import torch
from torchvision import transforms

from mussel.models.base import IMAGENET_MEAN, IMAGENET_STD, TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "teacher_checkpoint_load.pt"


@register_model(ModelType.OPENMIDNIGHT)
class OpenMidnightModel(TorchModel):
    """OpenMidnight — 1536-dim, 224px input, DINOv2 ViT-G/14 weights."""

    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize OpenMidnight.

        Args:
            model_path: HuggingFace repo ID (``SophontAI/OpenMidnight``) or
                path to a local ``teacher_checkpoint_load.pt`` file.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.OPENMIDNIGHT.path

        model_obj = self._load(model_path)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    @staticmethod
    def _load(model_path: str) -> torch.nn.Module:
        """Load OpenMidnight weights into a DINOv2 ViT-G/14 backbone."""
        import os

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError("huggingface_hub is required to load OpenMidnight") from e

        if os.path.isfile(model_path):
            ckpt_path = model_path
        else:
            repo_id = model_path.replace("hf-hub:", "")
            logger.info(f"Downloading OpenMidnight checkpoint from {repo_id}")
            ckpt_path = hf_hub_download(repo_id, _CHECKPOINT_FILENAME)

        model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitg14_reg",
            weights=None,
        )

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # The checkpoint encodes a 224px positional embedding; patch it in before
        # loading the state dict so the shapes match.
        model.pos_embed = torch.nn.Parameter(checkpoint["pos_embed"])
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def get_preprocessing_fun(self) -> Callable:
        """224×224 resize + ImageNet normalisation."""
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    @property
    def autocast_dtype(self) -> torch.dtype:
        return torch.float16 if self.device.type == "cuda" else torch.bfloat16

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.obj(x)
