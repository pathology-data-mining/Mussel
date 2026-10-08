"""KEEP vision encoder (Astaxanthin/KEEP).

KEEP (Knowledge-Enhanced Pathology) is a CLIP-style vision-language model whose
image tower is a ViT-L/16 followed by a two-layer projection head into a
768-dim space shared with its BERT text tower.  Features from this model can
be compared directly (after L2 normalisation) with KEEP text embeddings for
zero-shot classification and retrieval.

The HuggingFace checkpoint ships ``trust_remote_code`` modeling code that
monkey-patches ``timm.models.vision_transformer.LayerScale`` only to rename
its ``gamma`` parameter to ``weight``.  That patch breaks on recent timm
releases (``LayerScale`` is now constructed with ``device``/``dtype``
kwargs), so instead we:

  1. Download ``model.safetensors`` and ``config.json`` via HF Hub.
  2. Build a standard timm ViT-L/16 from the vision config.
  3. Load the ``visual.*`` weights, renaming ``ls{1,2}.weight`` ->
     ``ls{1,2}.gamma``, and the ``visual_head.*`` projection.

The text tower is not loaded.

Reference: https://huggingface.co/Astaxanthin/KEEP

Feature dimension: 768
Input: 224×224, ImageNet normalisation.
"""

import json
import logging
import os
import re
from typing import Callable, List

import timm
import torch
import torch.nn as nn
from torchvision import transforms

from mussel.models.base import IMAGENET_MEAN, IMAGENET_STD, TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "model.safetensors"
_CONFIG_FILENAME = "config.json"
_LAYERSCALE_KEY = re.compile(r"\.ls([12])\.weight$")


class _KeepVisualEncoder(nn.Module):
    """KEEP image tower: ViT-L/16 trunk + 2-layer MLP projection."""

    def __init__(self, vision_config: dict, projection_dim: int) -> None:
        super().__init__()
        self.visual = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            img_size=vision_config["img_size"],
            patch_size=vision_config["patch_size"],
            init_values=vision_config["init_values"],
            num_classes=0,
        )
        self.visual_head = nn.Sequential(
            nn.Linear(self.visual.num_features, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual_head(self.visual(x))


@register_model(ModelType.KEEP)
class KeepModel(TorchModel):
    """KEEP — 768-dim text-aligned embedding, 224px input (Astaxanthin/KEEP)."""

    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize KEEP.

        Args:
            model_path: HuggingFace repo ID (``hf-hub:Astaxanthin/KEEP``) or a
                local directory containing ``model.safetensors`` and
                ``config.json``.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.KEEP.path
        model_obj = self._load(model_path)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    @staticmethod
    def _load(model_path: str) -> nn.Module:
        from safetensors.torch import load_file

        if os.path.isdir(model_path):
            ckpt_path = os.path.join(model_path, _CHECKPOINT_FILENAME)
            config_path = os.path.join(model_path, _CONFIG_FILENAME)
        else:
            from huggingface_hub import hf_hub_download

            repo_id = model_path.replace("hf-hub:", "")
            logger.info("Downloading KEEP checkpoint from %s", repo_id)
            ckpt_path = hf_hub_download(repo_id, _CHECKPOINT_FILENAME)
            config_path = hf_hub_download(repo_id, _CONFIG_FILENAME)

        with open(config_path) as f:
            config = json.load(f)
        model = _KeepVisualEncoder(config["vision_config"], config["projection_dim"])

        checkpoint = load_file(ckpt_path)
        visual_sd = {
            _LAYERSCALE_KEY.sub(r".ls\1.gamma", k[len("visual.") :]): v
            for k, v in checkpoint.items()
            if k.startswith("visual.")
        }
        model.visual.load_state_dict(visual_sd, strict=True)
        head_sd = {
            k[len("visual_head.") :]: v
            for k, v in checkpoint.items()
            if k.startswith("visual_head.")
        }
        model.visual_head.load_state_dict(head_sd, strict=True)

        model.eval()
        return model

    def get_preprocessing_fun(self) -> Callable:
        """224×224 bicubic resize + ImageNet normalisation (KEEP model card)."""
        return transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
