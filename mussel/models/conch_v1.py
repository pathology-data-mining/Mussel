"""CONCH v1.0 model from MahmoodLab.

CONCH v1.0 (CONtrastive learning from Captions for Histopathology) is the
original 512-dim contrastive vision-language model from MahmoodLab, distinct
from CONCH v1.5 (768-dim, used by TITAN).  Weights are gated on HuggingFace
(requires accepting the MahmoodLab/CONCH license).

The checkpoint does not include a timm-compatible config.json, so it cannot
be loaded via ``timm.create_model('hf-hub:...')``.  Instead we:

  1. Download ``MahmoodLab/CONCH/pytorch_model.bin`` via HF Hub.
  2. Build a standard timm ViT-B/16 trunk at 448px (keys match exactly after
     stripping the ``visual.trunk.`` prefix).
  3. Build a custom attention-pool head matching the ``visual.attn_pool_contrast``
     sub-module in the checkpoint to produce 512-dim output.

Reference: https://huggingface.co/MahmoodLab/CONCH

Feature dimension: 512
Input: 448×448, ImageNet normalisation.
"""

import logging
import os
from typing import Callable, List

import timm
import torch
import torch.nn as nn
from torchvision import transforms

from mussel.models.base import IMAGENET_MEAN, IMAGENET_STD, TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)

_HF_REPO_ID = "MahmoodLab/CONCH"
_CHECKPOINT_FILENAME = "pytorch_model.bin"


class _ConchAttentionPool(nn.Module):
    """Attention-pool head from the CONCH v1.0 visual encoder.

    A single learnable query attends over all ViT patch tokens (768-dim) and
    produces a 512-dim pooled representation, matching the
    ``visual.attn_pool_contrast`` sub-module in the original checkpoint.
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, out_dim))
        self.ln_q = nn.LayerNorm(out_dim)
        self.ln_k = nn.LayerNorm(in_dim)
        # When kdim != embed_dim, PyTorch MHA stores q/k/v proj weights
        # separately (matching the checkpoint layout).
        self.attn = nn.MultiheadAttention(
            out_dim, num_heads, kdim=in_dim, vdim=in_dim, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool ViT tokens to a single vector.

        Args:
            x: Token sequence ``[B, N, in_dim]``.

        Returns:
            Pooled features ``[B, out_dim]``.
        """
        B = x.size(0)
        q = self.ln_q(self.query).unsqueeze(0).expand(B, 1, -1)  # [B, 1, out_dim]
        k = self.ln_k(x)  # [B, N, in_dim]
        out, _ = self.attn(q, k, k)  # [B, 1, out_dim]
        return out.squeeze(1)  # [B, out_dim]


class _ConchVisualEncoder(nn.Module):
    """CONCH v1.0 visual encoder: ViT-B/16 trunk + attention pool → 512-dim."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            img_size=448,
        )
        self.attn_pool_contrast = _ConchAttentionPool(768, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk.forward_features(x)  # [B, 785, 768]
        return self.attn_pool_contrast(features)  # [B, 512]


@register_model(ModelType.CONCH_V1)
class ConchV1Model(TorchModel):
    """CONCH v1.0 — 512-dim, 448px input, gated (MahmoodLab/CONCH)."""

    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize CONCH v1.0.

        Args:
            model_path: HuggingFace repo ID (``MahmoodLab/CONCH``) or path to
                a local ``pytorch_model.bin`` file.  Requires a HuggingFace
                token with access to the gated MahmoodLab/CONCH repository.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.CONCH_V1.path
        model_obj = self._load(model_path)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    @staticmethod
    def _load(model_path: str) -> nn.Module:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError("huggingface_hub is required to load CONCH v1.0") from e

        if os.path.isfile(model_path):
            ckpt_path = model_path
        else:
            repo_id = model_path.replace("hf-hub:", "")
            logger.info("Downloading CONCH v1.0 checkpoint from %s", repo_id)
            ckpt_path = hf_hub_download(repo_id, _CHECKPOINT_FILENAME)

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        model = _ConchVisualEncoder()

        trunk_sd = {
            k[len("visual.trunk.") :]: v
            for k, v in checkpoint.items()
            if k.startswith("visual.trunk.")
        }
        model.trunk.load_state_dict(trunk_sd, strict=True)

        pool_sd = {
            k[len("visual.attn_pool_contrast.") :]: v
            for k, v in checkpoint.items()
            if k.startswith("visual.attn_pool_contrast.")
        }
        model.attn_pool_contrast.load_state_dict(pool_sd, strict=True)

        model.eval()
        return model

    def get_preprocessing_fun(self) -> Callable:
        """448×448 resize + ImageNet normalisation."""
        return transforms.Compose(
            [
                transforms.Resize(448),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
