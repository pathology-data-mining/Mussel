"""MUSK vision encoder (xiangjx/musk).

MUSK (Multimodal transformer with Unified maSKed modeling) is a BEiT-3 style
vision-language model pretrained on unpaired pathology images and text, then
contrastively aligned on image-caption pairs.  This wrapper returns the image
CLS token passed through MUSK's vision head: the 1024-dim space shared with
its text tower.  Features can be compared directly (after L2 normalisation)
with MUSK text embeddings for zero-shot classification and retrieval.

MUSK's model code is not vendored (its CC-BY-NC-ND license forbids derivative
works).  Install it separately::

    pip install git+https://github.com/lilab-stanford/MUSK

The checkpoint is gated; accept the license on HuggingFace and set HF_TOKEN.
Weights are fetched with ``hf_hub_download`` into the standard HF cache rather
than MUSK's own helper, which writes a generically named
``~/.cache/model.safetensors``.

Multi-scale augmentation (``ms_aug``) is not applied: MUSK recommends it for
linear probing, but the contrastive alignment was trained on the single-scale
CLS token.

Reference: https://huggingface.co/xiangjx/musk

Feature dimension: 1024
Input: 384×384, Inception normalisation (mean = std = 0.5).
"""

import logging
import os
import sys
import types
from typing import Callable, List

import torch
import torch.nn as nn
from torchvision import transforms

from mussel.models.base import TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "model.safetensors"
_MUSK_FLASH_ATTENTION_MODULE = "musk.torchscale.component.flash_attention"
_INCEPTION_MEAN = [0.5, 0.5, 0.5]
_INCEPTION_STD = [0.5, 0.5, 0.5]


def _import_musk_modeling() -> None:
    """Import ``musk.modeling``, which registers musk_large_patch16_384 with timm.

    MUSK's attention module eagerly imports an optional fused-attention
    backend: flash-attn on sm80+ GPUs, xformers on older ones.  It only falls
    back to ``flash_attn_func = None`` on ModuleNotFoundError, so a backend
    that is installed but unimportable (e.g. xformers rejecting the installed
    flash-attn version on a V100) breaks the whole import.  The model never
    uses that backend (``flash_attention`` defaults to False and MUSK computes
    attention with plain matmuls), so in that case install MUSK's own
    "no backend" fallback and import again.
    """
    try:
        from musk import modeling  # noqa: F401

        return
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == "musk":
            # Keep the original "No module named 'musk'" text: callers (and
            # the integration tests' skip logic) match on it.
            raise ImportError(
                f"{e}. MUSK requires the 'musk' package: "
                "pip install git+https://github.com/lilab-stanford/MUSK"
            ) from e
        raise
    except ImportError as e:
        logger.warning(
            "MUSK's optional fused-attention backend failed to import (%s); "
            "using MUSK's built-in fallback, which the model uses anyway.",
            e,
        )
    stub = types.ModuleType(_MUSK_FLASH_ATTENTION_MODULE)
    stub.flash_attn_func = None
    sys.modules[_MUSK_FLASH_ATTENTION_MODULE] = stub
    from musk import modeling  # noqa: F401,F811


class _MuskImageEncoder(nn.Module):
    """Image-only forward through MUSK returning the text-aligned CLS embedding."""

    def __init__(self, musk_model: nn.Module) -> None:
        super().__init__()
        self.musk = musk_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MUSK's BEiT-3 embedding layers expect contiguous NCHW input.
        vision_cls, _ = self.musk(
            image=x.contiguous(),
            with_head=True,
            out_norm=False,
            ms_aug=False,
            return_global=True,
        )
        return vision_cls


@register_model(ModelType.MUSK)
class MuskModel(TorchModel):
    """MUSK — 1024-dim text-aligned embedding, 384px input, gated (xiangjx/musk)."""

    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize MUSK.

        Args:
            model_path: HuggingFace repo ID (``hf-hub:xiangjx/musk``) or path to
                a local ``model.safetensors`` file.  Requires a HuggingFace
                token with access to the gated xiangjx/musk repository.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.MUSK.path
        model_obj = self._load(model_path)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    @staticmethod
    def _load(model_path: str) -> nn.Module:
        _import_musk_modeling()
        from musk import utils as musk_utils
        from timm.models import create_model

        if os.path.isfile(model_path):
            ckpt_path = model_path
        else:
            from huggingface_hub import hf_hub_download

            repo_id = model_path.replace("hf-hub:", "")
            logger.info("Downloading MUSK checkpoint from %s", repo_id)
            ckpt_path = hf_hub_download(repo_id, _CHECKPOINT_FILENAME)

        model = create_model("musk_large_patch16_384")
        musk_utils.load_model_and_may_interpolate(ckpt_path, model, "model|module", "")
        model.eval()
        return _MuskImageEncoder(model)

    def get_preprocessing_fun(self) -> Callable:
        """384×384 bicubic resize + Inception normalisation (MUSK model card)."""
        return transforms.Compose(
            [
                transforms.Resize(
                    384, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize(mean=_INCEPTION_MEAN, std=_INCEPTION_STD),
            ]
        )
