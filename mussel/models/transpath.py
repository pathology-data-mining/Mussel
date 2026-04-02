"""TransPath (CTransPath) model with ConvStem patch embedding."""

import logging
import re
from pathlib import Path
from typing import List

import timm
import torch
import torch.nn as nn
from timm.layers import Format, nchw_to

from mussel.models.base import TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


class _ConvStem(nn.Module):
    """ConvStem patch embedding for CTransPath (Xiyue-Wang/TransPath).

    Reimplements the custom embed layer from the original TransPath repo using
    native timm (>= 0.9).  Modern timm's SwinTransformer passes
    ``output_fmt='NHWC'`` to the embed layer and reads ``patch_embed.grid_size``,
    so this class mirrors the ``PatchEmbed`` interface.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        output_fmt=None,
        **kwargs,
    ):
        super().__init__()
        assert patch_size == 4
        assert embed_dim % 8 == 0
        img_size = (
            (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        )
        patch_size_t = (
            (patch_size, patch_size)
            if isinstance(patch_size, int)
            else tuple(patch_size)
        )
        self.img_size = img_size
        self.patch_size = patch_size_t
        self.grid_size = (
            img_size[0] // patch_size_t[0],
            img_size[1] // patch_size_t[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        stem: List[nn.Module] = []
        input_dim, output_dim = 3, embed_dim // 8
        for _ in range(2):
            stem += [
                nn.Conv2d(input_dim, output_dim, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            ]
            input_dim, output_dim = output_dim, output_dim * 2
        stem.append(nn.Conv2d(input_dim, embed_dim, 1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)  # e.g. NHWC for SwinTransformer
        return self.norm(x)


@register_model(ModelType.CTRANSPATH)
class TransPathModel(TorchModel):
    _GDRIVE_FILE_ID = "1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX"
    _CACHE_FILENAME = "ctranspath.pth"

    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize TransPath model.

        If ``model_path`` is None the checkpoint is downloaded automatically
        from Google Drive (https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX)
        and cached in the HuggingFace hub cache directory.

        Args:
            model_path: Path to local weights file, or None to auto-download.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = self._download_checkpoint()

        model_obj = timm.create_model(
            "swin_tiny_patch4_window7_224",
            embed_layer=_ConvStem,
            pretrained=False,
            num_classes=0,
        )
        td = torch.load(model_path, weights_only=True)
        # Remap downsample key indices: old timm placed the PatchMerging at the
        # END of stage i (layers.{i}.downsample), new timm places it at the
        # START of stage i+1 (layers.{i+1}.downsample).
        remapped = {}
        for k, v in td["model"].items():
            m = re.match(r"^layers\.(\d+)\.(downsample\..+)$", k)
            remapped[f"layers.{int(m.group(1))+1}.{m.group(2)}" if m else k] = v
        # strict=False: old checkpoints may contain non-persistent buffers
        # (e.g. relative_position_index) that new timm no longer stores.
        # Missing keys are checked to ensure all model parameters are covered.
        result = model_obj.load_state_dict(remapped, strict=False)
        missing = [
            k for k in result.missing_keys if not k.endswith("num_batches_tracked")
        ]
        if missing:
            raise RuntimeError(f"TransPath checkpoint is missing keys: {missing}")
        # ctranspath uses standard ImageNet normalization — preprocessing is
        # handled by the feature-extraction pipeline (preprocessing=None → ImageNet default)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    @classmethod
    def _download_checkpoint(cls) -> str:
        """Download ctranspath.pth from Google Drive, caching in the HF hub cache dir."""
        from huggingface_hub import constants as hf_constants

        cache_dir = Path(hf_constants.HF_HUB_CACHE) / "transpath"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / cls._CACHE_FILENAME
        if dest.exists():
            logger.info("TransPath checkpoint already cached at %s", dest)
            return str(dest)
        import gdown

        url = f"https://drive.google.com/uc?id={cls._GDRIVE_FILE_ID}"
        logger.info("Downloading TransPath checkpoint from Google Drive → %s", dest)
        gdown.download(url, str(dest), quiet=False)
        return str(dest)
