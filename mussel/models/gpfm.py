"""GPFM model (majiabo/GPFM)."""

import logging
import os
from typing import Callable, List

import timm
import torch
from huggingface_hub import hf_hub_download
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

try:
    from mussel.utils.model_cache import model_download_lock
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def model_download_lock(model_name, **kwargs):
        yield True


from mussel.models.base import TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


@register_model(ModelType.GPFM)
class GPFMModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize GPFM model.

        GPFM (majiabo/GPFM) is a ViT-L/14 model whose HF repo only contains a raw
        ``GPFM.pth`` state-dict (no timm config.json).  We create the architecture
        explicitly and load the weights manually.

        Args:
            model_path: Path to a local ``.pth`` file or HuggingFace repo ID
                (``hf-hub:majiabo/GPFM``).
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.GPFM.path

        if model_path.startswith("hf-hub:"):
            repo_id = model_path[len("hf-hub:") :]
            hf_token = os.environ.get("HF_TOKEN")
            with model_download_lock(model_path) as _:
                pth_path = hf_hub_download(repo_id, "GPFM.pth", token=hf_token)
        else:
            pth_path = model_path

        # GPFM is a ViT-L/14 trained at 224px
        # (embed_dim=1024, depth=24, num_heads=16, patch_size=14, img_size=224 → 257 pos tokens)
        model_obj = timm.create_model(
            "vit_large_patch14_dinov2",
            pretrained=False,
            num_classes=0,
            img_size=224,
        )
        state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
        model_obj.load_state_dict(state_dict, strict=True)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for GPFM (224px input).

        GPFM was trained at 224px despite being based on a DINOv2 ViT-L/14 architecture
        (which defaults to 518px). Override input_size to match training.
        """
        cfg = resolve_data_config(self.obj.pretrained_cfg, model=self.obj)
        cfg["input_size"] = (3, 224, 224)
        return create_transform(**cfg)
