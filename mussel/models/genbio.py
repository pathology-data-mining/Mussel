"""GenBio-PathFM foundation model.

GenBio-PathFM is a 1.1B-parameter pathology foundation model trained with a
JEDI (JEPA + DINO) strategy.  It processes each RGB channel independently
through a single-channel ViT-G/16 backbone, and concatenates the three
per-channel CLS tokens into a 4608-dimensional feature vector.

References:
  - https://huggingface.co/genbio-ai/genbio-pathfm
  - https://github.com/genbio-ai/genbio-pathfm

Feature dimension: 4608 (3 × embed_dim=1536)
Input: 224×224 with pathology-specific normalisation
  mean=(0.697, 0.575, 0.728), std=(0.188, 0.240, 0.187)

The model architecture is vendored from the upstream repository into
``mussel/models/_genbio_pathfm.py`` (GenBio AI Community License — non-commercial
research use only).  Weights are downloaded automatically from HuggingFace Hub.
"""

import logging
import os
from typing import Callable, List

import torch

from mussel.models._genbio_pathfm import GenBio_PathFM_Inference
from mussel.models.base import TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "model.pth"


@register_model(ModelType.GENBIO_PATHFM)
class GenBioPathFMModel(TorchModel):
    """GenBio-PathFM — 4608-dim, 224px input."""

    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize GenBio-PathFM.

        Args:
            model_path: HuggingFace repo ID (``genbio-ai/genbio-pathfm``) or
                local path to ``model.pth``.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.GENBIO_PATHFM.path

        device_str = "cuda" if use_gpu else "cpu"
        model_obj = self._load_genbio(model_path, device_str)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    @staticmethod
    def _load_genbio(model_path: str, device: str) -> torch.nn.Module:
        """Download weights and instantiate ``GenBio_PathFM_Inference``."""
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("huggingface_hub is required to download weights") from e

        if os.path.isfile(model_path):
            weights_path = model_path
        else:
            repo_id = model_path.replace("hf-hub:", "")
            logger.info("Downloading GenBio-PathFM weights from %s", repo_id)
            weights_path = hf_hub_download(repo_id, _CHECKPOINT_FILENAME)

        return GenBio_PathFM_Inference(weights_path, device=device)

    def get_preprocessing_fun(self) -> Callable:
        """Pathology-normalised 224×224 transforms from GenBio_PathFM_Inference."""
        return self.obj.transform

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.obj(x)

    @property
    def autocast_dtype(self) -> torch.dtype:
        return torch.float16 if self.device.type == "cuda" else torch.bfloat16
