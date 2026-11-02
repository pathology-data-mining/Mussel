"""CLIP (QuiltNet) model using open_clip."""

import logging
from typing import Callable, List

import open_clip
import torch
import torch.nn as nn

from mussel.models.base import TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


class _ClipImageEncoder(nn.Module):
    """Expose an open_clip model's ``encode_image`` as ``forward``.

    Patching ``model.forward`` with a bound method stores it in the instance
    ``__dict__``; ``nn.DataParallel`` replicas copy ``__dict__``, so every
    replica would call the original (cuda:0) module and fail with a device
    mismatch.  Routing through a real ``forward`` resolves ``self.model`` on
    each replica.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(x)


@register_model(ModelType.CLIP)
class ClipModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize CLIP (QuiltNet) model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.CLIP.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj, _, self.preprocessing = open_clip.create_model_and_transforms(
                model_path,
            )
            model_obj = _ClipImageEncoder(model_obj)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for CLIP.

        Returns:
            Preprocessing transforms from open_clip.
        """
        return self.preprocessing
