"""ResNet-50 baseline model."""

import logging
from typing import List

from mussel.models.base import TorchModel
from mussel.models.model_factory import ModelType, register_model
from mussel.models.resnet_custom import resnet50_baseline

logger = logging.getLogger(__name__)


@register_model(ModelType.RESNET50)
class ResnetModel(TorchModel):
    def __init__(
        self,
        model_path=None,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize ResNet-50 model.

        Args:
            model_path: Ignored. ResNet-50 weights are loaded via torchvision.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        model_obj = resnet50_baseline(pretrained=True)
        super().__init__(None, model_obj, use_gpu, gpu_device_id)
