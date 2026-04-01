"""CONCH v1.0 model from MahmoodLab.

CONCH v1.0 (CONtrastive learning from Captions for Histopathology) is the
original 512-dim contrastive vision-language model from MahmoodLab, distinct
from CONCH v1.5 (768-dim, used by TITAN).  It is stored as a timm model on
HuggingFace (gated — requires accepting the MahmoodLab/CONCH license).

Reference: https://huggingface.co/MahmoodLab/CONCH

Feature dimension: 512
Input: 448×448, ImageNet normalisation (extracted from pretrained_cfg).
"""

import logging
from typing import Callable, List

import timm

from mussel.models.base import TorchModel, _timm_preprocessing
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


@register_model(ModelType.CONCH_V1)
class ConchV1Model(TorchModel):
    """CONCH v1.0 — 512-dim, 448px input, gated (MahmoodLab/CONCH).

    Uses timm ``hf-hub:MahmoodLab/CONCH`` with ``checkpoint_path=''`` to skip
    the default weight download and rely on HuggingFace Hub caching instead,
    per the standard timm HF-hub loading pattern.
    """

    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize CONCH v1.0.

        Args:
            model_path: HuggingFace repo ID (``hf-hub:MahmoodLab/CONCH``) or
                local path. Requires a HuggingFace token with access to
                MahmoodLab/CONCH.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.CONCH_V1.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj = timm.create_model(
                model_path,
                pretrained=True,
                num_classes=0,
            )
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """448×448 preprocessing extracted from CONCH pretrained_cfg."""
        return _timm_preprocessing(self.obj)
