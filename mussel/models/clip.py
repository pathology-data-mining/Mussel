"""CLIP (QuiltNet) model using open_clip."""

import logging
from functools import partial
from typing import Callable, List

import open_clip
import torch.nn as nn
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

from mussel.models.base import TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


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
            model_obj.forward = partial(model_obj.encode_image)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)
        if model_obj is None:
            # Loaded from a saved model file: build the eval transform that
            # create_model_and_transforms() returns for this model.
            self.preprocessing = self._file_model_preprocessing(self.obj)

    @staticmethod
    def _file_model_preprocessing(model_obj: nn.Module) -> Callable:
        """Eval transform for an open_clip model loaded from a saved model file.

        Args:
            model_obj: The loaded open_clip model (optionally DataParallel-wrapped).

        Returns:
            open_clip eval transform at the model's image size.
        """
        model = (
            model_obj.module if isinstance(model_obj, nn.DataParallel) else model_obj
        )
        visual = getattr(model, "visual", None)
        # open_clip stores the config its create_model_and_transforms() used on
        # model.visual.preprocess_cfg; older models may lack it.
        cfg = getattr(visual, "preprocess_cfg", None)
        cfg = cfg if isinstance(cfg, dict) else {}
        image_size = cfg.get("size", getattr(visual, "image_size", None))
        if isinstance(image_size, (tuple, list)):
            image_size = tuple(int(v) for v in image_size)
        elif not isinstance(image_size, int):
            image_size = 224
        return open_clip.image_transform(
            image_size,
            is_train=False,
            mean=cfg.get("mean", OPENAI_DATASET_MEAN),
            std=cfg.get("std", OPENAI_DATASET_STD),
            resize_mode=cfg.get("resize_mode"),
            interpolation=cfg.get("interpolation"),
            fill_color=cfg.get("fill_color", 0),
        )

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for CLIP.

        Returns:
            Preprocessing transforms from open_clip.
        """
        return self.preprocessing
