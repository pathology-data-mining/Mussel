"""GooglePath (Google Path Foundation) TensorFlow model."""

import logging
from typing import Callable, List

from mussel.models.base import Model
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


@register_model(ModelType.GOOGLEPATH)
class GooglePathModel(Model):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize GooglePath model.

        Args:
            model_path: Path to the model or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        import tensorflow as tf
        from huggingface_hub import from_pretrained_keras

        if model_path is None:
            model_path = ModelType.GOOGLEPATH.path

        if use_gpu and len(tf.config.list_physical_devices("GPU")) == 0:
            raise OSError("cuda not available")
        if use_gpu and gpu_device_id:
            if isinstance(gpu_device_id, list):
                devices = [
                    tf.config.list_physical_devices("GPU")[i] for i in gpu_device_id
                ]
            else:
                devices = tf.config.list_physical_devices("GPU")[gpu_device_id]
            tf.config.set_visible_devices(devices)

        model_obj = from_pretrained_keras(model_path)

        super().__init__(model_obj)

    def get_model_fun(self) -> Callable:
        """Get model inference function for GooglePath.

        Returns:
            Callable that preprocesses input and returns model output.
        """
        import tensorflow as tf

        def model_fun(x) -> Callable:
            """Preprocess and run inference for GooglePath model."""
            tensor = tf.cast(x, tf.float32) / 255.0
            tensor = tf.transpose(tensor, [0, 2, 3, 1])
            tensor = tf.image.resize(
                tensor, size=(224, 224), method=tf.image.ResizeMethod.BICUBIC
            )
            return self.obj(tensor)

        return model_fun

    def save(self, save_path: str):
        """Save GooglePath model (not implemented).

        Args:
            save_path: Path to save the model.

        Raises:
            NotImplementedError: GooglePath model saving is not yet implemented.
        """
        raise NotImplementedError("GooglePath model saving not implemented yet")
