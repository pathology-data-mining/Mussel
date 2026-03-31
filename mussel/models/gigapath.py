"""Prov-GigaPath tile and slide encoder models."""

import logging
import os
import tempfile
from typing import Callable, List

import timm
import torch
from torchvision import transforms

from mussel.models.base import IMAGENET_MEAN, IMAGENET_STD, TorchModel

try:
    from mussel.utils.model_cache import model_download_lock
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def model_download_lock(model_name, **kwargs):
        yield True


from mussel.models.model_factory import ModelType, register_model

# Import gigapath.slide_encoder early to register models with timm
try:
    import gigapath.slide_encoder
except ImportError:
    pass  # GigaPath not installed

logger = logging.getLogger(__name__)


@register_model(ModelType.GIGAPATH)
class GigapathModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize Prov-GigaPath model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.GIGAPATH.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj = timm.create_model(model_path, pretrained=True)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for Prov-GigaPath.

        Returns:
            Composed transforms for Prov-GigaPath input preprocessing.
        """
        return transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.obj(x)  # (N, num_tokens, embed_dim) or (N, embed_dim)
        if output.dim() == 3:
            output = output[:, 0]  # CLS token
        return output


@register_model(ModelType.GIGAPATH_SLIDE)
class GigapathSlideEncoderModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize Prov-GigaPath slide encoder model.

        This is the slide-level encoder from Prov-GigaPath that aggregates
        patch-level features into slide-level representations.

        Args:
            model_path: Path to slide encoder model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.GIGAPATH_SLIDE.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            # Load GigaPath slide encoder using their official API
            # See: https://github.com/prov-gigapath/prov-gigapath#inference-with-the-slide-encoder
            try:
                import gigapath.slide_encoder
            except ImportError as e:
                raise RuntimeError(
                    f"Failed to import gigapath.slide_encoder: {e}\n"
                    "The GigaPath slide encoder requires flash-attn, which must be "
                    "compiled for the CUDA version on your system. "
                    "Install the 'fastattn' extra built for your CUDA version, e.g.:\n"
                    "  uv sync --extra fastattn\n"
                    "or run without the GigaPath slide encoder."
                ) from e

            # gigapath expects "hf_hub:" format (underscore, not hyphen)
            model_path_fixed = model_path.replace("hf-hub:", "hf_hub:")

            # Use a writable local_dir for downloading
            # Try HF_HOME first, then fall back to temp directory
            local_dir = os.environ.get(
                "HF_HOME",
                os.environ.get("TRANSFORMERS_CACHE", tempfile.gettempdir()),
            )

            # Use locking when downloading from HuggingFace
            with model_download_lock(model_path) as should_download:
                # create_model(repo_id, model_name, in_chans, local_dir)
                # in_chans=1536 for GigaPath tile embeddings
                model_obj = gigapath.slide_encoder.create_model(
                    model_path_fixed,
                    "gigapath_slide_enc12l768d",
                    1536,
                    local_dir=local_dir,
                )
        else:
            raise ValueError(
                f"GIGAPATH_SLIDE only supports loading from HuggingFace hub. "
                f"Got model_path: {model_path}. "
                f"Expected format: 'hf-hub:prov-gigapath/prov-gigapath'"
            )
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_model_fun(self) -> Callable:
        """Get model inference function for GigaPath slide encoder.

        The GigaPath slide encoder uses the slide_encoder method which
        requires both tile embeddings and coordinates as arguments.

        Returns:
            Callable that takes tile embeddings and coordinates, returns slide-level features
            with shape (768,).
        """

        def model_fun(features, coords):
            """Run inference with mixed precision."""
            with (
                torch.no_grad(),
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type, dtype=torch.float16),
            ):
                features = features.to(self.device, non_blocking=True)
                coords = coords.to(self.device, non_blocking=True)
                # GigaPath slide encoder API: output = model(tile_embed, coordinates)
                # Returns a list with the tensor as first element
                result = self.obj(features, coords)
                if isinstance(result, list):
                    result = result[0]
                return result.squeeze(0).float().cpu()

        return model_fun

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing function for slide encoder.

        Slide encoders work on patch features, not images, so no preprocessing needed.

        Returns:
            None, as slide encoders don't preprocess images.
        """
        return None

    def save(self, save_path: str):
        """Cache GigaPath slide encoder from HuggingFace hub.

        This triggers the model to be downloaded and cached in the HuggingFace cache directory.
        The model cannot be saved to a local file, but this ensures it's available in the cache
        for future use.

        Args:
            save_path: Ignored. Model is cached in HuggingFace cache directory.
        """
        logger.info(
            f"GIGAPATH_SLIDE model cached from HuggingFace hub. "
            f"The model is stored in HuggingFace cache and will be reused automatically."
        )
