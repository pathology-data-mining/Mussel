import json
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable, List

import open_clip
import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torchvision import transforms
from transformers import AutoModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGENET_STD = [0.229, 0.224, 0.225]


class ModelType(Enum):
    def __init__(self, id, code, path):
        """Initialize a ModelType enum value.

        Args:
            id: Unique integer identifier for the model type.
            code: String code for the model type.
            path: Path or identifier to load the model from.
        """
        self.id = id
        self.code = code
        self.path = path

    RESNET50 = 1, "resnet50", ""
    CTRANSPATH = 2, "ctranspath", ""
    GIGAPATH = 3, "gigapath", "hf-hub:prov-gigapath/prov-gigapath"
    VIRCHOW = 4, "virchow", "hf-hub:paige-ai/Virchow"
    OPTIMUS = 5, "optimus", "hf-hub:bioptimus/H-optimus-0"
    CLIP = 6, "clip", "hf-hub:wisdomik/QuiltNet-B-16-PMB"
    GOOGLEPATH = 7, "googlepath", "google/path-foundation"
    CONCH1_5 = 8, "conch1_5", "MahmoodLab/TITAN"
    VIRCHOW2 = 9, "virchow2", "hf-hub:paige-ai/Virchow2"
    GIGAPATH_SLIDE = 10, "gigapath_slide", "hf-hub:prov-gigapath/prov-gigapath"
    TITAN_SLIDE = 11, "titan_slide", "MahmoodLab/TITAN"


# Mapping of slide encoder models to their compatible patch encoder models
SLIDE_ENCODER_COMPATIBILITY = {
    ModelType.GIGAPATH_SLIDE: ModelType.GIGAPATH,
    ModelType.TITAN_SLIDE: ModelType.CONCH1_5,
    # Add more slide encoder -> patch encoder mappings as they become available
}


def get_required_patch_encoder(slide_encoder: ModelType) -> ModelType:
    """Get the required patch encoder for a given slide encoder.

    Each slide encoder model is designed to work with features from a specific
    patch encoder model. This function returns the required patch encoder.

    Args:
        slide_encoder: The slide-level encoder model type.

    Returns:
        The required patch encoder model type.

    Raises:
        ValueError: If the slide encoder is not recognized.
    """
    if slide_encoder not in SLIDE_ENCODER_COMPATIBILITY:
        raise ValueError(
            f"Unknown slide encoder: {slide_encoder}. "
            f"Available slide encoders: {list(SLIDE_ENCODER_COMPATIBILITY.keys())}"
        )

    return SLIDE_ENCODER_COMPATIBILITY[slide_encoder]


def validate_slide_encoder_compatibility(
    patch_encoder: ModelType, slide_encoder: ModelType
) -> bool:
    """Validate that a slide encoder is compatible with a patch encoder.

    Each slide encoder model is designed to work with features from a specific
    patch encoder model. This function validates that the combination is valid.

    Args:
        patch_encoder: The patch-level encoder model type.
        slide_encoder: The slide-level encoder model type.

    Returns:
        True if the slide encoder is compatible with the patch encoder.

    Raises:
        ValueError: If the slide encoder is not compatible with the patch encoder.
    """
    if slide_encoder not in SLIDE_ENCODER_COMPATIBILITY:
        raise ValueError(
            f"Unknown slide encoder: {slide_encoder}. "
            f"Available slide encoders: {list(SLIDE_ENCODER_COMPATIBILITY.keys())}"
        )

    expected_patch_encoder = SLIDE_ENCODER_COMPATIBILITY[slide_encoder]
    if patch_encoder != expected_patch_encoder:
        raise ValueError(
            f"Slide encoder {slide_encoder} requires patch encoder {expected_patch_encoder}, "
            f"but {patch_encoder} was provided. Each slide encoder is tied to a specific "
            f"patch encoder model."
        )

    return True


class Model:
    def __init__(
        self,
        model_obj,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize base Model wrapper.

        Args:
            model_obj: The underlying model object.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        self.obj = model_obj

    def get_model_fun(self) -> Callable:
        """Get a callable function for model inference.

        Returns:
            Callable that takes input and returns model output.
        """
        return self.obj

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing function for input data.

        Returns:
            Callable for preprocessing or None if no preprocessing needed.
        """
        return None

    def save(self, save_path: str):
        """Save the model to disk.

        Args:
            save_path: Path to save the model.
        """
        pass


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


class TorchModel(Model):
    def __init__(
        self,
        model_path,
        model_obj=None,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize PyTorch model wrapper.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            model_obj: Optional pre-loaded model object (default: None).
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        self.use_gpu = use_gpu
        if model_obj is None:
            if model_path.startswith("hf-hub:"):
                repo_id = model_path.replace("hf-hub:", "")
                config_file = hf_hub_download(
                    repo_id=repo_id,
                    filename="config.json",
                )
                with open(config_file, "r") as f:
                    config = json.load(f)
                model_name = config.get("model_name", None)
                if not model_name:
                    raise ValueError(
                        f"model_name not found in config.json from {repo_id}"
                    )
                model_obj = timm.create_model(model_name, pretrained=True)
            elif Path(model_path).is_file():
                with open(model_path, "rb") as f:
                    model_obj = pickle.load(f)
            else:
                raise ValueError(f"invalid model_path: {model_path}")
        super().__init__(model_obj)
        if use_gpu and not torch.cuda.is_available():
            raise OSError("cuda not available")

        device_type = "cuda" if use_gpu else "cpu"
        device_id = (
            gpu_device_id[0]
            if isinstance(gpu_device_id, list) and len(gpu_device_id) > 0
            else gpu_device_id
        )
        self.device = (
            torch.device(device_type, device_id)
            if device_id is not None
            else torch.device(device_type)
        )

        if isinstance(gpu_device_id, list) and len(gpu_device_id) > 1:
            self.obj = nn.DataParallel(self.obj, device_ids=gpu_device_id)
        self.obj = self.obj.to(self.device)
        self.obj.eval()

    def get_model_fun(self) -> Callable:
        """Get model inference function with automatic mixed precision.

        Returns:
            Callable that runs inference on GPU or CPU with autocast.
        """

        def model_fun(x):
            """Run inference with mixed precision."""
            with (
                torch.no_grad(),
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type, dtype=torch.float16),
            ):
                x = x.to(self.device, non_blocking=True)
                return self.obj(x).cpu()

        return model_fun

    def save(self, save_path: str):
        """Save PyTorch model to disk using pickle.

        Args:
            save_path: Path to save the model.
        """
        with open(save_path, "wb") as f:
            pickle.dump(self.obj, f)


class Conch15Model(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize CONCH v1.5 model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.CONCH1_5.path
        model_obj = None
        if not Path(model_path).is_file():
            titan = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            model_obj, _ = titan.return_conch()
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for CONCH v1.5.

        Returns:
            Composed transforms for CONCH v1.5 input preprocessing.
        """
        preprocessing = transforms.Compose(
            [
                transforms.Resize(
                    448,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return preprocessing


class TitanSlideEncoderModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize TITAN slide encoder model.

        This is the slide-level encoder from MahmoodLab/TITAN that aggregates
        CONCH patch-level features into slide-level representations.

        Args:
            model_path: Path to slide encoder model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.TITAN_SLIDE.path
        model_obj = None
        if not Path(model_path).is_file():
            # Load the TITAN model - we'll use the whole model
            # and call encode_slide_from_patch_features on it
            model_obj = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_model_fun(self) -> Callable:
        """Get model inference function for TITAN slide encoder.

        The TITAN slide encoder uses encode_slide_from_patch_features method
        which requires patch features, coordinates, and patch size at level 0.

        Returns:
            Callable that takes patch features, coords, and patch_size, returns slide-level features.
        """

        def model_fun(patch_features, coords, patch_size):
            """Run TITAN slide encoder on patch features with coordinates and patch size."""
            with (
                torch.no_grad(),
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type, dtype=torch.float16),
            ):
                patch_features = patch_features.to(self.device, non_blocking=True)
                coords = coords.to(self.device, non_blocking=True)
                return self.obj.encode_slide_from_patch_features(
                    patch_features, coords, patch_size
                ).cpu()

        return model_fun

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing function for slide encoder.

        Slide encoders work on patch features, not images, so no preprocessing needed.

        Returns:
            None, as slide encoders don't preprocess images.
        """
        return None


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
        preprocessing = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return preprocessing


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
            # Load the full GigaPath model which includes the slide encoder
            repo_id = model_path.replace("hf-hub:", "")
            model_obj = timm.create_model(repo_id, pretrained=True)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_model_fun(self) -> Callable:
        """Get model inference function for GigaPath slide encoder.

        The GigaPath slide encoder uses the slide_encoder method which
        requires both tile embeddings and coordinates as arguments.

        Returns:
            Callable that takes tile embeddings and coordinates, returns slide-level features.
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
                return self.obj(features, coords)[0].cpu()

        return model_fun

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing function for slide encoder.

        Slide encoders work on patch features, not images, so no preprocessing needed.

        Returns:
            None, as slide encoders don't preprocess images.
        """
        return None


class OptimusModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize H-Optimus-0 model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.OPTIMUS.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj = timm.create_model(
                model_path,
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False,
            )
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for H-Optimus-0.

        Returns:
            Composed transforms for H-Optimus-0 input preprocessing.
        """
        preprocessing = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )
        return preprocessing


class VirchowModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize Virchow model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.VIRCHOW.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj = timm.create_model(
                model_path,
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for Virchow.

        Returns:
            Preprocessing transforms resolved from model config.
        """
        preprocessing = create_transform(
            **resolve_data_config(self.obj.pretrained_cfg, model=self.obj)
        )
        return preprocessing


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

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for CLIP.

        Returns:
            Preprocessing transforms from open_clip.
        """
        return self.preprocessing


class TransPathModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize TransPath model.

        Args:
            model_path: Path to model weights file.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).

        Raises:
            ValueError: If model_path is not provided.
        """
        if model_path is None:
            raise ValueError("model_path must be provided for TransPath model")
        from transpath.ctran import ctranspath

        model_obj = ctranspath()
        model_obj.head = nn.Identity()
        td = torch.load(model_path, weights_only=True)
        model_obj.load_state_dict(td["model"], strict=True)
        # ctranspath() module has required torch transforms built in so
        # preprocessing should be None here
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)


class ResnetModel(TorchModel):
    def __init__(
        self, use_gpu: bool = True, gpu_device_id: int | List[int] | None = None
    ):
        """Initialize ResNet-50 model.

        Args:
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        from mussel.models.resnet_custom import resnet50_baseline

        model_obj = resnet50_baseline(pretrained=True)
        super().__init__(None, model_obj, use_gpu, gpu_device_id)


MODEL_FACTORIES = {}


def register_model_factory(model_type: ModelType):
    """Decorator to register a model factory for a given model type.

    Args:
        model_type: The ModelType to register the factory for.

    Returns:
        Decorator function that registers the factory.
    """

    def decorator(fn):
        """Register factory function."""
        MODEL_FACTORIES[model_type] = fn
        return fn

    return decorator


class ModelFactory(ABC):
    @abstractmethod
    def get_model(self, model_path, use_gpu, gpu_device_id) -> Model:
        """Get a model instance.

        Args:
            model_path: Path to model weights or config.
            use_gpu: Whether to use GPU.
            gpu_device_id: GPU device ID or list of IDs.

        Returns:
            Model instance.
        """
        pass


@register_model_factory(ModelType.GOOGLEPATH)
class GooglePathModelFactory(ModelFactory):
    def get_model(self, model_path, use_gpu=True, gpu_device_id=None) -> Model:
        """Create GooglePath model instance."""
        return GooglePathModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.RESNET50)
class Resnet50ModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create ResNet-50 model instance."""
        return ResnetModel(use_gpu, gpu_device_id)


@register_model_factory(ModelType.CTRANSPATH)
class CTransPathModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create TransPath model instance."""
        return TransPathModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.GIGAPATH)
class GigapathModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create Prov-GigaPath model instance."""
        return GigapathModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.GIGAPATH_SLIDE)
class GigapathSlideEncoderModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create Prov-GigaPath slide encoder model instance."""
        return GigapathSlideEncoderModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.VIRCHOW)
class VirchowModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create Virchow model instance."""
        return VirchowModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.VIRCHOW2)
class Virchow2ModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create Virchow2 model instance."""
        return VirchowModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.CONCH1_5)
class Conch15ModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create CONCH v1.5 model instance."""
        return Conch15Model(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.TITAN_SLIDE)
class TitanSlideEncoderModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create TITAN slide encoder model instance."""
        return TitanSlideEncoderModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.OPTIMUS)
class OptimusModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create H-Optimus-0 model instance."""
        return OptimusModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.CLIP)
class ClipModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        """Create CLIP (QuiltNet) model instance."""
        return ClipModel(model_path, use_gpu, gpu_device_id)


def get_model_factory(
    model_type: ModelType | str = ModelType.CTRANSPATH,
) -> ModelFactory:
    """Get the model factory for a given model type.

    Args:
        model_type: ModelType enum or string name of the model (default: ModelType.CTRANSPATH).

    Returns:
        ModelFactory instance for the specified model type.

    Raises:
        ValueError: If model_type string is not recognized.
    """
    if isinstance(model_type, str):
        try:
            model_type = ModelType[model_type.upper()]
        except KeyError:
            raise ValueError(f"unknown model type: {model_type}")
    return MODEL_FACTORIES.get(model_type)()
