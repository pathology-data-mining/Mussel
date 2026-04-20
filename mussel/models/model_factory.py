"""Model registry, factory, and type definitions for all supported foundation models."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mussel.models.base import Model


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
    UNI2H = 10, "uni2h", "hf-hub:MahmoodLab/UNI2-h"
    UNI = 11, "uni", "hf-hub:MahmoodLab/UNI"
    GIGAPATH_SLIDE = 12, "gigapath_slide", "hf-hub:prov-gigapath/prov-gigapath"
    TITAN_SLIDE = 13, "titan_slide", "MahmoodLab/TITAN"
    PHIKON = 14, "phikon", "hf-hub:owkin/phikon"
    PHIKON_V2 = 15, "phikon_v2", "hf-hub:owkin/phikon-v2"
    H_OPTIMUS_1 = 16, "hoptimus1", "hf-hub:bioptimus/H-optimus-1"
    H0_MINI = 17, "h0mini", "hf-hub:bioptimus/H0-mini"
    MIDNIGHT12K = 18, "midnight12k", "hf-hub:kaiko-ai/midnight"
    GPFM = 19, "gpfm", "hf-hub:majiabo/GPFM"
    HIBOU_L = 20, "hibou_l", "histai/hibou-L"
    PRISM_SLIDE = 21, "prism_slide", "hf-hub:paige-ai/Prism"
    FEATHER_SLIDE = 22, "feather_slide", "MahmoodLab/abmil.base.conch_v15.pc108-24k"
    CHIEF_SLIDE = 23, "chief_slide", ""
    MADELEINE_SLIDE = 24, "madeleine_slide", "MahmoodLab/madeleine"
    CONCH_V1 = 25, "conch_v1", "hf-hub:MahmoodLab/CONCH"
    KAIKO_VITS8 = (
        26,
        "kaiko_vits8",
        "hf-hub:1aurent/vit_small_patch8_224.kaiko_ai_towards_large_pathology_fms",
    )
    KAIKO_VITS16 = (
        27,
        "kaiko_vits16",
        "hf-hub:1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms",
    )
    KAIKO_VITB8 = (
        28,
        "kaiko_vitb8",
        "hf-hub:1aurent/vit_base_patch8_224.kaiko_ai_towards_large_pathology_fms",
    )
    KAIKO_VITB16 = (
        29,
        "kaiko_vitb16",
        "hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
    )
    KAIKO_VITL14 = (
        30,
        "kaiko_vitl14",
        "hf-hub:1aurent/vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms",
    )
    LUNIT_VITS8 = 31, "lunit_vits8", "hf-hub:1aurent/vit_small_patch8_224.lunit_dino"
    LUNIT_VITS16 = 32, "lunit_vits16", "hf-hub:1aurent/vit_small_patch16_224.lunit_dino"
    OPENMIDNIGHT = 33, "openmidnight", "SophontAI/OpenMidnight"
    GENBIO_PATHFM = 34, "genbio_pathfm", "genbio-ai/genbio-pathfm"
    ABMIL_SLIDE = 35, "abmil_slide", ""


# Mapping of slide encoder models to their compatible patch encoder models
SLIDE_ENCODER_COMPATIBILITY = {
    ModelType.GIGAPATH_SLIDE: ModelType.GIGAPATH,
    ModelType.TITAN_SLIDE: ModelType.CONCH1_5,
    ModelType.PRISM_SLIDE: ModelType.VIRCHOW,
    ModelType.FEATHER_SLIDE: ModelType.CONCH1_5,
    ModelType.CHIEF_SLIDE: ModelType.CTRANSPATH,
    # MADELEINE was trained with CONCH v1.0 (512-dim) features; CLIP is the
    # available 512-dim encoder in ModelType and is used here as a dimension proxy.
    ModelType.MADELEINE_SLIDE: ModelType.CLIP,
    # Add more slide encoder -> patch encoder mappings as they become available
}

# Recommended patch sizes for each model type
MODEL_PATCH_SIZES = {
    ModelType.RESNET50: 256,
    ModelType.CTRANSPATH: 224,
    ModelType.GIGAPATH: 256,
    ModelType.VIRCHOW: 224,
    ModelType.VIRCHOW2: 224,
    ModelType.OPTIMUS: 224,
    ModelType.CLIP: 224,  # QuiltNet
    ModelType.GOOGLEPATH: 224,  # Google Path Foundation
    ModelType.CONCH1_5: 512,
    ModelType.UNI: 256,
    ModelType.UNI2H: 256,
    # Slide encoders inherit from their patch encoders
    ModelType.GIGAPATH_SLIDE: 256,
    ModelType.TITAN_SLIDE: 512,
    ModelType.PRISM_SLIDE: 224,
    ModelType.FEATHER_SLIDE: 512,
    ModelType.CHIEF_SLIDE: 224,
    ModelType.MADELEINE_SLIDE: 512,
    ModelType.PHIKON: 224,
    ModelType.PHIKON_V2: 224,
    ModelType.H_OPTIMUS_1: 224,
    ModelType.H0_MINI: 224,
    ModelType.MIDNIGHT12K: 224,
    ModelType.GPFM: 224,
    ModelType.HIBOU_L: 224,
    ModelType.CONCH_V1: 448,
    ModelType.KAIKO_VITS8: 224,
    ModelType.KAIKO_VITS16: 224,
    ModelType.KAIKO_VITB8: 224,
    ModelType.KAIKO_VITB16: 224,
    ModelType.KAIKO_VITL14: 224,
    ModelType.LUNIT_VITS8: 224,
    ModelType.LUNIT_VITS16: 224,
    ModelType.OPENMIDNIGHT: 224,
    ModelType.GENBIO_PATHFM: 224,
    # ABMIL slide encoder: encoder-agnostic; default matches common patch encoders
    ModelType.ABMIL_SLIDE: 256,
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


def get_default_patch_size(model_type: ModelType) -> int:
    """Get the recommended default patch size for a model type.

    Args:
        model_type: The model type to get the patch size for.

    Returns:
        The recommended patch size in pixels.

    Raises:
        ValueError: If the model type is not recognized.
    """
    if model_type not in MODEL_PATCH_SIZES:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available model types: {list(MODEL_PATCH_SIZES.keys())}"
        )

    return MODEL_PATCH_SIZES[model_type]


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


# Registry mapping ModelType -> Model class, populated via @register_model decorators
MODEL_REGISTRY: dict = {}

# Backward-compatible alias for any code that still references MODEL_FACTORIES
MODEL_FACTORIES = MODEL_REGISTRY


def register_model(model_type: ModelType):
    """Decorator to register a Model class for a given ModelType.

    Args:
        model_type: The ModelType to associate with the decorated class.

    Returns:
        Decorator function that stores the class in MODEL_REGISTRY.
    """

    def decorator(cls):
        MODEL_REGISTRY[model_type] = cls
        return cls

    return decorator


class ModelFactory(ABC):
    @abstractmethod
    def get_model(self, model_path, use_gpu, gpu_device_id) -> "Model":
        """Get a model instance.

        Args:
            model_path: Path to model weights or config.
            use_gpu: Whether to use GPU.
            gpu_device_id: GPU device ID or list of IDs.

        Returns:
            Model instance.
        """
        pass


class _SimpleModelFactory(ModelFactory):
    """Generic factory that instantiates any registered Model class."""

    def __init__(self, model_cls):
        self._cls = model_cls

    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None) -> "Model":
        return self._cls(model_path, use_gpu, gpu_device_id)


def get_model_factory(
    model_type: "ModelType | str" = ModelType.CTRANSPATH,
) -> ModelFactory:
    """Get the model factory for a given model type.

    Args:
        model_type: ModelType enum or string name of the model (default: ModelType.CTRANSPATH).

    Returns:
        ModelFactory instance for the specified model type.

    Raises:
        ValueError: If model_type string is not recognized or has no registered factory.
    """
    if isinstance(model_type, str):
        try:
            model_type = ModelType[model_type.upper()]
        except KeyError:
            raise ValueError(f"unknown model type: {model_type}")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"no factory registered for model type: {model_type}")
    return _SimpleModelFactory(MODEL_REGISTRY[model_type])
