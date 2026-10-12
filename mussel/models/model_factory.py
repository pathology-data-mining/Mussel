"""Model registry, factory, and type definitions for all supported foundation models."""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mussel.models.base import Model

logger = logging.getLogger(__name__)


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
    HOPTIMUS1 = 16, "hoptimus1", "hf-hub:bioptimus/H-optimus-1"
    H0MINI = 17, "h0mini", "hf-hub:bioptimus/H0-mini"
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
    # MADELEINE was trained on CONCH v1.0 features (512-dim) of 256 px tiles at 10x
    # (MahmoodLab/madeleine model card).
    ModelType.MADELEINE_SLIDE: ModelType.CONCH_V1,
    # Add more slide encoder -> patch encoder mappings as they become available
}

# Recommended tile size (px, read at MODEL_TARGET_MPP where listed) for each model
# type: the tiles the authors cut from the slide, before the model's own resize/crop.
# A slide encoder's entry is the tiles its patch encoder's features must come from.
# Sources are noted where a size differs from the model's input size or from the
# common 224 px.
MODEL_PATCH_SIZES = {
    ModelType.RESNET50: 256,
    # CTransPath: no tile size given for inference (trained on 1024 px tiles at 20x);
    # its README recommends extracting features "at 1.0mpp" first. CHIEF's use of it
    # (256 px at 1.0 µm/px) is under CHIEF_SLIDE.
    ModelType.CTRANSPATH: 224,
    ModelType.GIGAPATH: 256,
    ModelType.VIRCHOW: 224,
    ModelType.VIRCHOW2: 224,
    ModelType.OPTIMUS: 224,
    ModelType.CLIP: 224,  # QuiltNet
    ModelType.GOOGLEPATH: 224,  # Google Path Foundation
    ModelType.CONCH1_5: 512,
    ModelType.UNI: 256,  # 256 px at 20x (MahmoodLab/UNI card, Chen et al. 2024)
    ModelType.UNI2H: 256,  # 256 px at 20x (MahmoodLab TRIDENT; the card gives none)
    # Slide encoders inherit from their patch encoders
    ModelType.GIGAPATH_SLIDE: 256,
    ModelType.TITAN_SLIDE: 512,
    ModelType.PRISM_SLIDE: 224,
    ModelType.FEATHER_SLIDE: 512,
    # CHIEF: CTransPath features of 256 px tiles at 10x, 1.0 µm/px (Wang et al. 2024).
    ModelType.CHIEF_SLIDE: 256,
    # MADELEINE: "CONCH features at 10x magnification on 256x256-pixel patches" (card).
    ModelType.MADELEINE_SLIDE: 256,
    ModelType.PHIKON: 224,
    ModelType.PHIKON_V2: 224,
    ModelType.HOPTIMUS1: 224,
    ModelType.H0MINI: 224,
    ModelType.MIDNIGHT12K: 224,
    # GPFM: "recommended original tile size ~512×512 (40×), ... resized to 224×224" (card).
    ModelType.GPFM: 512,
    ModelType.HIBOU_L: 224,
    # CONCH v1: no single tile spec from the authors (paper: 256 px at 10x for slide
    # tasks; TRIDENT: 512 px at 20x); 448 is its input size. Not MPP-checked.
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


# Resolution (µm/px) a model's tiles must be read at, where its authors specify
# one (20x = 0.5, 10x = 1.0, 40x = 0.25 µm/px). Models not listed here are not
# checked for MPP: they were trained on several magnifications with no inference
# resolution recommended (Virchow2, Kaiko, Lunit, Midnight, Path Foundation,
# GenBio-PathFM, QuiltNet), or their authors state none (CTransPath on its own,
# CONCH v1, OpenMidnight, Hibou-L, ResNet-50, ABMIL).
MODEL_TARGET_MPP = {
    # TITAN: CONCH v1.5 features of non-overlapping 512x512 px patches at 20x
    # (Ding et al. 2024; MahmoodLab/TITAN model card).
    ModelType.CONCH1_5: 0.5,
    ModelType.TITAN_SLIDE: 0.5,
    # FEATHER: "CONCH v1.5 patch features with patch size of 512x512 pixels at 20x" (card).
    ModelType.FEATHER_SLIDE: 0.5,
    # "expects images of size 224x224 that were extracted at 0.5 microns per pixel" (cards).
    ModelType.OPTIMUS: 0.5,
    ModelType.HOPTIMUS1: 0.5,
    # H0-mini: distilled on "224x224 at 20x magnification" tiles (Filiot et al. 2025).
    ModelType.H0MINI: 0.5,
    # UNI: 20x (card, paper); UNI2-h: 20x (MahmoodLab TRIDENT).
    ModelType.UNI: 0.5,
    ModelType.UNI2H: 0.5,
    # Prov-GigaPath: slides "resized ... to a standard resolution of 0.5 μm per pixel" (Xu et al. 2024).
    ModelType.GIGAPATH: 0.5,
    ModelType.GIGAPATH_SLIDE: 0.5,
    # Virchow (v1): "tiles sampled at 0.5 microns per pixel resolution (20x)" (card);
    # PRISM: Virchow v1 features of 224 px tiles at 0.5 µm/px (Shaikovski et al. 2024).
    ModelType.VIRCHOW: 0.5,
    ModelType.PRISM_SLIDE: 0.5,
    # Phikon: 0.5 µm/px (owkin/HistoSSLscaling); Phikon-v2: 20x, 0.5 µm/px (Filiot et al. 2024).
    ModelType.PHIKON: 0.5,
    ModelType.PHIKON_V2: 0.5,
    # CHIEF: 10x, 1.0 µm/px (Wang et al. 2024); MADELEINE: 10x (card).
    ModelType.CHIEF_SLIDE: 1.0,
    ModelType.MADELEINE_SLIDE: 1.0,
    # GPFM: ~512 px tiles at 40x (card).
    ModelType.GPFM: 0.25,
}

TILE_SIZE_CHECK_MODES = ("warn", "error", "off")


def check_tile_size(
    model_type: ModelType,
    tile_attrs: Optional[dict],
    mode: str = "warn",
    context: str = "",
    mpp_tolerance: float = 0.05,
) -> Optional[str]:
    """Check that tiles match the model's recommended tile size (and MPP, if known).

    Compares ``patch_size_to_resize_to_for_desired_mpp`` and ``mpp`` from a
    patch H5's ``coords`` attributes (written by ``tessellate``) with
    :data:`MODEL_PATCH_SIZES` and :data:`MODEL_TARGET_MPP`. A model run on
    tiles of another size still produces features, just not the ones it was
    trained to produce; e.g. TITAN fed CONCH v1.5 features of 224 px tiles.

    Args:
        model_type: Patch or slide encoder the tiles are for.
        tile_attrs: ``coords`` attributes of the patch H5 (None or missing keys skip the check).
        mode: ``"warn"`` logs a warning, ``"error"`` raises ``ValueError``,
            ``"off"`` skips the check.
        context: Text prepended to the message (e.g. the slide ID).
        mpp_tolerance: Relative MPP difference tolerated.

    Returns:
        The mismatch message, or None if the tiles match or cannot be checked.
    """
    if mode not in TILE_SIZE_CHECK_MODES:
        raise ValueError(f"tile_size_check must be one of {TILE_SIZE_CHECK_MODES}, got {mode!r}")
    if mode == "off" or not tile_attrs:
        return None
    problems = []
    tile_px = tile_attrs.get("patch_size_to_resize_to_for_desired_mpp")
    expected_px = MODEL_PATCH_SIZES.get(model_type)
    if tile_px is not None and expected_px is not None and int(tile_px) != expected_px:
        problems.append(f"tiles are {int(tile_px)} px, recommended {expected_px} px")
    tile_mpp = tile_attrs.get("mpp")
    expected_mpp = MODEL_TARGET_MPP.get(model_type)
    if (
        tile_mpp is not None
        and expected_mpp is not None
        and abs(float(tile_mpp) - expected_mpp) > mpp_tolerance * expected_mpp
    ):
        problems.append(f"tiles are at {float(tile_mpp):g} µm/px, recommended {expected_mpp:g}")
    if not problems:
        return None
    name = getattr(model_type, "name", str(model_type))
    message = (
        f"{context + ': ' if context else ''}{name} {'; '.join(problems)}. "
        "Re-tessellate at the recommended size, or set tile_size_check=off if this is intended."
    )
    if mode == "error":
        raise ValueError(message)
    logger.warning(message)
    return message


def recommended_patch_sizes() -> dict:
    """Recommended tile size per model code, e.g. {"titan_slide": 512, ...}."""
    return {model_type.code: size for model_type, size in MODEL_PATCH_SIZES.items()}


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
    def get_model(self, model_path, use_gpu, gpu_device_id, **kwargs) -> "Model":
        """Get a model instance.

        Args:
            model_path: Path to model weights or config.
            use_gpu: Whether to use GPU.
            gpu_device_id: GPU device ID or list of IDs.
            **kwargs: Extra keyword arguments forwarded to the model constructor.

        Returns:
            Model instance.
        """
        pass


class _SimpleModelFactory(ModelFactory):
    """Generic factory that instantiates any registered Model class."""

    def __init__(self, model_cls):
        self._cls = model_cls

    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None, **kwargs) -> "Model":
        return self._cls(model_path, use_gpu, gpu_device_id, **kwargs)


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
