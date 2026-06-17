"""Artifact and pen-mark removal for whole-slide image segmentation masks.

Provides :class:`GrandQCArtifactRemover`, a concrete implementation of the
``artifact_remover_fn(img, mask, mpp) -> mask`` hook accepted by
:func:`mussel.utils.segment.segment_tissue`.

The implementation is based on the GrandQC model (Nature 2024):
    https://www.nature.com/articles/s41467-024-54769-y

Model weights are downloaded automatically from
``MahmoodLab/hest-tissue-seg`` on HuggingFace.

Requires the ``torch-gpu`` or ``torch-cpu`` extra::

    uv sync --extra torch-gpu

Class constants and preset exclusion sets are exported at module level::

    from mussel.utils.artifact_removal import (
        CLASS_PEN_MARKING, CLASS_BACKGROUND,
        EXCLUDE_PENMARKS_ONLY, EXCLUDE_ALL_ARTIFACTS,
        GrandQCArtifactRemover,
    )
"""

from __future__ import annotations

import logging
import warnings
from typing import FrozenSet, Optional

import numpy as np

logger = logging.getLogger(__name__)

# GrandQC model constants
_GRANDQC_REPO_ID = "MahmoodLab/hest-tissue-seg"
_GRANDQC_CKPT = "GrandQC_MPP1_state_dict.pth"
_GRANDQC_ENCODER = "timm-efficientnet-b0"
_GRANDQC_TARGET_MPP = 1.0  # model trained at 10x ≈ 1 µm/px
_GRANDQC_TILE_SIZE = 512
_GRANDQC_N_CLASSES = 8

# ---------------------------------------------------------------------------
# Public GrandQC output class indices (0-indexed, as labelled in the paper).
# ---------------------------------------------------------------------------

#: Glass / clear-slide background.
CLASS_GLASS: int = 0
#: Normal (stained) tissue.
CLASS_NORMAL_TISSUE: int = 1
#: Blood / haemorrhage.
CLASS_BLOOD: int = 2
#: Necrosis.
CLASS_NECROSIS: int = 3
#: Pen marking.
CLASS_PEN_MARKING: int = 4
#: Tissue fold.
CLASS_FOLD: int = 5
#: Hole or physical slide damage.
CLASS_HOLE: int = 6
#: Non-tissue background.
CLASS_BACKGROUND: int = 7

# ---------------------------------------------------------------------------
# Preset exclusion sets — pass as ``exclude_classes`` to GrandQCArtifactRemover.
# ---------------------------------------------------------------------------

#: Conservative: remove only pen markings and background.
#: Keeps blood, necrosis, folds and holes as tissue.
#: Recommended default to avoid removing out-of-distribution tissue types.
EXCLUDE_PENMARKS_ONLY: FrozenSet[int] = frozenset(
    {CLASS_PEN_MARKING, CLASS_BACKGROUND}
)

#: Moderate: also remove folds and physical damage, but keep blood and necrosis.
EXCLUDE_FOLDS_AND_PENMARKS: FrozenSet[int] = frozenset(
    {CLASS_PEN_MARKING, CLASS_FOLD, CLASS_HOLE, CLASS_BACKGROUND}
)

#: Aggressive: remove all non-normal-tissue classes.
#: May over-remove tissue in slides with significant necrosis or haemorrhage
#: (e.g. glioblastoma, high-grade sarcoma).
EXCLUDE_ALL_ARTIFACTS: FrozenSet[int] = frozenset(
    {
        CLASS_BLOOD,
        CLASS_NECROSIS,
        CLASS_PEN_MARKING,
        CLASS_FOLD,
        CLASS_HOLE,
        CLASS_BACKGROUND,
    }
)

# Private aliases kept for internal backward compat.
_CLASS_NORMAL_TISSUE = CLASS_NORMAL_TISSUE
_CLASS_PEN_MARKING = CLASS_PEN_MARKING
_CLASS_BACKGROUND = CLASS_BACKGROUND


class GrandQCArtifactRemover:
    """Removes artifacts and/or pen marks from a binary tissue mask.

    Uses the GrandQC U-Net (EfficientNet-B0 encoder) to classify each pixel
    of the slide thumbnail, then zeros out any masked region whose predicted
    class is in ``exclude_classes``.

    Implements the ``artifact_remover_fn(img, mask, mpp) -> mask`` protocol
    expected by :func:`~mussel.utils.segment.segment_tissue`.

    GrandQC output classes (0-indexed):
        0 – :data:`CLASS_GLASS`         — glass / clear-slide background
        1 – :data:`CLASS_NORMAL_TISSUE` — normal stained tissue
        2 – :data:`CLASS_BLOOD`         — blood / haemorrhage
        3 – :data:`CLASS_NECROSIS`      — necrosis
        4 – :data:`CLASS_PEN_MARKING`   — pen marking
        5 – :data:`CLASS_FOLD`          — tissue fold
        6 – :data:`CLASS_HOLE`          — hole / physical slide damage
        7 – :data:`CLASS_BACKGROUND`    — non-tissue background

    Args:
        exclude_classes: Set of GrandQC class indices to remove from the
            tissue mask.  Defaults to :data:`EXCLUDE_PENMARKS_ONLY`
            ``{CLASS_PEN_MARKING, CLASS_BACKGROUND}`` — a conservative
            setting that keeps blood, necrosis and folds as tissue.
            Use :data:`EXCLUDE_ALL_ARTIFACTS` for aggressive removal, or
            build a custom set from the ``CLASS_*`` constants.
        device: Torch device string (e.g. ``"cuda"``, ``"cpu"``).  Defaults
            to CUDA if available.
        batch_size: Number of 512 × 512 tiles to process per forward pass.
        max_input_mpp: Input thumbnails coarser than this µm/px value are
            rejected (mask returned unchanged).  Default ``8.0``.
        remove_penmarks_only: **Deprecated** — use ``exclude_classes``
            instead.  ``True`` maps to :data:`EXCLUDE_PENMARKS_ONLY`;
            ``False`` maps to :data:`EXCLUDE_ALL_ARTIFACTS`.

    Examples::

        from mussel.utils.artifact_removal import (
            GrandQCArtifactRemover,
            EXCLUDE_PENMARKS_ONLY,
            EXCLUDE_FOLDS_AND_PENMARKS,
            EXCLUDE_ALL_ARTIFACTS,
            CLASS_BLOOD, CLASS_PEN_MARKING, CLASS_BACKGROUND,
        )

        # Conservative (default): pen marks + background only
        remover = GrandQCArtifactRemover()

        # Moderate: also remove folds and holes
        remover = GrandQCArtifactRemover(exclude_classes=EXCLUDE_FOLDS_AND_PENMARKS)

        # Aggressive: everything except normal tissue
        remover = GrandQCArtifactRemover(exclude_classes=EXCLUDE_ALL_ARTIFACTS)

        # Custom: blood and pen marks only
        remover = GrandQCArtifactRemover(
            exclude_classes=frozenset({CLASS_BLOOD, CLASS_PEN_MARKING, CLASS_BACKGROUND})
        )
    """

    def __init__(
        self,
        exclude_classes: Optional[FrozenSet[int]] = None,
        *,
        device: Optional[str] = None,
        batch_size: int = 8,
        max_input_mpp: float = 8.0,
        # Deprecated parameter kept for backward compatibility.
        remove_penmarks_only: Optional[bool] = None,
    ) -> None:
        if remove_penmarks_only is not None:
            warnings.warn(
                "GrandQCArtifactRemover: 'remove_penmarks_only' is deprecated and will be "
                "removed in a future release.  Use 'exclude_classes' instead:\n"
                "  remove_penmarks_only=True  → exclude_classes=EXCLUDE_PENMARKS_ONLY\n"
                "  remove_penmarks_only=False → exclude_classes=EXCLUDE_ALL_ARTIFACTS",
                DeprecationWarning,
                stacklevel=2,
            )
            if exclude_classes is None:
                exclude_classes = (
                    EXCLUDE_PENMARKS_ONLY
                    if remove_penmarks_only
                    else EXCLUDE_ALL_ARTIFACTS
                )

        self.exclude_classes: FrozenSet[int] = (
            frozenset(exclude_classes)
            if exclude_classes is not None
            else EXCLUDE_PENMARKS_ONLY
        )
        self.batch_size = batch_size
        self.max_input_mpp = max_input_mpp
        self._device: Optional[str] = device
        self._model = None
        self._transforms = None

    @property
    def device(self) -> str:
        if self._device is not None:
            return self._device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_model(self) -> None:
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "segmentation-models-pytorch is required for GrandQCArtifactRemover. "
                "Install the torch-gpu or torch-cpu extra: uv sync --extra torch-gpu"
            ) from exc

        import torch
        from huggingface_hub import hf_hub_download
        from torchvision import transforms

        logger.info("Downloading GrandQC weights from %s ...", _GRANDQC_REPO_ID)
        ckpt_path = hf_hub_download(
            repo_id=_GRANDQC_REPO_ID,
            filename=_GRANDQC_CKPT,
        )

        model = smp.Unet(
            encoder_name=_GRANDQC_ENCODER,
            encoder_weights=None,
            classes=_GRANDQC_N_CLASSES,
            activation=None,
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(self.device)
        self._model = model
        from mussel.models.base import IMAGENET_MEAN, IMAGENET_STD

        self._transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        mpp: float,
    ) -> np.ndarray:
        """Apply GrandQC artifact removal to a binary tissue mask.

        Args:
            img:  RGB (or RGBA) thumbnail at the segmentation level,
                  shape ``(H, W, C)``.
            mask: Binary tissue mask, shape ``(H, W)``, dtype uint8.
            mpp:  Microns-per-pixel of ``img``.

        Returns:
            Corrected binary mask of the same shape and dtype as ``mask``,
            with pixels belonging to any class in :attr:`exclude_classes`
            zeroed out.
        """
        import cv2
        import torch
        from PIL import Image

        if self._model is None:
            self._load_model()

        h, w = img.shape[:2]
        img_rgb = img[:, :, :3]  # drop alpha channel if present

        scale = mpp / _GRANDQC_TARGET_MPP
        if scale > self.max_input_mpp / _GRANDQC_TARGET_MPP:
            logger.warning(
                "GrandQCArtifactRemover: input MPP %.2f µm/px is %.1f× coarser than "
                "the model's target %.1f µm/px (max_input_mpp=%.1f). "
                "The thumbnail lacks sufficient detail for reliable artifact detection. "
                "Returning original mask unchanged. Pass a thumbnail at ≤ %.1f µm/px.",
                mpp,
                scale,
                _GRANDQC_TARGET_MPP,
                self.max_input_mpp,
                self.max_input_mpp,
            )
            return mask.copy()

        # scale < 1 → img is higher-res than target → downsample (INTER_AREA).
        # scale > 1 → img is lower-res than target → upsample (INTER_LINEAR).
        if abs(scale - 1.0) > 0.05:
            proc_h = max(1, round(h * scale))
            proc_w = max(1, round(w * scale))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            img_rgb = cv2.resize(img_rgb, (proc_w, proc_h), interpolation=interp)

        proc_h, proc_w = img_rgb.shape[:2]
        tile = _GRANDQC_TILE_SIZE

        # Pad to a multiple of tile size using reflection padding.
        pad_h = (tile - proc_h % tile) % tile
        pad_w = (tile - proc_w % tile) % tile
        img_padded = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        ph, pw = img_padded.shape[:2]
        n_h, n_w = ph // tile, pw // tile

        # Pre-build the exclude tensor once for this call.
        exclude_tensor = torch.tensor(
            sorted(self.exclude_classes), dtype=torch.long, device=self.device
        )

        # Extract tiles and apply ImageNet normalisation.
        tiles = [
            self._transforms(
                Image.fromarray(
                    img_padded[i * tile : (i + 1) * tile, j * tile : (j + 1) * tile]
                )
            )
            for i in range(n_h)
            for j in range(n_w)
        ]

        # Run inference in batches.
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(tiles), self.batch_size):
                batch = torch.stack(tiles[start : start + self.batch_size]).to(
                    self.device
                )
                logits = self._model(batch)
                probs = torch.softmax(logits, dim=1)
                _, cls = torch.max(probs, dim=1)  # (B, tile, tile)
                keep = ~torch.isin(cls, exclude_tensor)
                preds.append(keep.cpu().numpy().astype(np.uint8))

        # Reassemble tiles into a full prediction map and unpad.
        all_preds = np.concatenate(preds, axis=0)  # (n_h*n_w, tile, tile)
        pred_map = all_preds.reshape(n_h, n_w, tile, tile)
        # transpose(0,2,1,3) → (n_h, tile, n_w, tile), reshape → (ph, pw)
        pred_full = pred_map.transpose(0, 2, 1, 3).reshape(ph, pw)[:proc_h, :proc_w]

        # Scale prediction back to the original mask dimensions.
        if pred_full.shape != (h, w):
            pred_full = cv2.resize(
                pred_full.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

        return (mask * pred_full).astype(mask.dtype)
