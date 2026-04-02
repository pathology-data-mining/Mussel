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
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# GrandQC model constants
_GRANDQC_REPO_ID = "MahmoodLab/hest-tissue-seg"
_GRANDQC_CKPT = "GrandQC_MPP1_state_dict.pth"
_GRANDQC_ENCODER = "timm-efficientnet-b0"
_GRANDQC_TARGET_MPP = 1.0  # model trained at 10x ≈ 1 µm/px
_GRANDQC_TILE_SIZE = 512
_GRANDQC_N_CLASSES = 8

# Output class indices (0-indexed) as labelled in the GrandQC paper
_CLASS_NORMAL_TISSUE = 1
_CLASS_PEN_MARKING = 4
_CLASS_BACKGROUND = 7


class GrandQCArtifactRemover:
    """Removes artifacts and/or pen marks from a binary tissue mask.

    Uses the GrandQC U-Net (EfficientNet-B0 encoder) to classify each pixel
    of the slide thumbnail into tissue vs artifact categories.  Artifact
    regions are zeroed out from the existing binary tissue mask.

    Implements the ``artifact_remover_fn(img, mask, mpp) -> mask`` protocol
    expected by :func:`~mussel.utils.segment.segment_tissue`.

    Args:
        remove_penmarks_only: If ``True``, only pen markings and background
            are suppressed; folds, dark spots, etc. are kept.  If ``False``
            (default), all non-normal-tissue classes are removed.
        device: Torch device string (e.g. ``"cuda"``, ``"cpu"``).  Defaults
            to CUDA if available.
        batch_size: Number of 512 × 512 tiles to process per forward pass.

    Example::

        from mussel.utils.artifact_removal import GrandQCArtifactRemover
        from mussel.utils.segment import segment_tissue

        remover = GrandQCArtifactRemover()
        segment_tissue(
            slide_path="slide.svs",
            remove_artifacts=True,
            artifact_remover_fn=remover,
        )
    """

    def __init__(
        self,
        remove_penmarks_only: bool = False,
        device: Optional[str] = None,
        batch_size: int = 8,
        max_input_mpp: float = 8.0,
    ) -> None:
        self.remove_penmarks_only = remove_penmarks_only
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
            Corrected binary mask of the same shape and dtype as ``mask``.
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
                if self.remove_penmarks_only:
                    keep = ~((cls == _CLASS_PEN_MARKING) | (cls == _CLASS_BACKGROUND))
                else:
                    keep = cls <= _CLASS_NORMAL_TISSUE
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
