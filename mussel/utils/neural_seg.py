"""Native neural tissue segmentation for Mussel.

This module implements deep-learning-based tissue segmentation using a
DeepLabV3 model (ResNet-50 backbone) trained on histopathology slides.

The model architecture and pre-trained weights are from the HEST tissue
segmentation project (MahmoodLab/hest-tissue-seg on HuggingFace).
This implementation is independent of the HEST package —
it only requires ``torch``, ``torchvision``, and ``huggingface_hub``
(all included in Mussel's ``torch-gpu`` / ``torch-cpu`` extras).

Usage::

    from mussel.utils.neural_seg import NeuralTissueSegmenter

    seg = NeuralTissueSegmenter()          # auto-downloads weights
    mask = seg.segment(img_rgb, slide_mpp=4.0)  # -> uint8 ndarray (0/255)
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

from mussel.models.base import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)

# HuggingFace repo hosting the pre-trained tissue segmentation checkpoint.
_HF_REPO_ID = "MahmoodLab/hest-tissue-seg"
_CKPT_FILENAME = "deeplabv3_seg_v4.ckpt"

# Model inference constants (match the training configuration).
_INPUT_SIZE = 512       # patch size in pixels
_TARGET_MPP = 1.0       # inference resolution: 1 µm/px (~10x)


class NeuralTissueSegmenter:
    """Deep-learning tissue segmenter for whole-slide images.

    Uses a DeepLabV3-ResNet50 model (2-class: tissue vs background) with
    pre-trained weights from the HEST tissue segmentation project.

    The model operates at 1 µm/px resolution; input images are
    automatically rescaled before inference and the resulting mask is
    rescaled back to the original input size.

    Args:
        weights_path: Path to the checkpoint file. If ``None``, the
            checkpoint is downloaded from HuggingFace automatically
            (``MahmoodLab/hest-tissue-seg``).
        device: PyTorch device string or ``"auto"`` (default) to select
            CUDA when available, otherwise CPU.
        batch_size: Number of 512×512 patches to process per forward pass.
        confidence_thresh: Sigmoid threshold for tissue/background decision.
            Lower values → more tissue retained. Default 0.5.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 8,
        confidence_thresh: float = 0.5,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.batch_size = batch_size
        self.confidence_thresh = confidence_thresh
        self._model = None
        self._weights_path = weights_path
        self._transform = None  # built lazily alongside the model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, img: np.ndarray, slide_mpp: float = 1.0) -> np.ndarray:
        """Segment tissue in a slide image.

        Args:
            img: RGB uint8 numpy array of shape ``(H, W, 3)`` read at the
                segmentation pyramid level.
            slide_mpp: Microns per pixel of ``img``.  Used to rescale the
                image to the model's 1 µm/px operating resolution before
                inference.

        Returns:
            Binary uint8 mask of shape ``(H, W)`` where 255 = tissue and
            0 = background, at the same spatial resolution as ``img``.
        """
        self._ensure_model_loaded()

        H, W = img.shape[:2]

        # 1. Rescale to target inference resolution (1 µm/px).
        scale = slide_mpp / _TARGET_MPP
        if scale != 1.0:
            target_H = max(1, int(round(H * scale)))
            target_W = max(1, int(round(W * scale)))
            resized = cv2.resize(img, (target_W, target_H), interpolation=cv2.INTER_CUBIC)
        else:
            target_H, target_W = H, W
            resized = img

        # 2. Tile into _INPUT_SIZE × _INPUT_SIZE patches and run inference.
        full_mask = self._run_tiled_inference(resized, target_H, target_W)

        # 3. Resize prediction mask back to the original image resolution.
        if scale != 1.0:
            full_mask = cv2.resize(full_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        return (full_mask * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Load the model and transforms if not already done."""
        if self._model is not None:
            return

        weights_path = self._weights_path or _download_checkpoint()
        self._model = self._build_model(weights_path)
        self._model.eval()
        self._model = self._model.to(self.device)
        if self.device.type == "cuda":
            self._model = self._model.half()
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        logger.info(f"NeuralTissueSegmenter loaded on {self.device}")

    def _build_model(self, weights_path: str):
        """Build and load the DeepLabV3 model from a checkpoint file."""
        model = deeplabv3_resnet50(weights=None)
        model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1, stride=1)

        # Load checkpoint; strip "model." prefix added by PyTorch Lightning.
        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception:
            logger.warning(
                "Could not load checkpoint with weights_only=True; "
                "falling back to weights_only=False. Only load checkpoints from trusted sources."
            )
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = {
            k.replace("model.", ""): v
            for k, v in checkpoint.get("state_dict", {}).items()
            if "aux" not in k
        }
        model.load_state_dict(state_dict)
        return model

    def _run_tiled_inference(
        self, img: np.ndarray, H: int, W: int
    ) -> np.ndarray:
        """Tile `img` into patches, run inference, and assemble the mask."""
        patch_size = _INPUT_SIZE
        full_mask = np.zeros((H, W), dtype=np.uint8)

        patches: list[np.ndarray] = []
        positions: list[tuple[int, int, int, int]] = []

        for y in range(0, H, patch_size):
            for x in range(0, W, patch_size):
                crop = img[y : y + patch_size, x : x + patch_size]
                y1 = min(y + patch_size, H)
                x1 = min(x + patch_size, W)
                # Pad to full patch_size if the crop is smaller (border tiles).
                if crop.shape[0] < patch_size or crop.shape[1] < patch_size:
                    padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    padded[: crop.shape[0], : crop.shape[1]] = crop
                    crop = padded
                patches.append(crop)
                positions.append((y, x, y1, x1))

        use_fp16 = self.device.type == "cuda"
        dtype = torch.float16 if use_fp16 else torch.float32

        for batch_start in range(0, len(patches), self.batch_size):
            batch = patches[batch_start : batch_start + self.batch_size]
            batch_pos = positions[batch_start : batch_start + self.batch_size]

            tensors = torch.stack(
                [self._transform(Image.fromarray(p)) for p in batch]
            ).to(self.device, dtype=dtype)

            with torch.no_grad():
                logits = self._model(tensors)["out"]  # (B, 2, 512, 512)
                probs = F.softmax(logits.float(), dim=1)
                # Channel 1 = tissue probability.
                preds = (probs[:, 1] > self.confidence_thresh).to(torch.uint8)
                preds = preds.cpu().numpy()  # (B, 512, 512)

            for pred, (y0, x0, y1, x1) in zip(preds, batch_pos):
                full_mask[y0:y1, x0:x1] = pred[: y1 - y0, : x1 - x0]

        return full_mask  # values 0 or 1


# ---------------------------------------------------------------------------
# Weight download helper
# ---------------------------------------------------------------------------


def _download_checkpoint() -> str:
    """Download ``deeplabv3_seg_v4.ckpt`` from HuggingFace and return its path.

    Uses ``huggingface_hub.snapshot_download`` so the file is cached in the
    standard HuggingFace cache directory (``~/.cache/huggingface/hub``).

    Returns:
        Absolute path to the local checkpoint file.

    Raises:
        OSError: If the download fails and no cached copy exists.
    """
    cache_dir = os.environ.get("MUSSEL_MODEL_CACHE", None)
    logger.info(f"Downloading {_CKPT_FILENAME} from {_HF_REPO_ID} …")
    local_dir = snapshot_download(
        repo_id=_HF_REPO_ID,
        repo_type="model",
        allow_patterns=[_CKPT_FILENAME],
        cache_dir=cache_dir,
    )
    ckpt_path = Path(local_dir) / _CKPT_FILENAME
    if not ckpt_path.exists():
        raise OSError(
            f"Downloaded snapshot but could not find {_CKPT_FILENAME} at {ckpt_path}."
        )
    logger.info(f"Checkpoint cached at {ckpt_path}")
    return str(ckpt_path)
