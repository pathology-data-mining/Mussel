"""OpenMidnight model from SophontAI.

OpenMidnight is an open-weights pathology foundation model based on
DINOv2 ViT-G/14 trained on ~30k pathology slides.  The checkpoint is stored
on HuggingFace as ``SophontAI/OpenMidnight`` (``teacher_checkpoint_load.pt``).

Loading procedure (from the model card):
  1. Instantiate ``dinov2_vitg14_reg`` via ``torch.hub.load``
  2. Patch the positional embedding from the checkpoint (model was trained at
     224px resolution, whereas the base DINOv2 uses 392px)
  3. Load the state dict

The DINOv2 architecture code uses SwiGLU FFN which is not available in timm,
so the ``facebookresearch/dinov2`` torch.hub repo is required.  If torch.hub
cannot access GitHub (e.g. on compute clusters), the repo zip is downloaded
via ``requests`` and cached at ``~/.cache/torch/hub/facebookresearch_dinov2_<sha>/``.

Reference: https://huggingface.co/SophontAI/OpenMidnight

Feature dimension: 1536 (ViT-Giant)
Input: 224×224, ImageNet normalisation.
"""

import io
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, List

import torch
from torchvision import transforms

from mussel.models.base import IMAGENET_MEAN, IMAGENET_STD, TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "teacher_checkpoint_load.pt"
_DINOV2_HUB_OWNER = "facebookresearch"
_DINOV2_HUB_REPO = "dinov2"
# Pinned to a specific commit for reproducibility and supply-chain safety.
_DINOV2_COMMIT = "7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c"
_DINOV2_ZIP_URL = (
    f"https://github.com/{_DINOV2_HUB_OWNER}/{_DINOV2_HUB_REPO}/"
    f"archive/{_DINOV2_COMMIT}.zip"
)


def _ensure_dinov2_hub_cache() -> Path:
    """Return path to the dinov2 hub repo, downloading if necessary.

    Uses an atomic extract-then-rename pattern so concurrent SLURM workers
    never see a partially-extracted directory.  The cache dir is named after
    the pinned commit SHA to ensure reproducibility.
    """
    from mussel.utils.model_cache import model_download_lock

    hub_dir = Path(torch.hub.get_dir())
    # Include short commit SHA in dir name so the pinned version is always used.
    repo_dir = hub_dir / f"{_DINOV2_HUB_OWNER}_{_DINOV2_HUB_REPO}_{_DINOV2_COMMIT[:8]}"

    if repo_dir.exists():
        return repo_dir

    logger.info("DINOv2 hub repo not cached; downloading via requests → %s", repo_dir)

    lock_name = f"dinov2_hub_{_DINOV2_COMMIT[:8]}"
    with model_download_lock(lock_name, cache_dir=str(hub_dir)):
        # Double-check after acquiring the lock (another worker may have finished).
        if repo_dir.exists():
            return repo_dir

        hub_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=".dinov2-tmp-", dir=hub_dir))
        try:
            import requests  # noqa: PLC0415

            response = requests.get(_DINOV2_ZIP_URL, timeout=120)
            response.raise_for_status()

            zip_prefix = f"{_DINOV2_HUB_REPO}-{_DINOV2_COMMIT}"
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                # Zip Slip protection: reject any member with absolute or
                # traversal paths before touching the filesystem.
                for member in zf.infolist():
                    member_parts = Path(member.filename).parts
                    if not member_parts:
                        continue
                    if Path(member.filename).is_absolute() or ".." in member_parts:
                        raise ValueError(
                            f"Unsafe path in DINOv2 archive: {member.filename!r}"
                        )
                zf.extractall(tmp_dir)

            # Atomic rename: extracted top-level is "dinov2-<commit>/"
            extracted = tmp_dir / zip_prefix
            os.rename(extracted, repo_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download DINOv2 hub repo from {_DINOV2_ZIP_URL}: {e}\n"
                "Pre-populate the cache manually:\n"
                f"  git clone https://github.com/{_DINOV2_HUB_OWNER}/{_DINOV2_HUB_REPO}.git {repo_dir}"
            ) from e
        finally:
            # Clean up the temp dir (it may still hold partially-extracted files).
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    return repo_dir


@register_model(ModelType.OPENMIDNIGHT)
class OpenMidnightModel(TorchModel):
    """OpenMidnight — 1536-dim, 224px input, DINOv2 ViT-G/14 weights."""

    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize OpenMidnight.

        Args:
            model_path: HuggingFace repo ID (``SophontAI/OpenMidnight``) or
                path to a local ``teacher_checkpoint_load.pt`` file.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.OPENMIDNIGHT.path

        model_obj = self._load(model_path)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    @staticmethod
    def _load(model_path: str) -> torch.nn.Module:
        """Load OpenMidnight weights into a DINOv2 ViT-G/14 backbone."""
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("huggingface_hub is required to load OpenMidnight") from e

        if os.path.isfile(model_path):
            ckpt_path = model_path
        else:
            repo_id = model_path.replace("hf-hub:", "")
            logger.info("Downloading OpenMidnight checkpoint from %s", repo_id)
            ckpt_path = hf_hub_download(repo_id, _CHECKPOINT_FILENAME)

        repo_dir = _ensure_dinov2_hub_cache()
        model = torch.hub.load(
            str(repo_dir),
            "dinov2_vitg14_reg",
            pretrained=False,
            source="local",
            trust_repo=True,
        )

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # The checkpoint encodes a 224px positional embedding; patch it in before
        # loading the state dict so the shapes match.
        model.pos_embed = torch.nn.Parameter(checkpoint["pos_embed"])
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def get_preprocessing_fun(self) -> Callable:
        """224×224 resize + ImageNet normalisation."""
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    @property
    def autocast_dtype(self) -> torch.dtype:
        return torch.float16 if self.device.type == "cuda" else torch.bfloat16

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.obj(x)
