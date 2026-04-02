"""GenBio-PathFM foundation model.

GenBio-PathFM is a 1.1B-parameter pathology foundation model trained with a
JEDI (JEPA + DINO) strategy.  It processes each RGB channel independently
through a single-channel ViT-G/16 backbone, and concatenates the three
per-channel CLS tokens into a 4608-dimensional feature vector.

References:
  - https://huggingface.co/genbio-ai/genbio-pathfm
  - https://github.com/genbio-ai/genbio-pathfm

Feature dimension: 4608 (3 × embed_dim=1536)
Input: 224×224 with pathology-specific normalisation
  mean=(0.697, 0.575, 0.728), std=(0.188, 0.240, 0.187)

The model architecture code is published under the GenBio AI Community License
(non-commercial research use only), which is incompatible with Mussel's GPL-3.0
license and therefore cannot be vendored here.  Instead, ``genbio_pathfm/model.py``
is downloaded from GitHub on first use and cached locally in
``~/.cache/mussel/genbio_pathfm/``, following the same pattern as ``torch.hub``.
Weights are downloaded automatically from HuggingFace Hub.
"""

import importlib.util
import logging
import os
import tempfile
import types
from pathlib import Path
from typing import Callable, List

import torch

from mussel.models.base import TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "model.pth"
# Pinned to a specific commit for reproducibility and supply-chain safety.
_MODEL_CODE_COMMIT = "822654b881af40db0259ab6aa7e9c9dfbe0bac78"
_MODEL_CODE_URL = (
    "https://raw.githubusercontent.com/genbio-ai/genbio-pathfm/"
    f"{_MODEL_CODE_COMMIT}/genbio_pathfm/model.py"
)
_LICENSE_URL = "https://github.com/genbio-ai/genbio-pathfm/blob/main/LICENSE.txt"


@register_model(ModelType.GENBIO_PATHFM)
class GenBioPathFMModel(TorchModel):
    """GenBio-PathFM — 4608-dim, 224px input.

    Model architecture code is downloaded from GitHub on first use and cached
    in ``~/.cache/mussel/genbio_pathfm/``.  Weights come from HuggingFace Hub.
    """

    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize GenBio-PathFM.

        Args:
            model_path: HuggingFace repo ID (``genbio-ai/genbio-pathfm``) or
                local path to ``model.pth``.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.GENBIO_PATHFM.path

        device_str = "cuda" if use_gpu else "cpu"
        model_obj = self._load_genbio(model_path, device_str)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    @staticmethod
    def _fetch_model_code() -> types.ModuleType:
        """Download genbio_pathfm/model.py from GitHub and import it.

        Cached at ``~/.cache/mussel/genbio_pathfm/model.py``; downloaded only
        once.  By proceeding the user accepts the GenBio AI Community License
        (non-commercial research use only).  Uses an atomic write (temp file +
        ``os.replace``) and a file lock to be safe under concurrent SLURM workers.
        """
        from mussel.utils.model_cache import model_download_lock

        cache_dir = Path.home() / ".cache" / "mussel" / "genbio_pathfm"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_file = cache_dir / "model.py"

        if not cached_file.exists():
            with model_download_lock(
                "genbio_pathfm_model_code", cache_dir=str(cache_dir)
            ):
                # Double-check after acquiring lock.
                if not cached_file.exists():
                    logger.info(
                        "Downloading GenBio-PathFM model code → %s\n"
                        "By proceeding you accept the GenBio AI Community License "
                        "(non-commercial research use only): %s",
                        cached_file,
                        _LICENSE_URL,
                    )
                    try:
                        import requests  # noqa: PLC0415

                        response = requests.get(_MODEL_CODE_URL, timeout=60)
                        response.raise_for_status()
                        content = response.content
                    except Exception:
                        import urllib.request  # noqa: PLC0415

                        with urllib.request.urlopen(_MODEL_CODE_URL) as r:
                            content = r.read()

                    # Atomic write: write to a temp file then rename.
                    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py.tmp", dir=cache_dir)
                    try:
                        with os.fdopen(tmp_fd, "wb") as fh:
                            fh.write(content)
                        os.replace(tmp_path, cached_file)
                    except Exception:
                        os.unlink(tmp_path)
                        raise

        spec = importlib.util.spec_from_file_location(
            "_genbio_pathfm_model", cached_file
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Cannot load GenBio-PathFM model code from {cached_file}. "
                f"Try deleting the cache and re-running: rm -f {cached_file}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _load_genbio(model_path: str, device: str) -> torch.nn.Module:
        """Download weights and instantiate ``GenBio_PathFM_Inference``."""
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("huggingface_hub is required to download weights") from e

        if os.path.isfile(model_path):
            weights_path = model_path
        else:
            repo_id = model_path.replace("hf-hub:", "")
            logger.info("Downloading GenBio-PathFM weights from %s", repo_id)
            weights_path = hf_hub_download(repo_id, _CHECKPOINT_FILENAME)

        module = GenBioPathFMModel._fetch_model_code()
        GenBio_PathFM_Inference = module.GenBio_PathFM_Inference
        return GenBio_PathFM_Inference(weights_path, device=device)

    def get_preprocessing_fun(self) -> Callable:
        """Pathology-normalised 224×224 transforms from GenBio_PathFM_Inference."""
        return self.obj.transform

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # GenBio uses .view() internally which requires contiguous memory;
        # channels_last layout (applied by TorchModel) would break it.
        return self.obj(x.contiguous())

    @property
    def autocast_dtype(self) -> torch.dtype:
        return torch.float16 if self.device.type == "cuda" else torch.bfloat16
