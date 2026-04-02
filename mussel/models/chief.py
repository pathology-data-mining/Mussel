"""CHIEF slide encoder model (hms-dbmi/CHIEF)."""

import logging
from pathlib import Path
from typing import Callable, List

import torch
import torch.nn as nn

from mussel.models.base import TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


class _AttnNetGated(nn.Module):
    """Gated attention network used by CHIEF (hms-dbmi/CHIEF)."""

    def __init__(self, L: int = 1024, D: int = 256, dropout: bool = False):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(L, D), nn.Tanh(), *([] if not dropout else [nn.Dropout(0.25)])
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D), nn.Sigmoid(), *([] if not dropout else [nn.Dropout(0.25)])
        )
        self.attention_c = nn.Linear(D, 1)

    def forward(self, x):
        A = self.attention_a(x) * self.attention_b(x)
        return self.attention_c(A), x  # (N,1), (N,L)


class _CHIEFAttHead(nn.Module):
    """Attention head used by CHIEF — named fc1/fc2 to match checkpoint keys."""

    def __init__(self, L: int, D: int):
        super().__init__()
        self.fc1 = nn.Linear(L, D)
        self.fc2 = nn.Linear(D, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


class _CHIEFSlideModel(nn.Module):
    """CHIEF WSI-level encoder (size_arg='small', hms-dbmi/CHIEF).

    Module structure mirrors the checkpoint key layout exactly:
      attention_net = Sequential(Linear@0, ReLU@1, Dropout@2, AttnNetGated@3)
      att_head      = _CHIEFAttHead (fc1, fc2)

    Input:  patch features  [N, 768]  (from CTRANSPATH)
    Output: slide embedding [1, 768]  (attention-pooled original patch features)
    """

    def __init__(self, n_classes: int = 2, dropout: bool = True):
        super().__init__()
        L_in, L_hidden, D_attn = 768, 512, 256
        # attention_net indices match checkpoint: 0=Linear, 1=ReLU, 2=Dropout, 3=AttnNetGated
        self.attention_net = nn.Sequential(
            nn.Linear(L_in, L_hidden),
            nn.ReLU(),
            nn.Dropout(0.25) if dropout else nn.Identity(),
            _AttnNetGated(L=L_hidden, D=D_attn, dropout=dropout),
        )
        self.classifiers = nn.Linear(L_hidden, n_classes)
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(L_hidden, 2) for _ in range(n_classes)]
        )
        self.att_head = _CHIEFAttHead(L_hidden, D_attn)
        self.text_to_vision = nn.Sequential(
            nn.Linear(768, L_hidden), nn.ReLU(), nn.Dropout(p=0.25)
        )
        self.register_buffer("organ_embedding", torch.zeros(19, 768))

    def forward(self, h):
        h_ori = h.float()
        # Run fc layers then gated attention manually (Sequential can't return tuples)
        h_proj = self.attention_net[0](h_ori)
        h_proj = self.attention_net[1](h_proj)
        h_proj = self.attention_net[2](h_proj)
        A, _ = self.attention_net[3](h_proj)
        A = torch.softmax(A.transpose(1, 0), dim=1)  # [1, N]
        return torch.mm(A, h_ori)  # [1, 768]


@register_model(ModelType.CHIEF_SLIDE)
class CHIEFSlideEncoderModel(TorchModel):
    _GDRIVE_FOLDER_ID = "1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV"
    _CHECKPOINT_FILENAME = "CHIEF_pretraining.pth"

    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize CHIEF slide encoder model (hms-dbmi/CHIEF).

        If ``model_path`` is None the checkpoint is downloaded automatically
        from the CHIEF Google Drive folder using ``gdown``.  Access to the
        folder must first be requested at
        https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV

        Args:
            model_path: Path to ``CHIEF_pretraining.pth``, or None to auto-download.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = self._download_checkpoint()

        model_obj = _CHIEFSlideModel()
        td = torch.load(model_path, weights_only=True, map_location="cpu")
        result = model_obj.load_state_dict(td, strict=False)
        missing = [
            k for k in result.missing_keys if not k.endswith("num_batches_tracked")
        ]
        if missing:
            raise RuntimeError(f"CHIEF checkpoint is missing keys: {missing}")
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    @classmethod
    def _download_checkpoint(cls) -> str:
        """Download CHIEF_pretraining.pth from Google Drive folder."""
        from huggingface_hub import constants as hf_constants

        cache_dir = Path(hf_constants.HF_HUB_CACHE) / "chief"
        dest = cache_dir / cls._CHECKPOINT_FILENAME
        if dest.exists():
            logger.info("CHIEF checkpoint already cached at %s", dest)
            return str(dest)
        import gdown

        cache_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/drive/folders/{cls._GDRIVE_FOLDER_ID}"
        logger.info("Downloading CHIEF weights from Google Drive → %s", cache_dir)
        gdown.download_folder(
            url, output=str(cache_dir), quiet=False, use_cookies=False
        )
        if not dest.exists():
            raise FileNotFoundError(
                f"{cls._CHECKPOINT_FILENAME} not found after download. "
                "Ensure you have been granted access to the CHIEF Google Drive folder: "
                f"{url}"
            )
        return str(dest)

    def get_model_fun(self) -> Callable:
        """Return callable that aggregates patch features into a slide embedding."""

        def model_fun(features_tensor):
            with torch.no_grad():
                # features_tensor arrives as [1, N, D] (batch dim from _apply_slide_aggregation)
                h = (
                    features_tensor.squeeze(0)
                    .to(self.device, non_blocking=True)
                    .float()
                )
                return self.obj(h).squeeze().cpu()

        return model_fun

    def get_preprocessing_fun(self) -> Callable:
        """Slide encoders work on patch features; no image preprocessing needed."""
        return None
