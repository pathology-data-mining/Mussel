"""Madeleine slide encoder from MahmoodLab."""

import json
import logging
import os
from pathlib import Path
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

try:
    from mussel.utils.model_cache import model_download_lock
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def model_download_lock(model_name, **kwargs):
        yield True


from mussel.models.base import TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


# Upstream hard-codes the attention hidden size (and so the projector width) to 512.
_ATTN_HIDDEN_DIM = 512


class _MadeleineGatedAttnHead(nn.Module):
    """Single gated-attention head (upstream ``BatchedABMIL`` with n_classes=1).

    Returns the attention weights over tokens; the caller pools with them.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = _ATTN_HIDDEN_DIM,
        activation: str = "softmax",
    ):
        super().__init__()
        self.activation = activation
        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Dropout(0.25)
        )
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(0.25)
        )
        self.attention_c = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N, input_dim] -> attention [B, N, 1]
        A = self.attention_c(self.attention_a(h) * self.attention_b(h))
        if self.activation == "softmax":
            return F.softmax(A, dim=1)
        if self.activation == "leaky_relu":
            return F.leaky_relu(A)
        if self.activation == "relu":
            return F.relu(A)
        if self.activation == "sigmoid":
            return torch.sigmoid(A)
        raise NotImplementedError(
            f"Unsupported MADELEINE activation: {self.activation}"
        )


class _MadeleineModel(nn.Module):
    """MADELEINE slide encoder, ported from upstream mahmoodlab/MADELEINE.

    Mirrors ``MADELEINE.encode_he`` (madeleine/models/Model.py) with the
    ``ABMILEmbedder``:
      - pre_attn: (Linear–LayerNorm–GELU–Dropout) ×3, the last expanding to
        ``hidden_dim * n_heads``
      - the expanded features are split into heads interleaved, as
        ``rearrange('b t (e c) -> b t e c', c=n_heads)``
      - each head's gated attention pools its slice over tokens -> [B, e, c]
      - flattened in that (e, c) order and passed through ``projector``
        -> 512-dim slide embedding.
    ``token_projector`` (128-dim) is only used for training and is not applied.

    Note: MADELEINE was trained on CONCH v1.0 features (in_dim=512).
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 512,
        n_heads: int = 4,
        activation: str = "softmax",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        expanded = hidden_dim * n_heads

        self.wsi_embedders = nn.ModuleDict(
            {
                "pre_attn": nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, expanded),
                    nn.LayerNorm(expanded),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ),
                "attn": nn.ModuleList(
                    [
                        _MadeleineGatedAttnHead(
                            hidden_dim, _ATTN_HIDDEN_DIM, activation
                        )
                        for _ in range(n_heads)
                    ]
                ),
            }
        )
        self.projector = nn.Linear(_ATTN_HIDDEN_DIM * n_heads, _ATTN_HIDDEN_DIM)
        self.token_projector = nn.Linear(_ATTN_HIDDEN_DIM * n_heads, 128)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N, in_dim]
        h = self.wsi_embedders["pre_attn"](h)  # [B, N, hidden*n_heads]
        B, N, D = h.shape
        # 'b t (e c) -> b t e c': channel j belongs to head j % n_heads.
        h = h.view(B, N, D // self.n_heads, self.n_heads)  # [B, N, e, c]
        attention = torch.stack(
            [attn(h[:, :, :, i]) for i, attn in enumerate(self.wsi_embedders["attn"])],
            dim=-1,
        )  # [B, N, 1, c]
        slide = (h * attention).sum(dim=1)  # [B, e, c]
        return self.projector(slide.reshape(B, -1))  # [B, 512]


@register_model(ModelType.MADELEINE_SLIDE)
class MadeleineSlideEncoderModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize Madeleine slide encoder model.

        Madeleine (MahmoodLab/madeleine) is a multimodal slide encoder trained on
        CONCH v1.0 patch features (512-dim).  The repo provides a raw ``model.pt``
        state-dict (DDP-wrapped) rather than a standard HuggingFace model, so we
        build the architecture explicitly and load the weights manually.

        Args:
            model_path: HuggingFace repo ID (``MahmoodLab/madeleine``) or local
                directory containing ``model.pt`` and ``model_config.json``.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.MADELEINE_SLIDE.path

        if os.path.isdir(model_path):
            cfg_path = os.path.join(model_path, "model_config.json")
            pt_path = os.path.join(model_path, "model.pt")
        else:
            hf_token = os.environ.get("HF_TOKEN")
            with model_download_lock(model_path) as _:
                cfg_path = hf_hub_download(
                    model_path, "model_config.json", token=hf_token
                )
                pt_path = hf_hub_download(model_path, "model.pt", token=hf_token)

        with open(cfg_path) as f:
            self.config = json.load(f)
        model_obj = self._build(self.config)
        try:
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        except Exception:
            logger.warning(
                "Could not load checkpoint with weights_only=True; "
                "falling back to weights_only=False. Only load checkpoints from trusted sources."
            )
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
        model_obj.load_state_dict(self._strip_module_prefix(state_dict), strict=True)
        model_obj.eval()
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    @staticmethod
    def _build(cfg: dict) -> "_MadeleineModel":
        """Build the MADELEINE architecture from a ``model_config.json`` dict."""
        return _MadeleineModel(
            in_dim=cfg.get("patch_embedding_dim", 512),
            hidden_dim=cfg.get("wsi_encoder_hidden_dim", 512),
            n_heads=cfg.get("n_heads", 4),
            activation=cfg.get("activation", "softmax"),
        )

    @staticmethod
    def _strip_module_prefix(state_dict: dict) -> dict:
        """Strip the DDP ``module.`` prefix if present, as upstream does."""
        prefix = "module."
        return {
            (k[len(prefix) :] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }

    def get_model_fun(self) -> Callable:
        """Get model inference function for Madeleine slide encoder.

        Returns:
            Callable that takes patch_features [B, N, D] and returns slide-level features [B, 512].
        """

        def model_fun(patch_features):
            with torch.no_grad(), torch.inference_mode():
                patch_features = patch_features.to(
                    self.device, non_blocking=True
                ).float()
                return self.obj(patch_features).squeeze().cpu()

        return model_fun

    def get_preprocessing_fun(self) -> Callable:
        """Slide encoders work on patch features; no image preprocessing needed."""
        return None

    def save(self, save_path: str):
        """Save Madeleine slide encoder model to disk.

        Args:
            save_path: Path to save the model (must be a directory, not a file).

        Raises:
            ValueError: If save_path has a file extension.
        """
        if Path(save_path).suffix:
            raise ValueError(
                f"MADELEINE_SLIDE model must be saved to a directory, not a file ({save_path})."
            )
        Path(save_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.obj.state_dict(), Path(save_path) / "model.pt")
        # Saved alongside model.pt so the directory can be passed back as model_path.
        config = getattr(self, "config", None)
        if config is not None:
            with open(Path(save_path) / "model_config.json", "w") as f:
                json.dump(config, f)
        logger.info(f"Saved Madeleine slide encoder to {save_path}")
