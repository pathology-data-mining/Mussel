"""Madeleine slide encoder from MahmoodLab."""

import json
import logging
import os
from pathlib import Path
from typing import Callable, List

import torch
import torch.nn as nn
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


class _MadeleineGatedAttnHead(nn.Module):
    """Single gated-attention head for the Madeleine WSI embedder."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.attention_c = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N, dim]
        a = self.attention_a(h)
        b = self.attention_b(h)
        A = torch.softmax(self.attention_c(a * b), dim=1)  # [B, N, 1]
        return (A * h).sum(dim=1)  # [B, dim]


class _MadeleineModel(nn.Module):
    """Minimal MADELEINE architecture matching the MahmoodLab/madeleine checkpoint.

    Architecture (reconstructed from state-dict inspection):
      - pre_attn MLP: Linear(in_dim→hidden)–Norm–Act–Drop ×2 then Linear(hidden→n_heads*hidden)–Norm
      - n_heads gated-attention heads, each operating on a hidden-dim slice
      - projector: Linear(n_heads*hidden → hidden)   [alignment head, unused at inference]
      - token_projector: Linear(n_heads*hidden → 128) [retrieval embedding, used at inference]

    Note: MADELEINE was trained on CONCH v1.0 features (in_dim=512).
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 512,
        n_heads: int = 4,
        dropout: float = 0.25,
    ):
        super().__init__()
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
                ),
                "attn": nn.ModuleList(
                    [_MadeleineGatedAttnHead(hidden_dim) for _ in range(n_heads)]
                ),
            }
        )
        self.projector = nn.Linear(expanded, hidden_dim)
        self.token_projector = nn.Linear(expanded, 128)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N, in_dim]
        h = self.wsi_embedders["pre_attn"](h)  # [B, N, n_heads*hidden]
        B, N, D = h.shape
        n_heads = len(self.wsi_embedders["attn"])
        head_dim = D // n_heads
        h = h.view(B, N, n_heads, head_dim)
        heads = [self.wsi_embedders["attn"][i](h[:, :, i, :]) for i in range(n_heads)]
        feats = torch.cat(heads, dim=-1)  # [B, n_heads*hidden]
        return self.token_projector(feats)  # [B, 128]


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

        hf_token = os.environ.get("HF_TOKEN")
        with model_download_lock(model_path) as _:
            cfg_path = hf_hub_download(model_path, "model_config.json", token=hf_token)
            pt_path = hf_hub_download(model_path, "model.pt", token=hf_token)

        cfg = json.load(open(cfg_path))
        model_obj = _MadeleineModel(
            in_dim=cfg.get("patch_embedding_dim", 512),
            hidden_dim=cfg.get("wsi_encoder_hidden_dim", 512),
            n_heads=cfg.get("n_heads", 4),
        )
        # Checkpoint was saved with DDP wrapper — strip the "module." prefix
        try:
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        except Exception:
            logger.warning(
                "Could not load checkpoint with weights_only=True; "
                "falling back to weights_only=False. Only load checkpoints from trusted sources."
            )
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        model_obj.load_state_dict(state_dict, strict=True)
        model_obj.eval()
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_model_fun(self) -> Callable:
        """Get model inference function for Madeleine slide encoder.

        Returns:
            Callable that takes patch_features [B, N, D] and returns slide-level features [B, 128].
        """

        def model_fun(patch_features):
            with torch.no_grad(), torch.inference_mode():
                patch_features = patch_features.to(self.device, non_blocking=True).float()
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
        logger.info(f"Saved Madeleine slide encoder to {save_path}")
