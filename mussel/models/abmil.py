"""ABMIL (Attention-Based MIL) slide encoder."""

import logging
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mussel.models.base import TorchModel
from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


class ABMIL(nn.Module):
    """Multi-headed attention-based MIL pooling module.

    Implements the tanh-attention (and optional sigmoid-gating) formulation from
    Ilse et al. (2018), https://arxiv.org/abs/1802.04712, extended to multiple
    independent attention heads.

    Args:
        feature_dim: Input (and output) feature dimension. Defaults to 1024.
        head_dim: Hidden dimension for each attention head. Defaults to 256.
        n_heads: Number of independent attention heads. Defaults to 8.
        dropout: Dropout probability applied inside each head. Defaults to 0.0.
        n_branches: Number of attention branches (one per class for CLAM-style
            models, 1 for standard ABMIL). Defaults to 1.
        gated: If True, sigmoid gating is applied (Gated-ABMIL). Defaults to False.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        head_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.0,
        n_branches: int = 1,
        gated: bool = False,
    ):
        super().__init__()
        self.gated = gated
        self.n_heads = n_heads

        self.attention_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, head_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                )
                for _ in range(n_heads)
            ]
        )

        if self.gated:
            self.gating_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(feature_dim, head_dim),
                        nn.Sigmoid(),
                        nn.Dropout(dropout),
                    )
                    for _ in range(n_heads)
                ]
            )

        self.branching_layers = nn.ModuleList(
            [nn.Linear(head_dim, n_branches) for _ in range(n_heads)]
        )

        if n_heads > 1:
            self.condensing_layer = nn.Linear(n_heads * feature_dim, feature_dim)

    def forward(
        self,
        features: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """Aggregate patch features via multi-head ABMIL attention.

        Args:
            features: Patch feature tensor of shape ``[B, N, D]``.
            attn_mask: Boolean mask of shape ``[B, N]``. True entries are kept;
                False entries receive ``-1e9`` attention logits before softmax.

        Returns:
            Tuple of:
            - aggregated_features: ``[B, n_branches, D]``
            - attention_scores: ``[B, n_branches, n_heads, N]``
        """
        if features.dim() != 3:
            raise ValueError(
                f"Input features must be 3-dimensional (B x N x D). Got {features.shape}."
            )
        if attn_mask is not None:
            if attn_mask.dim() != 2:
                raise ValueError(
                    f"Attention mask must be 2-dimensional (B x N). Got {attn_mask.shape}."
                )
            if features.shape[:2] != attn_mask.shape:
                raise ValueError(
                    f"Batch size and N must match: {features.shape[:2]} vs {attn_mask.shape}."
                )

        head_attentions: List[torch.Tensor] = []
        head_features: List[torch.Tensor] = []

        for i in range(self.n_heads):
            attn_vec = self.attention_heads[i](features)  # [B, N, head_dim]

            if self.gated:
                gate_vec = self.gating_layers[i](features)  # [B, N, head_dim]
                attn_vec = attn_vec * gate_vec

            # [B, N, n_branches]
            attn_scores = self.branching_layers[i](attn_vec)

            if attn_mask is not None:
                attn_scores = attn_scores.masked_fill(~attn_mask.unsqueeze(-1), -1e9)

            # Softmax over the N (patch) dimension → [B, N, n_branches]
            attn_weights = F.softmax(attn_scores, dim=1)

            # Weighted sum: [B, n_branches, D]
            weighted = torch.einsum("bnr,bnd->brd", attn_weights, features)

            head_attentions.append(attn_scores)  # [B, N, n_branches]
            head_features.append(weighted)  # [B, n_branches, D]

        # Concatenate heads along feature dim then condense back to D
        aggregated = torch.cat(head_features, dim=-1)  # [B, n_branches, n_heads*D]
        if self.n_heads > 1:
            aggregated = self.condensing_layer(aggregated)  # [B, n_branches, D]

        # Stack attention scores: [B, N, n_branches, n_heads] → [B, n_branches, n_heads, N]
        stacked_attn = torch.stack(
            head_attentions, dim=-1
        )  # [B, N, n_branches, n_heads]
        stacked_attn = stacked_attn.permute(
            0, 2, 3, 1
        ).contiguous()  # [B, n_branches, n_heads, N]

        return aggregated, stacked_attn


class _ABMILSlideEncoder(nn.Module):
    """Full slide-encoder: projection → ABMIL pooling → projection.

    Args:
        feature_dim: Dimension of input patch features.
        head_dim: Hidden dim for ABMIL attention heads.
        n_heads: Number of ABMIL attention heads.
        dropout: Dropout probability.
        gated: Whether to use gated ABMIL.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        head_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.0,
        gated: bool = False,
    ):
        super().__init__()

        self.pre_attention_layers = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.pooler = ABMIL(
            feature_dim=feature_dim,
            head_dim=head_dim,
            n_heads=n_heads,
            dropout=dropout,
            n_branches=1,
            gated=gated,
        )

        self.post_attention_layers = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode a bag of patch features into a single slide embedding.

        Args:
            features: Patch feature tensor of shape ``[1, N, D]``.

        Returns:
            Slide-level embedding of shape ``[1, D]``.
        """
        features = self.pre_attention_layers(features)  # [1, N, D]
        aggregated, _ = self.pooler(features)  # [1, 1, D]
        aggregated = aggregated.squeeze(1)  # [1, D]
        aggregated = self.post_attention_layers(aggregated)  # [1, D]
        return aggregated


@register_model(ModelType.ABMIL_SLIDE)
class ABMILSlideModel(TorchModel):
    """ABMIL-based slide encoder that aggregates pre-extracted patch features.

    Unlike other slide encoders (PRISM, Feather, …), ``ABMIL_SLIDE`` is
    encoder-agnostic: you provide patch features from *any* patch encoder and
    the ABMIL network pools them into a slide-level representation.

    The checkpoint format is::

        {
            "config": {
                "feature_dim": int,
                "head_dim": int,
                "n_heads": int,
                "dropout": float,
                "gated": bool,
            },
            "state_dict": { ... }
        }

    Args:
        model_path: Path to a ``.pt`` checkpoint file.  Pass ``None`` or an
            empty string to raise an informative error (a checkpoint must be
            supplied because the architecture is data-dependent).
        use_gpu: Whether to use GPU (default: True).
        gpu_device_id: GPU device ID or list of IDs (default: None).

    Raises:
        ValueError: When ``model_path`` is ``None`` or empty – a checkpoint is
            always required.
    """

    def __init__(
        self,
        model_path: Optional[str],
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        if not model_path:
            raise ValueError(
                "ABMILSlideModel requires a checkpoint path.  "
                "Pass the path to a .pt file saved with ABMILSlideModel.save()."
            )

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        state_dict = checkpoint["state_dict"]

        model_obj = _ABMILSlideEncoder(
            feature_dim=config.get("feature_dim", 1024),
            head_dim=config.get("head_dim", 256),
            n_heads=config.get("n_heads", 8),
            dropout=config.get("dropout", 0.0),
            gated=config.get("gated", False),
        )
        model_obj.load_state_dict(state_dict)

        self._config = config
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    @property
    def autocast_dtype(self) -> torch.dtype:
        """Use float32 for slide encoders (same as Feather)."""
        return torch.float32

    def get_model_fun(self) -> Callable:
        """Return inference callable for the ABMIL slide encoder.

        Returns:
            Callable that takes patch features ``[1, N, D]`` and returns a
            slide-level embedding ``[D]`` (batch dim squeezed).
        """

        def model_fun(patch_features: torch.Tensor) -> torch.Tensor:
            with torch.no_grad(), torch.inference_mode():
                patch_features = patch_features.to(self.device, non_blocking=True)
                slide_emb = self.obj(patch_features)  # [1, D]
                return slide_emb.squeeze(0).cpu()  # [D]

        return model_fun

    def get_preprocessing_fun(self) -> None:
        """Slide encoders work on patch features – no image preprocessing needed."""
        return None

    def save(self, save_path: str):
        """Save the ABMIL slide encoder checkpoint.

        Args:
            save_path: Destination ``.pt`` file path.
        """
        model_to_save = self.obj.module if hasattr(self.obj, "module") else self.obj
        torch.save(
            {"config": self._config, "state_dict": model_to_save.state_dict()},
            save_path,
        )
        logger.info("Saved ABMIL slide encoder to %s", save_path)
