"""Feature adapters: map one encoder's tile features into another's embedding space.

An adapter is trained to predict a text-aligned teacher's L2-normalised tile
embeddings (e.g. KEEP's) from a non-text-aligned encoder's features of the same
tiles (e.g. H-optimus-1's). Applied to stored features, it lets text prompts
score tiles that were only ever featurized with the student encoder.

Checkpoint format (``torch.save`` of a dict):
    state_dict: ``mean``/``std`` (input standardisation, shape ``(in_dim,)``)
                and ``net.*`` weights: a single ``nn.Linear`` (``kind="ridge"``)
                or ``Linear -> GELU -> Linear`` (``kind="mlp"``)
    kind, teacher, in_dim, out_dim: metadata; ``teacher`` names the ModelType
                whose text tower the outputs should be compared with
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FeatureAdapter(nn.Module):
    """``normalize(net((x - mean) / std))``."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        net: nn.Module,
        teacher: str | None = None,
    ):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())
        self.net = net
        self.teacher = teacher

    @property
    def in_dim(self) -> int:
        return self.mean.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net((x.float() - self.mean) / self.std), dim=-1)

    @classmethod
    def load(cls, path: str) -> "FeatureAdapter":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        sd, kind = ckpt["state_dict"], ckpt["kind"]
        if kind == "ridge":
            w = sd["net.weight"]
            net = nn.Linear(w.shape[1], w.shape[0])
        elif kind == "mlp":
            w0, w2 = sd["net.0.weight"], sd["net.2.weight"]
            net = nn.Sequential(
                nn.Linear(w0.shape[1], w0.shape[0]),
                nn.GELU(),
                nn.Linear(w2.shape[1], w2.shape[0]),
            )
        else:
            raise ValueError(f"{path}: unknown adapter kind {kind!r}")
        adapter = cls(sd["mean"], sd["std"], net, ckpt.get("teacher"))
        adapter.load_state_dict(sd, strict=True)
        adapter.train(False)
        logger.info(
            "Loaded %s adapter %s (%d-dim input) -> %s",
            kind,
            path,
            adapter.in_dim,
            adapter.teacher,
        )
        return adapter

    @torch.inference_mode()
    def apply(self, features: torch.Tensor, batch_size: int = 65536) -> torch.Tensor:
        """Adapted, L2-normalised features for ``features`` of shape ``(N, in_dim)``."""
        if features.shape[-1] != self.in_dim:
            raise ValueError(
                f"features have dim {features.shape[-1]}, adapter expects {self.in_dim}"
            )
        return torch.cat(
            [
                self(features[s : s + batch_size])
                for s in range(0, len(features), batch_size)
            ]
        )
