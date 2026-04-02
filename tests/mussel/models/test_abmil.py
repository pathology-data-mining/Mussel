"""Tests for ABMIL slide encoder (mussel/models/abmil.py)."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

import mussel.models  # noqa: F401 — ensures all models registered
from mussel.models.abmil import ABMIL, ABMILSlideModel, _ABMILSlideEncoder
from mussel.models.model_factory import ModelType, get_model_factory


# ---------------------------------------------------------------------------
# ABMIL nn.Module
# ---------------------------------------------------------------------------


class TestABMILModule:
    def test_forward_basic_shapes(self):
        """Basic forward: aggregated [B, n_branches, D] + attn [B, n_branches, n_heads, N]."""
        B, N, D = 2, 50, 128
        model = ABMIL(feature_dim=D, head_dim=64, n_heads=4, n_branches=1)
        x = torch.randn(B, N, D)
        agg, attn = model(x)
        assert agg.shape == (B, 1, D), f"Expected ({B}, 1, {D}), got {agg.shape}"
        assert attn.shape == (B, 1, 4, N), f"Expected ({B}, 1, 4, {N}), got {attn.shape}"

    def test_forward_single_head_no_condensing(self):
        """With n_heads=1 the condensing layer is absent and output dim stays D."""
        B, N, D = 1, 10, 64
        model = ABMIL(feature_dim=D, head_dim=32, n_heads=1, n_branches=1)
        assert not hasattr(model, "condensing_layer"), "n_heads=1 should have no condensing layer"
        agg, attn = model(torch.randn(B, N, D))
        assert agg.shape == (B, 1, D)
        assert attn.shape == (B, 1, 1, N)

    def test_forward_multi_branch(self):
        """n_branches > 1 produces correct aggregated shape."""
        B, N, D = 1, 20, 64
        branches = 3
        model = ABMIL(feature_dim=D, head_dim=32, n_heads=2, n_branches=branches)
        agg, attn = model(torch.randn(B, N, D))
        assert agg.shape == (B, branches, D)
        assert attn.shape == (B, branches, 2, N)

    def test_forward_gated(self):
        """Gated ABMIL has gating_layers and produces correct output shapes."""
        B, N, D = 1, 15, 64
        model = ABMIL(feature_dim=D, head_dim=32, n_heads=2, gated=True)
        assert hasattr(model, "gating_layers") and len(model.gating_layers) == 2
        agg, attn = model(torch.randn(B, N, D))
        assert agg.shape == (B, 1, D)

    def test_attention_mask_suppresses_positions(self):
        """Masked (False) positions should receive near-zero attention weight."""
        B, N, D = 1, 10, 32
        model = ABMIL(feature_dim=D, head_dim=16, n_heads=1, gated=False)
        model.eval()

        # Only the first 5 positions are unmasked
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[:, :5] = True

        features = torch.randn(B, N, D)
        _, attn = model(features, attn_mask=mask)
        # attn shape: [B, n_branches, n_heads, N] → scores over positions
        # After softmax the masked positions should have essentially 0 weight
        attn_weights = torch.softmax(attn[0, 0, 0], dim=-1)  # shape [N]
        assert attn_weights[5:].sum().item() < 1e-3, (
            "Masked positions should have near-zero attention weight"
        )

    def test_input_dimension_assertion(self):
        """Non-3D input should raise ValueError."""
        model = ABMIL(feature_dim=32, head_dim=16, n_heads=1)
        with pytest.raises(ValueError):
            model(torch.randn(10, 32))  # 2D, not 3D

    def test_mask_shape_assertion(self):
        """Mask with wrong batch dimension should raise ValueError."""
        model = ABMIL(feature_dim=32, head_dim=16, n_heads=1)
        x = torch.randn(2, 10, 32)
        bad_mask = torch.ones(3, 10, dtype=torch.bool)  # batch dim mismatch
        with pytest.raises(ValueError):
            model(x, attn_mask=bad_mask)


# ---------------------------------------------------------------------------
# _ABMILSlideEncoder
# ---------------------------------------------------------------------------


class TestABMILSlideEncoder:
    def test_forward_produces_slide_embedding(self):
        """_ABMILSlideEncoder maps [1, N, D] → [1, D]."""
        D, N = 128, 30
        enc = _ABMILSlideEncoder(feature_dim=D, head_dim=64, n_heads=2)
        out = enc(torch.randn(1, N, D))
        assert out.shape == (1, D), f"Expected (1, {D}), got {out.shape}"

    def test_forward_output_dtype(self):
        """Output should be float32 by default."""
        enc = _ABMILSlideEncoder(feature_dim=64)
        out = enc(torch.randn(1, 20, 64))
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# ABMILSlideModel — checkpoint save/load
# ---------------------------------------------------------------------------


def _make_checkpoint(tmp_path: Path, feature_dim: int = 64) -> str:
    """Create and save a minimal ABMIL checkpoint, return its path."""
    config = {
        "feature_dim": feature_dim,
        "head_dim": 32,
        "n_heads": 2,
        "dropout": 0.0,
        "gated": False,
    }
    encoder = _ABMILSlideEncoder(**config)
    ckpt_path = str(tmp_path / "abmil.pt")
    torch.save({"config": config, "state_dict": encoder.state_dict()}, ckpt_path)
    return ckpt_path


class TestABMILSlideModel:
    def test_raises_without_model_path(self):
        """ABMILSlideModel raises ValueError when no path is given."""
        with pytest.raises(ValueError, match="requires a checkpoint path"):
            ABMILSlideModel(model_path=None, use_gpu=False)

    def test_raises_with_empty_model_path(self):
        with pytest.raises(ValueError, match="requires a checkpoint path"):
            ABMILSlideModel(model_path="", use_gpu=False)

    def test_load_from_checkpoint(self, tmp_path):
        """Model loads cleanly from a valid checkpoint."""
        ckpt = _make_checkpoint(tmp_path)
        model = ABMILSlideModel(model_path=ckpt, use_gpu=False)
        assert isinstance(model.obj, _ABMILSlideEncoder)

    def test_save_reload_round_trip(self, tmp_path):
        """save() + reload produces identical inference output."""
        ckpt = _make_checkpoint(tmp_path, feature_dim=64)
        model = ABMILSlideModel(model_path=ckpt, use_gpu=False)

        save_path = str(tmp_path / "abmil_saved.pt")
        model.save(save_path)

        model2 = ABMILSlideModel(model_path=save_path, use_gpu=False)
        model.obj.eval()
        model2.obj.eval()

        x = torch.randn(1, 20, 64)
        with torch.no_grad():
            out1 = model.obj(x)
            out2 = model2.obj(x)
        assert torch.allclose(out1, out2, atol=1e-5), "Reloaded model should be identical"

    def test_get_model_fun_inference(self, tmp_path):
        """get_model_fun() returns a callable that produces [D] output (batch dim squeezed)."""
        D = 64
        ckpt = _make_checkpoint(tmp_path, feature_dim=D)
        model = ABMILSlideModel(model_path=ckpt, use_gpu=False)
        model_fn = model.get_model_fun()

        x = torch.randn(1, 25, D)
        result = model_fn(x)
        assert result.shape == (D,), f"Expected ({D},), got {result.shape}"
        assert result.device == torch.device("cpu"), "Output should be on CPU"

    def test_get_preprocessing_fun_is_none(self, tmp_path):
        """Slide encoders need no image preprocessing."""
        ckpt = _make_checkpoint(tmp_path)
        model = ABMILSlideModel(model_path=ckpt, use_gpu=False)
        assert model.get_preprocessing_fun() is None

    def test_autocast_dtype_is_float32(self, tmp_path):
        ckpt = _make_checkpoint(tmp_path)
        model = ABMILSlideModel(model_path=ckpt, use_gpu=False)
        assert model.autocast_dtype == torch.float32

    def test_model_type_in_registry(self, tmp_path):
        """ModelType.ABMIL_SLIDE should be loadable via get_model_factory."""
        assert hasattr(ModelType, "ABMIL_SLIDE"), "ModelType.ABMIL_SLIDE not defined"
        assert ModelType.ABMIL_SLIDE.value[0] == 35

        ckpt = _make_checkpoint(tmp_path)
        factory = get_model_factory(ModelType.ABMIL_SLIDE)
        assert factory is not None

        instance = factory.get_model(model_path=ckpt, use_gpu=False)
        assert isinstance(instance, ABMILSlideModel)
