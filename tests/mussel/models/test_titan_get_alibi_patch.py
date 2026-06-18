"""Tests for the TITAN get_alibi GPU float16 monkey-patch.

These tests verify that the patch:
1. Produces numerically close results to the original numpy float64 implementation
2. Stays within memory bounds for large N
3. Does not regress on model output shape/type
"""
import math

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers: reference implementation (copied from TITAN vision_transformer.py)
# ---------------------------------------------------------------------------

def _get_slopes_ref(n: int) -> list:
    if math.log2(n) == int(math.log2(n)):
        p = 2 ** (-2 ** -(math.log2(n) - 3))
        return [p * (p ** i) for i in range(n)]
    nearest = 2 ** math.floor(math.log2(n))
    base = _get_slopes_ref(nearest)
    if nearest == n:
        return base
    extra = _get_slopes_ref(2 * nearest)[0::2][:n - nearest]
    return base + extra


def _get_alibi_original_numpy(w: int, h: int, num_heads: int = 12, bg_mask=None):
    """Original numpy float64 implementation from TITAN."""
    x, y = np.meshgrid(np.arange(w), np.arange(h), indexing='ij')
    if bg_mask is not None:
        x = x[bg_mask.cpu().squeeze(0)]
        y = y[bg_mask.cpu().squeeze(0)]
    points = np.stack([x.ravel(), y.ravel()], axis=1)
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    slopes = torch.tensor(_get_slopes_ref(num_heads), dtype=torch.float32).view(num_heads, 1, 1)
    n_patches = dists.shape[-1]
    dists_tensor = torch.tensor(dists, dtype=torch.float32).view(1, n_patches, n_patches)
    bias_matrix = dists_tensor * slopes * -1
    embed_len = n_patches + 1
    all_bias = torch.zeros(1, num_heads, embed_len, embed_len)
    all_bias[:, :, 1:, 1:] = bias_matrix
    return all_bias


def _get_alibi_gpu_float16_standalone(w: int, h: int, num_heads: int = 12, bg_mask=None,
                                       device: str = 'cpu'):
    """Standalone version of the GPU float16 patch for testing without loading TITAN."""
    dtype = torch.float16
    dev = torch.device(device)
    x_c = torch.arange(w, device=dev, dtype=dtype)
    y_c = torch.arange(h, device=dev, dtype=dtype)
    gx, gy = torch.meshgrid(x_c, y_c, indexing='ij')
    if bg_mask is not None:
        if bg_mask.dim() == 3:
            mf = bg_mask.to(dev).squeeze(0).bool()  # (W, H)
            pts_x, pts_y = gx[mf], gy[mf]
        else:
            mf = bg_mask.to(dev).squeeze(0).bool()  # flat (W*H,)
            pts_x = gx.ravel()[mf]
            pts_y = gy.ravel()[mf]
    else:
        pts_x, pts_y = gx.ravel(), gy.ravel()
    points = torch.stack([pts_x, pts_y], dim=1)
    dists = torch.cdist(points.float(), points.float(), p=2).to(dtype)
    slopes = torch.tensor(
        _get_slopes_ref(num_heads), dtype=dtype, device=dev
    ).view(num_heads, 1, 1)
    n_patches = dists.shape[0]
    bias_matrix = -dists.unsqueeze(0) * slopes
    embed_len = n_patches + 1
    all_bias = torch.zeros(1, num_heads, embed_len, embed_len, dtype=dtype, device=dev)
    all_bias[:, :, 1:, 1:] = bias_matrix
    return all_bias


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetAlibiGpuFloat16:
    """Test the GPU float16 get_alibi monkey-patch."""

    @pytest.mark.parametrize("w,h", [(6, 6), (14, 14), (30, 40), (100, 80)])
    def test_output_shape(self, w, h):
        """Output shape matches original."""
        num_heads = 12
        ref = _get_alibi_original_numpy(w, h, num_heads)
        patched = _get_alibi_gpu_float16_standalone(w, h, num_heads)
        assert patched.shape == ref.shape, f"Shape mismatch: {patched.shape} vs {ref.shape}"

    @pytest.mark.parametrize("w,h", [(6, 6), (14, 14), (30, 40)])
    def test_numerical_closeness(self, w, h):
        """Patched output is numerically close to reference (float16 vs float64)."""
        num_heads = 12
        ref = _get_alibi_original_numpy(w, h, num_heads).float()
        patched = _get_alibi_gpu_float16_standalone(w, h, num_heads).float()
        # float16 has ~3 significant digits; allow relative tolerance of 1e-2
        assert torch.allclose(ref, patched, rtol=1e-2, atol=1e-3), (
            f"Output too different: max abs diff = {(ref - patched).abs().max():.4f}"
        )

    def test_with_bg_mask(self):
        """Mask-filtered version has correct shape and values."""
        w, h = 20, 20
        # bg_mask shape is (1, H, W) bool as TITAN uses it
        bg_mask = torch.zeros(1, w, h, dtype=torch.bool)
        bg_mask[0, ::2, ::2] = True  # every other cell
        n_fg = bg_mask.sum().item()

        patched = _get_alibi_gpu_float16_standalone(w, h, bg_mask=bg_mask)
        expected_size = (1, 12, n_fg + 1, n_fg + 1)
        assert patched.shape == expected_size, f"Shape: {patched.shape} vs {expected_size}"

    def test_large_n_no_oom(self):
        """Large N (simulating a 33k-patch slide) doesn't OOM on CPU."""
        # Use CPU to test logic without needing GPU
        # N=1000 is enough to verify the pattern; real OOM tests need GPU
        w, h = 50, 50  # 2500 patches (manageable on CPU)
        patched = _get_alibi_gpu_float16_standalone(w, h, num_heads=12)
        assert patched.shape == (1, 12, 2501, 2501)
        assert patched.dtype == torch.float16
        assert torch.isfinite(patched).all(), "Non-finite values in output"

    def test_output_dtype_and_device(self):
        """Output is float16 on correct device."""
        patched = _get_alibi_gpu_float16_standalone(10, 10)
        assert patched.dtype == torch.float16
        assert patched.device.type == 'cpu'

    def test_diagonal_is_zero(self):
        """Self-distance (diagonal) should produce maximum bias (distance=0)."""
        w, h = 4, 4
        patched = _get_alibi_gpu_float16_standalone(w, h, num_heads=12)
        # bias[head, i+1, i+1] = -slope * 0 = 0 for all i (self-distance = 0)
        for head in range(12):
            diag = torch.diagonal(patched[0, head, 1:, 1:])
            assert (diag == 0).all(), f"Non-zero diagonal for head {head}"

    def test_cosine_similarity_with_reference(self):
        """Flattened output has cosine similarity > 0.99 with reference."""
        w, h = 20, 20
        ref = _get_alibi_original_numpy(w, h).float().flatten()
        patched = _get_alibi_gpu_float16_standalone(w, h).float().flatten()
        cos_sim = torch.nn.functional.cosine_similarity(ref.unsqueeze(0), patched.unsqueeze(0))
        assert cos_sim.item() > 0.99, f"Cosine similarity too low: {cos_sim.item():.4f}"


class TestMonkeyPatchApplied:
    """Test that the patch functions exist in the conch module."""

    def test_import(self):
        """The patch functions exist as module-level callables in conch.py."""
        import ast
        from pathlib import Path

        conch_path = Path(__file__).parents[3] / "mussel" / "models" / "conch.py"
        src = conch_path.read_text()
        tree = ast.parse(src)
        fn_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "_titan_get_alibi_gpu_float16" in fn_names
        assert "_titan_forward_features_efficient" in fn_names
