"""Tests for the TITAN get_alibi GPU monkey-patch (_titan_get_alibi_gpu).

The patch must reproduce TITAN's original float64 ALiBi bias, including on
reef-sized grids, while avoiding the original's O(N²) CPU arrays.
"""
import math

import numpy as np
import pytest
import torch

from mussel.models.conch import _get_slopes, _titan_get_alibi_gpu


# ---------------------------------------------------------------------------
# Helpers: reference implementation (original numpy float64 from TITAN)
# ---------------------------------------------------------------------------

def _get_alibi_original_numpy(w: int, h: int, num_heads: int = 12, bg_mask=None):
    """Original numpy float64 implementation from TITAN."""
    x, y = np.meshgrid(np.arange(w), np.arange(h), indexing='ij')
    if bg_mask is not None:
        x = x[bg_mask.cpu().squeeze(0)]
        y = y[bg_mask.cpu().squeeze(0)]
    points = np.stack([x.ravel(), y.ravel()], axis=1)
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    slopes = torch.tensor(_get_slopes(num_heads), dtype=torch.float32).view(num_heads, 1, 1)
    n_patches = dists.shape[-1]
    dists_tensor = torch.tensor(dists, dtype=torch.float32).view(1, n_patches, n_patches)
    bias_matrix = dists_tensor * slopes * -1
    embed_len = n_patches + 1
    all_bias = torch.zeros(1, num_heads, embed_len, embed_len)
    all_bias[:, :, 1:, 1:] = bias_matrix
    return all_bias


class _FakeVisionEncoder(torch.nn.Module):
    """The two attributes _titan_get_alibi_gpu reads from TITAN's vision encoder."""

    def __init__(self, num_heads: int = 12):
        super().__init__()
        self.num_heads = num_heads
        self.anchor = torch.nn.Parameter(torch.zeros(1))  # device comes from parameters()


def _sparse_mask(w: int, h: int, density: float):
    """(1, W, H) tissue mask like TITAN's bg_mask; density 1.0 means no mask."""
    if density >= 1.0:
        return None
    return torch.from_numpy(np.random.default_rng(0).random((1, w, h)) < density)


def _patched(w: int, h: int, num_heads: int = 12, bg_mask=None):
    return _titan_get_alibi_gpu(_FakeVisionEncoder(num_heads), w, h, bg_mask)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetAlibiGpu:
    """Test the GPU get_alibi monkey-patch (the real function, on CPU)."""

    @pytest.mark.parametrize("w,h", [(6, 6), (14, 14), (30, 40), (100, 80)])
    def test_output_shape(self, w, h):
        ref = _get_alibi_original_numpy(w, h, 12)
        assert _patched(w, h, 12).shape == ref.shape

    @pytest.mark.parametrize("w,h,density", [(6, 6, 1.0), (30, 40, 1.0), (150, 120, 0.15)])
    def test_matches_reference_at_reef_scale(self, w, h, density):
        """Matches TITAN's float64 reference to float32 precision, also on a grid
        spanning a reef slide (sparse tissue, biases near -100), where float16
        was off by ~0.06."""
        mask = _sparse_mask(w, h, density)
        ref = _get_alibi_original_numpy(w, h, 16, bg_mask=mask)
        patched = _patched(w, h, 16, bg_mask=mask)
        assert patched.dtype == torch.float32
        torch.testing.assert_close(patched, ref, rtol=1e-6, atol=1e-4)

    def test_float16_would_fail_at_reef_scale(self):
        """Guards the reason for float32: float16 rounding of these biases is ~0.06."""
        ref = _get_alibi_original_numpy(150, 120, 16, bg_mask=_sparse_mask(150, 120, 0.15))
        assert (ref.half().float() - ref).abs().max() > 0.01

    def test_with_bg_mask(self):
        """Mask-filtered version matches the reference on the kept cells."""
        w, h = 20, 20
        bg_mask = torch.zeros(1, w, h, dtype=torch.bool)  # (1, H, W) as TITAN passes it
        bg_mask[0, ::2, ::2] = True
        n_fg = bg_mask.sum().item()
        patched = _patched(w, h, bg_mask=bg_mask)
        assert patched.shape == (1, 12, n_fg + 1, n_fg + 1)
        torch.testing.assert_close(patched, _get_alibi_original_numpy(w, h, 12, bg_mask=bg_mask),
                                   rtol=1e-6, atol=1e-4)

    def test_diagonal_is_zero(self):
        patched = _patched(4, 4, num_heads=12)
        for head in range(12):
            assert (torch.diagonal(patched[0, head, 1:, 1:]) == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_matches_reference_on_gpu_with_tf32_at_reef_scale():
    """On GPU with TF32 matmul on (as mussel.models.base sets it), a reef slide's
    156x71 grid with a sparse tissue mask must still match the float64 reference.
    cdist's matmul path was off by up to 2.8 in the bias here."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        w, h = 156, 71
        bg_mask = _sparse_mask(w, h, 0.3)
        enc = _FakeVisionEncoder(12).cuda()
        patched = _titan_get_alibi_gpu(enc, w, h, bg_mask).cpu()
        ref = _get_alibi_original_numpy(w, h, 12, bg_mask=bg_mask)
        torch.testing.assert_close(patched, ref, rtol=1e-5, atol=1e-3)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
