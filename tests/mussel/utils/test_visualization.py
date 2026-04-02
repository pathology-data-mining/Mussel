"""Tests for heatmap visualization utilities (mussel/utils/visualization.py)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mussel.utils.visualization import (
    _rank_normalize,
    apply_colormap,
    create_overlay,
    visualize_heatmap,
)


# ---------------------------------------------------------------------------
# _rank_normalize
# ---------------------------------------------------------------------------


class TestRankNormalize:
    def test_output_range(self):
        """All outputs must be in [0, 1]."""
        scores = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.0])
        result = _rank_normalize(scores)
        assert result.min() >= 0.0 and result.max() <= 1.0

    def test_monotone(self):
        """Rank order is preserved: higher input → higher normalized output."""
        scores = np.array([10.0, 5.0, 8.0, 2.0, 1.0])
        ranks = _rank_normalize(scores)
        sorted_input_idx = np.argsort(scores)
        sorted_rank_idx = np.argsort(ranks)
        np.testing.assert_array_equal(sorted_input_idx, sorted_rank_idx)

    def test_length_preserved(self):
        scores = np.arange(20, dtype=float)
        assert len(_rank_normalize(scores)) == 20

    def test_ties_get_same_rank(self):
        """Tied values receive the same normalised score (average ranking)."""
        scores = np.array([1.0, 2.0, 2.0, 3.0])
        ranks = _rank_normalize(scores)
        assert ranks[1] == pytest.approx(ranks[2]), "Tied scores should map to same rank"


# ---------------------------------------------------------------------------
# create_overlay
# ---------------------------------------------------------------------------


class TestCreateOverlay:
    def _make_inputs(self, n_patches=10, img_size=(100, 100), patch_px=20):
        """Produce consistent test inputs."""
        np.random.seed(0)
        coords = np.array([[i * 5, j * 5] for i in range(4) for j in range(4)])[:n_patches]
        scores = np.random.rand(len(coords))
        scale = np.array([1.0 / 4, 1.0 / 4])
        return scores, coords, patch_px, scale, img_size

    def test_output_shape(self):
        """Overlay shape is (height, width) = reversed region_size."""
        scores, coords, patch_px, scale, region_size = self._make_inputs()
        W, H = region_size
        overlay = create_overlay(scores, coords, patch_px, scale, region_size)
        assert overlay.shape == (H, W)

    def test_uncovered_pixels_are_nan(self):
        """Pixels with no patch coverage should be NaN."""
        # Single patch at top-left only
        scores = np.array([1.0])
        coords = np.array([[0, 0]])
        patch_px = 10
        scale = np.array([0.25, 0.25])
        region_size = (200, 200)
        overlay = create_overlay(scores, coords, patch_px, scale, region_size)
        # Most of the 200x200 image has no coverage
        assert np.isnan(overlay).sum() > 0, "There should be NaN pixels where no patches overlap"

    def test_covered_pixels_not_nan(self):
        """Covered pixels should not be NaN."""
        scores = np.array([0.5, 0.8])
        coords = np.array([[0, 0], [5, 5]])
        patch_px = 8
        scale = np.array([1.0, 1.0])
        region_size = (20, 20)
        overlay = create_overlay(scores, coords, patch_px, scale, region_size)
        # Top-left corner must be covered
        assert not np.isnan(overlay[0, 0])

    def test_overlapping_patches_averaged(self):
        """Two fully overlapping patches with scores 0 and 1 → 0.5."""
        scores = np.array([0.0, 1.0])
        coords = np.array([[0, 0], [0, 0]])  # Identical position
        patch_px = 10
        scale = np.array([1.0, 1.0])
        region_size = (10, 10)
        overlay = create_overlay(scores, coords, patch_px, scale, region_size)
        # Every pixel in the patch should be ~0.5
        covered = overlay[~np.isnan(overlay)]
        np.testing.assert_allclose(covered, 0.5, atol=1e-6)


# ---------------------------------------------------------------------------
# apply_colormap
# ---------------------------------------------------------------------------


class TestApplyColormap:
    def test_output_shape_and_dtype(self):
        """Output must be (H, W, 3) uint8."""
        overlay = np.random.rand(50, 60)
        result = apply_colormap(overlay, "coolwarm")
        assert result.shape == (50, 60, 3)
        assert result.dtype == np.uint8

    def test_nan_pixels_are_black(self):
        """NaN pixels in overlay map to (0, 0, 0) in the RGB result."""
        overlay = np.full((10, 10), np.nan)
        result = apply_colormap(overlay, "coolwarm")
        assert (result == 0).all(), "All-NaN overlay should produce all-black image"

    def test_valid_pixels_nonzero(self):
        """Non-NaN pixels should generally produce non-zero RGB values."""
        overlay = np.ones((10, 10)) * 0.5
        result = apply_colormap(overlay, "coolwarm")
        # For the 'coolwarm' cmap, value 0.5 should be roughly white/neutral
        # At minimum it shouldn't be all zero
        assert result.max() > 0


# ---------------------------------------------------------------------------
# visualize_heatmap — uses the real test slide
# ---------------------------------------------------------------------------


SVS_PATH = "tests/testdata/948176.svs"
PATCH_SIZE_LEVEL0 = 256


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists(SVS_PATH),
    reason="Test slide not available",
)
class TestVisualizeHeatmap:
    def _make_scores_coords(self, n=20):
        """Generate synthetic scores and tile coordinates."""
        np.random.seed(42)
        coords = np.array([[i * PATCH_SIZE_LEVEL0, j * PATCH_SIZE_LEVEL0]
                           for i in range(5) for j in range(4)])[:n]
        scores = np.random.rand(len(coords)).astype(np.float32)
        return scores, coords

    def test_creates_heatmap_file(self, tmp_path):
        """visualize_heatmap() should create the output PNG file."""
        scores, coords = self._make_scores_coords()
        out = visualize_heatmap(
            slide_path=SVS_PATH,
            scores=scores,
            coords=coords,
            patch_size_level0=PATCH_SIZE_LEVEL0,
            output_path=str(tmp_path / "test_heatmap.png"),
        )
        assert os.path.exists(out), f"Expected output file {out} to exist"
        assert out.endswith("test_heatmap.png")

    def test_overlay_only_mode(self, tmp_path):
        """overlay_only=True still creates an output file."""
        scores, coords = self._make_scores_coords()
        out = visualize_heatmap(
            slide_path=SVS_PATH,
            scores=scores,
            coords=coords,
            patch_size_level0=PATCH_SIZE_LEVEL0,
            output_path=str(tmp_path / "overlay.png"),
            overlay_only=True,
        )
        assert os.path.exists(out)

    def test_normalize_false(self, tmp_path):
        """normalize=False does not rank-normalise scores."""
        scores, coords = self._make_scores_coords()
        out = visualize_heatmap(
            slide_path=SVS_PATH,
            scores=scores,
            coords=coords,
            patch_size_level0=PATCH_SIZE_LEVEL0,
            output_path=str(tmp_path / "no_norm.png"),
            normalize=False,
        )
        assert os.path.exists(out)

    def test_top_k_patches_default_subdir(self, tmp_path):
        """num_top_patches>0 without output_patch_dir uses topk_patches/ next to heatmap."""
        scores, coords = self._make_scores_coords()
        out_path = str(tmp_path / "topk.png")
        visualize_heatmap(
            slide_path=SVS_PATH,
            scores=scores,
            coords=coords,
            patch_size_level0=PATCH_SIZE_LEVEL0,
            output_path=out_path,
            num_top_patches=3,
        )
        topk_dir = tmp_path / "topk_patches"
        assert topk_dir.exists(), "topk_patches directory should be created next to heatmap"
        patch_files = list(topk_dir.glob("*.png"))
        assert len(patch_files) == 3, f"Expected 3 patch files, got {len(patch_files)}"

    def test_top_k_patches_explicit_dir(self, tmp_path):
        """output_patch_dir writes patch tiles to the specified directory."""
        scores, coords = self._make_scores_coords()
        patch_dir = str(tmp_path / "my_patches")
        visualize_heatmap(
            slide_path=SVS_PATH,
            scores=scores,
            coords=coords,
            patch_size_level0=PATCH_SIZE_LEVEL0,
            output_path=str(tmp_path / "heatmap.png"),
            num_top_patches=2,
            output_patch_dir=patch_dir,
        )
        patch_files = list(Path(patch_dir).glob("*.png"))
        assert len(patch_files) == 2, f"Expected 2 patch files, got {len(patch_files)}"

    def test_custom_colormap(self, tmp_path):
        """Different colourmap name should not raise."""
        scores, coords = self._make_scores_coords()
        out = visualize_heatmap(
            slide_path=SVS_PATH,
            scores=scores,
            coords=coords,
            patch_size_level0=PATCH_SIZE_LEVEL0,
            cmap="viridis",
            output_path=str(tmp_path / "viridis.png"),
        )
        assert os.path.exists(out)
