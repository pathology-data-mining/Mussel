"""Comparison tests between Mussel tessellation and an external WSI patching pipeline.

These tests verify that Mussel produces comparable results to a reference pipeline
when using the same input slide and equivalent parameters (Otsu segmentation,
same target magnification/MPP, same patch size, no overlap).

Requires the reference pipeline to be installed at:
    /gpfs/cdsi_ess/home/limr/ess/repos/wsi-patching-ref

Run with:
    uv run pytest tests/mussel/test_wsi_pipeline_comparison.py -m slow -v
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Any

import h5py
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REF_PIPELINE_DIR = os.environ.get(
    "WSI_REF_PIPELINE_DIR",
    "/gpfs/cdsi_ess/home/limr/ess/repos/wsi-reference-pipeline",
)
REF_PIPELINE_PYTHON = f"{REF_PIPELINE_DIR}/venv/bin/python"
MUSSEL_TEST_WSI = str(
    Path(__file__).parent.parent / "testdata" / "948176.svs"
)

REF_PIPELINE_AVAILABLE = (
    os.path.isfile(REF_PIPELINE_PYTHON)
    and os.path.isdir(REF_PIPELINE_DIR)
    and os.path.isfile(MUSSEL_TEST_WSI)
)

ref_pipeline_required = pytest.mark.skipif(
    not REF_PIPELINE_AVAILABLE,
    reason=(
        f"Reference WSI pipeline not available at {REF_PIPELINE_DIR} or test WSI missing at {MUSSEL_TEST_WSI}"
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_ref_patching(
    wsi_path: str,
    tmpdir: str,
    target_mag: int = 20,
    patch_size: int = 256,
    overlap: int = 0,
    min_tissue_proportion: float = 0.0,
    timeout: int = 300,
) -> str:
    """Run reference pipeline Otsu segmentation + patching via subprocess.

    Returns the path to the resulting coordinates H5 file.
    """
    script = textwrap.dedent(
        f"""
        import sys, os
        sys.path.insert(0, {REF_PIPELINE_DIR!r})
        from trident import load_wsi
        from trident.segmentation_models import segmentation_model_factory

        wsi = load_wsi({wsi_path!r})
        seg_model = segmentation_model_factory('otsu')
        wsi.segment_tissue(
            seg_model,
            target_mag=10,
            job_dir={tmpdir!r},
            device='cpu',
            verbose=False,
        )
        h5_path = wsi.extract_tissue_coords(
            target_mag={target_mag},
            patch_size={patch_size},
            save_coords={tmpdir!r},
            overlap={overlap},
            min_tissue_proportion={min_tissue_proportion},
        )
        print(h5_path)
        """
    )
    result = subprocess.run(
        [REF_PIPELINE_PYTHON, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Reference pipeline subprocess failed (rc={result.returncode}):\n{result.stderr}"
        )
    h5_path = result.stdout.strip().splitlines()[-1]
    assert os.path.isfile(h5_path), f"Expected reference H5 at {h5_path!r}, got stdout: {result.stdout!r}"
    return h5_path


def _read_coords_h5(path: str) -> tuple[np.ndarray, Dict[str, Any]]:
    """Return (coords array, attrs dict) from an H5 coords file."""
    with h5py.File(path, "r") as f:
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs)
    return coords, attrs


def _run_mussel_patching(
    wsi_path: str,
    output_h5_path: str,
    patch_size: int = 256,
    mpp: float = 0.5,
    overlap: int = 0,
    min_tissue_proportion: float = 0.0,
) -> str:
    """Run Mussel segment_tissue() with HSV segmentation.

    Uses tissue_area_threshold=1 to disable area filtering — the default
    threshold (100) is scaled by the segmentation level downsample factor
    and can exceed actual contour areas on small or coarse-resolution slides,
    producing zero contours.  For these comparison tests we want all tissue
    regions to participate in the patch grid, which is consistent with
    reference pipeline default behaviour (no minimum contour area filter).

    Returns *output_h5_path*.
    """
    from mussel.utils.segment import segment_tissue

    segment_tissue(
        slide_path=wsi_path,
        patch_size=patch_size,
        mpp=mpp,
        tissue_area_threshold=1,
        output_h5_path=output_h5_path,
        overlap=overlap,
        min_tissue_proportion=min_tissue_proportion,
    )
    return output_h5_path


# ---------------------------------------------------------------------------
# Tests: H5 format validation (no external pipeline needed)
# ---------------------------------------------------------------------------


class TestMusselH5Format:
    """Validate that Mussel's H5 output has the expected structure."""

    def test_existing_patch_h5_has_coords_key(self):
        """The pre-generated test fixture has a 'coords' dataset."""
        fixture = Path(MUSSEL_TEST_WSI).parent / "948176.patch.h5"
        assert fixture.is_file(), f"Fixture missing: {fixture}"
        with h5py.File(fixture, "r") as f:
            assert "coords" in f, "Missing 'coords' key in Mussel H5"

    def test_existing_patch_h5_coords_shape(self):
        """Coords have shape (N, 2) with int64 dtype."""
        fixture = Path(MUSSEL_TEST_WSI).parent / "948176.patch.h5"
        coords, _ = _read_coords_h5(str(fixture))
        assert coords.ndim == 2
        assert coords.shape[1] == 2
        assert coords.dtype == np.int64 or np.issubdtype(coords.dtype, np.integer)

    def test_existing_patch_h5_attrs(self):
        """Required metadata attributes are present."""
        fixture = Path(MUSSEL_TEST_WSI).parent / "948176.patch.h5"
        _, attrs = _read_coords_h5(str(fixture))
        required = {"name", "patch_size", "mpp", "native_mpp", "patch_level"}
        missing = required - set(attrs.keys())
        assert not missing, f"Missing attrs: {missing}"

    def test_existing_patch_h5_coords_within_slide_bounds(self):
        """All coordinates are within the slide's level-0 dimensions."""
        fixture = Path(MUSSEL_TEST_WSI).parent / "948176.patch.h5"
        coords, attrs = _read_coords_h5(str(fixture))
        level_dim = attrs["level_dim"]  # [width, height]
        assert np.all(coords[:, 0] >= 0), "Negative x coordinates"
        assert np.all(coords[:, 1] >= 0), "Negative y coordinates"
        assert np.all(coords[:, 0] < level_dim[0]), "x coords exceed slide width"
        assert np.all(coords[:, 1] < level_dim[1]), "y coords exceed slide height"

    def test_mussel_tessellate_produces_valid_h5(self, tmp_path):
        """segment_tissue() writes a valid H5 with expected structure.

        Uses tissue_area_threshold=1 to bypass the default threshold
        scaling, which can filter all contours on small/coarse slides.
        """
        from mussel.utils.segment import segment_tissue

        out = str(tmp_path / "test.h5")
        segment_tissue(
            slide_path=MUSSEL_TEST_WSI,
            patch_size=256,
            mpp=0.5,
            tissue_area_threshold=1,
            output_h5_path=out,
        )
        assert os.path.isfile(out)
        coords, attrs = _read_coords_h5(out)
        assert coords.ndim == 2
        assert coords.shape[1] == 2
        assert coords.shape[0] > 0
        assert "name" in attrs
        assert "patch_size" in attrs


# ---------------------------------------------------------------------------
# Tests: Reference pipeline H5 format validation
# ---------------------------------------------------------------------------


@ref_pipeline_required
@pytest.mark.slow
class TestTridentH5Format:
    """Validate that the reference pipeline H5 output has the expected structure."""

    @pytest.fixture(scope="class")
    def ref_h5(self, tmp_path_factory):
        """Run reference pipeline once for the class; cache result path."""
        tmpdir = str(tmp_path_factory.mktemp("ref_run"))
        return _run_ref_patching(MUSSEL_TEST_WSI, tmpdir)

    def test_has_coords_key(self, ref_h5):
        with h5py.File(ref_h5, "r") as f:
            assert "coords" in f, "Missing 'coords' key in reference H5"

    def test_coords_shape(self, ref_h5):
        coords, _ = _read_coords_h5(ref_h5)
        assert coords.ndim == 2
        assert coords.shape[1] == 2
        assert coords.shape[0] > 0

    def test_coords_dtype_integer(self, ref_h5):
        coords, _ = _read_coords_h5(ref_h5)
        assert np.issubdtype(coords.dtype, np.integer), f"Expected int dtype, got {coords.dtype}"

    def test_required_attrs_present(self, ref_h5):
        _, attrs = _read_coords_h5(ref_h5)
        required = {
            "patch_size",
            "patch_size_level0",
            "level0_magnification",
            "target_magnification",
            "overlap",
            "name",
        }
        missing = required - set(attrs.keys())
        assert not missing, f"Missing reference pipeline attrs: {missing}"

    def test_coords_within_slide_bounds(self, ref_h5):
        coords, attrs = _read_coords_h5(ref_h5)
        w = attrs["level0_width"]
        h = attrs["level0_height"]
        assert np.all(coords[:, 0] >= 0)
        assert np.all(coords[:, 1] >= 0)
        assert np.all(coords[:, 0] < w), f"x coords exceed slide width {w}"
        assert np.all(coords[:, 1] < h), f"y coords exceed slide height {h}"


# ---------------------------------------------------------------------------
# Tests: Mussel vs reference pipeline comparison
# ---------------------------------------------------------------------------


@ref_pipeline_required
@pytest.mark.slow
class TestTridentMusselComparison:
    """Compare Mussel tessellation output against a reference WSI patching pipeline.

    Both pipelines use Otsu segmentation on the same WSI with matching
    parameters (20x / 0.5 MPP, 256px patches, no overlap).
    """

    @pytest.fixture(scope="class")
    def both_h5(self, tmp_path_factory):
        """Run both pipelines and return (mussel_coords, mussel_attrs, ref_coords, ref_attrs)."""
        td = str(tmp_path_factory.mktemp("ref_run"))
        md = str(tmp_path_factory.mktemp("mussel_run"))

        ref_h5 = _run_ref_patching(MUSSEL_TEST_WSI, td, target_mag=20, patch_size=256)
        mussel_h5 = _run_mussel_patching(
            MUSSEL_TEST_WSI,
            os.path.join(md, "mussel.h5"),
            patch_size=256,
            mpp=0.5,
        )

        tc, ta = _read_coords_h5(ref_h5)
        mc, ma = _read_coords_h5(mussel_h5)
        return mc, ma, tc, ta

    def test_patch_count_within_20_percent(self, both_h5):
        """Both pipelines should produce similar patch counts (within 20%).

        Mussel uses HSV-based segmentation; the reference pipeline uses Otsu on the saturation
        channel.  Both operate on the same slide at approximately 0.5 MPP, but
        at different internal segmentation resolutions (Mussel: level 3 / ~32x
        downsample; reference pipeline: 10x thumbnail).  A 20% tolerance accommodates the
        resulting minor differences in detected tissue area.
        """
        mc, _, tc, _ = both_h5
        n_mussel = len(mc)
        n_ref = len(tc)
        ratio = abs(n_mussel - n_ref) / max(n_mussel, n_ref)
        assert ratio <= 0.20, (
            f"Patch count divergence too large: Mussel={n_mussel}, reference={n_ref} "
            f"({ratio:.1%} difference, threshold 20%)"
        )

    def test_both_use_level0_coordinate_space(self, both_h5):
        """Both pipelines should produce coordinates in level-0 pixel space.

        The test slide is 85656 × 19917 at level 0. Valid coordinates
        must be within these bounds.
        """
        mc, _, tc, _ = both_h5
        slide_w, slide_h = 85656, 19917
        # Mussel
        assert np.all(mc[:, 0] < slide_w), "Mussel x coords exceed slide width"
        assert np.all(mc[:, 1] < slide_h), "Mussel y coords exceed slide height"
        # reference pipeline
        assert np.all(tc[:, 0] < slide_w), "Reference x coords exceed slide width"
        assert np.all(tc[:, 1] < slide_h), "Reference y coords exceed slide height"

    def test_coordinate_range_similar(self, both_h5):
        """The spatial extent (span) of patch grids should be roughly similar.

        Both pipelines segment the same tissue, so the width and height of
        the bounding box that contains all patches should agree to within 20%
        of the slide dimensions.  We compare spans rather than absolute min/max
        because the two segmenters may disagree on whether narrow slide margins
        count as tissue.
        """
        mc, _, tc, _ = both_h5
        slide_w, slide_h = 85656, 19917
        tol_x = slide_w * 0.20  # 20% of slide width
        tol_y = slide_h * 0.20

        mussel_span_x = int(mc[:, 0].max()) - int(mc[:, 0].min())
        ref_span_x = int(tc[:, 0].max()) - int(tc[:, 0].min())
        mussel_span_y = int(mc[:, 1].max()) - int(mc[:, 1].min())
        ref_span_y = int(tc[:, 1].max()) - int(tc[:, 1].min())

        assert abs(mussel_span_x - ref_span_x) < tol_x, (
            f"x span differs by more than 20% of slide width: "
            f"Mussel={mussel_span_x}, reference={ref_span_x}"
        )
        assert abs(mussel_span_y - ref_span_y) < tol_y, (
            f"y span differs by more than 20% of slide height: "
            f"Mussel={mussel_span_y}, reference={ref_span_y}"
        )

    def test_mussel_patch_size_attr_set(self, both_h5):
        """Mussel H5 should record patch_size attribute."""
        _, ma, _, _ = both_h5
        assert "patch_size" in ma, "Mussel H5 missing 'patch_size' attr"

    def test_ref_patch_size_attr_matches_input(self, both_h5):
        """Reference H5 should record patch_size=256 and target_magnification=20."""
        _, _, _, ta = both_h5
        assert ta["patch_size"] == 256
        assert ta["target_magnification"] == 20

    def test_no_duplicate_coords_mussel(self, both_h5):
        """Mussel should not produce duplicate patch coordinates."""
        mc, _, _, _ = both_h5
        unique = np.unique(mc, axis=0)
        assert len(unique) == len(mc), (
            f"Mussel has {len(mc) - len(unique)} duplicate coordinates"
        )

    def test_no_duplicate_coords_ref(self, both_h5):
        """Reference pipeline should not produce duplicate patch coordinates."""
        _, _, tc, _ = both_h5
        unique = np.unique(tc, axis=0)
        assert len(unique) == len(tc), (
            f"Reference pipeline has {len(tc) - len(unique)} duplicate coordinates"
        )

    def test_overlap_zero_produces_no_overlap_mussel(self, tmp_path):
        """With overlap=0, Mussel patches should not overlap each other.

        For non-overlapping patches, x-coordinates should step by at least
        patch_size pixels between consecutive sorted patches in a row.
        Uses tissue_area_threshold=1 to ensure tissue is found.
        """
        from mussel.utils.segment import segment_tissue

        out = str(tmp_path / "nooverlap.h5")
        segment_tissue(
            slide_path=MUSSEL_TEST_WSI,
            patch_size=256,
            mpp=0.5,
            tissue_area_threshold=1,
            output_h5_path=out,
            overlap=0,
        )
        coords, attrs = _read_coords_h5(out)
        patch_size_px = int(attrs["patch_size"])

        # Group by y, check x spacing
        ys = np.unique(coords[:, 1])
        for y in ys:
            row = np.sort(coords[coords[:, 1] == y][:, 0])
            if len(row) > 1:
                steps = np.diff(row)
                assert np.all(steps >= patch_size_px), (
                    f"Overlapping patches in row y={y}: steps={steps[steps < patch_size_px]}"
                )
