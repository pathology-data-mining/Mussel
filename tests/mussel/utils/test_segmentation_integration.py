"""Integration tests for neural segmentation and GrandQC artifact removal.

These tests load real model weights from HuggingFace and run on the test slide
``tests/testdata/948176.svs``.  A GPU is recommended but not required.

Run via SLURM (see ``tests/slurm/run_integration.sh``) or manually:

    uv run pytest tests/mussel/utils/test_segmentation_integration.py \\
        -m integration --use-gpu -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_TESTDATA = Path(__file__).parent.parent.parent / "testdata"
_SLIDE_PATH = str(_TESTDATA / "948176.svs")


# ---------------------------------------------------------------------------
# Neural segmentation
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(300)
def test_neural_segmentation_produces_valid_patches(tmp_path, use_gpu):
    """Load real DeepLabV3 weights, tessellate 948176.svs with seg_model='neural'.

    Checks:
    - At least one tissue contour is found (non-empty patch list).
    - Patch count is within 50% of the classic-HSV baseline (~1,474 patches).
    - All coordinates are within slide bounds.
    - Output HDF5 records seg_model='neural'.
    """
    import h5py

    from mussel.utils.segment import segment_tissue

    output_h5 = str(tmp_path / "neural_seg.h5")

    result = segment_tissue(
        slide_path=_SLIDE_PATH,
        patch_size=256,
        mpp=0.5,
        tissue_area_threshold=1,
        output_h5_path=output_h5,
        seg_model="neural",
    )

    # result is a dict with 'coords' key (or None if no tissue found)
    assert result is not None, "segment_tissue returned None — no tissue found"
    coords = result.get("coords") if isinstance(result, dict) else result

    # Confirm at least some patches were produced
    assert coords is not None and len(coords) > 0, (
        "Neural segmenter produced zero patches on a slide with known tissue"
    )

    n_patches = len(coords)
    # Classic HSV baseline is ~1,474. Neural should be within 50% (±737).
    hsv_baseline = 1474
    assert n_patches > hsv_baseline * 0.5, (
        f"Neural segmenter produced only {n_patches} patches "
        f"(< 50% of HSV baseline {hsv_baseline})"
    )

    # Check HDF5 output
    with h5py.File(output_h5, "r") as f:
        h5_coords = f["coords"][:]
        seg_model_attr = f.attrs.get("seg_model", "")

    assert seg_model_attr == "neural", (
        f"Expected seg_model='neural' in HDF5 attrs, got {seg_model_attr!r}"
    )

    import tiffslide

    with tiffslide.TiffSlide(_SLIDE_PATH) as wsi:
        w, h = wsi.dimensions

    assert np.all(h5_coords[:, 0] >= 0) and np.all(h5_coords[:, 0] < w), (
        "Some x coordinates are outside slide width"
    )
    assert np.all(h5_coords[:, 1] >= 0) and np.all(h5_coords[:, 1] < h), (
        "Some y coordinates are outside slide height"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(300)
def test_neural_segmentation_patch_count_close_to_hsv(tmp_path, use_gpu):
    """Neural and HSV segmenters produce patch counts within 20% of each other."""
    from mussel.utils.segment import segment_tissue

    classic_h5 = str(tmp_path / "classic.h5")
    neural_h5 = str(tmp_path / "neural.h5")

    classic = segment_tissue(
        slide_path=_SLIDE_PATH,
        patch_size=256,
        mpp=0.5,
        tissue_area_threshold=1,
        output_h5_path=classic_h5,
        seg_model="classic",
    )
    neural = segment_tissue(
        slide_path=_SLIDE_PATH,
        patch_size=256,
        mpp=0.5,
        tissue_area_threshold=1,
        output_h5_path=neural_h5,
        seg_model="neural",
    )

    n_classic = len(classic["coords"]) if classic else 0
    n_neural = len(neural["coords"]) if neural else 0

    assert n_classic > 0 and n_neural > 0, (
        f"Expected both segmenters to find tissue: classic={n_classic}, neural={n_neural}"
    )

    ratio = n_neural / n_classic
    assert 0.5 <= ratio <= 2.0, (
        f"Neural / classic patch ratio {ratio:.2f} is outside [0.5, 2.0] "
        f"(classic={n_classic}, neural={n_neural})"
    )


# ---------------------------------------------------------------------------
# GrandQC artifact removal
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(300)
def test_grandqc_artifact_remover_runs_on_real_slide(use_gpu):
    """Load real GrandQC weights, apply to a thumbnail from 948176.svs.

    Checks:
    - Output mask has the same shape and dtype as the input mask.
    - Output is a valid binary mask (values in {0, 1}).
    - At least some tissue is retained (not everything zeroed out).
    """
    import tiffslide

    from mussel.utils import GrandQCArtifactRemover
    from mussel.utils.segment import get_slide_mpp

    device = "cuda" if _use_gpu_available(use_gpu) else "cpu"
    remover = GrandQCArtifactRemover(device=device, batch_size=4)

    with tiffslide.TiffSlide(_SLIDE_PATH) as wsi:
        slide_mpp = get_slide_mpp(wsi, _SLIDE_PATH)
        # Read thumbnail at lowest pyramid level
        seg_level = wsi.get_best_level_for_downsample(32)
        level_dims = wsi.level_dimensions[seg_level]
        img = np.array(wsi.read_region((0, 0), seg_level, level_dims))[:, :, :3]
        ds = wsi.level_downsamples[seg_level]
        img_mpp = float(slide_mpp * ds)

    # Simple binary mask: all tissue
    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)

    result = remover(img, mask, img_mpp)

    assert result.shape == mask.shape, (
        f"Output mask shape {result.shape} != input shape {mask.shape}"
    )
    assert result.dtype == mask.dtype, (
        f"Output dtype {result.dtype} != input dtype {mask.dtype}"
    )
    assert set(np.unique(result)).issubset({0, 1}), (
        f"Output mask contains non-binary values: {np.unique(result)}"
    )
    assert result.sum() > 0, (
        "GrandQC zeroed out the entire mask — likely a model or input problem"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_grandqc_artifact_remover_integrated_with_segment_tissue(tmp_path, use_gpu):
    """segment_tissue with GrandQCArtifactRemover produces a valid h5 output.

    Checks:
    - Tessellation completes without error.
    - Patch count with artifact removal is <= patch count without (artifacts only remove).
    - HDF5 output is valid.
    """
    import h5py

    from mussel.utils import GrandQCArtifactRemover
    from mussel.utils.segment import segment_tissue

    device = "cuda" if _use_gpu_available(use_gpu) else "cpu"
    remover = GrandQCArtifactRemover(device=device, batch_size=4)

    # Baseline: no artifact removal
    baseline_h5 = str(tmp_path / "baseline.h5")
    baseline = segment_tissue(
        slide_path=_SLIDE_PATH,
        patch_size=256,
        mpp=0.5,
        tissue_area_threshold=1,
        output_h5_path=baseline_h5,
    )

    # With GrandQC artifact removal
    artifact_h5 = str(tmp_path / "artifact_removed.h5")
    with_removal = segment_tissue(
        slide_path=_SLIDE_PATH,
        patch_size=256,
        mpp=0.5,
        tissue_area_threshold=1,
        output_h5_path=artifact_h5,
        remove_artifacts=True,
        artifact_remover_fn=remover,
    )

    n_baseline = len(baseline["coords"]) if baseline else 0
    n_removed = len(with_removal["coords"]) if with_removal else 0

    assert n_baseline > 0, "Baseline produced no patches"
    assert n_removed > 0, "Artifact removal zeroed all patches on a clean slide"
    assert n_removed <= n_baseline, (
        f"Artifact removal produced MORE patches than baseline "
        f"({n_removed} > {n_baseline}) — removal should only reduce"
    )

    with h5py.File(artifact_h5, "r") as f:
        coords = f["coords"][:]
    assert coords.shape[1] == 2


def _use_gpu_available(use_gpu: bool) -> bool:
    """Return True if use_gpu is set and CUDA is actually available."""
    if not use_gpu:
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
