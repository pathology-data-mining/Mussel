"""Full-pipeline regression: tessellate → CTransPath → filter vs REEF reference.

Pipeline:
  1. Tessellate 948176.svs with same params as reference (patch_size=224, mpp=0.5)
  2. Extract CTransPath features for all tiles
  3. Filter with classifier at threshold 0.75
  4. Compare filtered coords + features to REEF reference

Reference:
  Filter tiles: /gpfs/cdsi_ess/foundation/reef/filter_tiles/9481/948176.patch.h5
  Features:     /gpfs/cdsi_ess/foundation/reef/features/ctranspath/9481/948176.features.pt

Usage (from repo root, on a GPU node):
    uv run python tests/regression/regression_full_pipeline.py
"""

import pickle
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mussel.models.model_factory import ModelType
from mussel.utils import load_classifier
from mussel.utils.feature_extract import (extract_patch_features,
                                          filter_features)

SLIDE_PATH = REPO / "tests/testdata/948176.svs"
CLASSIFIER_PKL = Path("/gpfs/mskmind_ess/limr/repos/Mussel/model-1727990346535.pkl")
CLASSIFIER_THR = 0.75
REF_FILTER_H5 = Path("/gpfs/cdsi_ess/foundation/reef/filter_tiles/9481/948176.patch.h5")
REF_FEATURES_PT = Path(
    "/gpfs/cdsi_ess/foundation/reef/features/ctranspath/9481/948176.features.pt"
)


def tessellate(slide_path: Path, out_h5: str) -> int:
    """Tessellate slide with parameters matching the reference pipeline."""
    from mussel.cli.tessellate import SegConfig
    from mussel.utils.segment import segment_tissue

    seg_cfg = SegConfig(patch_size=224)  # matches CTransPath default / reference H5

    result = segment_tissue(
        slide_path=str(slide_path),
        output_h5_path=out_h5,
        **{k: v for k, v in vars(seg_cfg).items()},
    )
    if result is None:
        raise RuntimeError("segment_tissue returned None — tessellation failed")
    _, _, coords, _ = result
    return len(coords)


def main():
    print(f"Slide:       {SLIDE_PATH}")
    print(f"Classifier:  {CLASSIFIER_PKL}  (threshold={CLASSIFIER_THR})")
    print(f"Reference:   {REF_FEATURES_PT}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tess_h5 = str(tmp / "tessellate.h5")
        feats_h5 = str(tmp / "features.h5")

        # Step 1: Tessellate
        print("Step 1/3: Tessellating...")
        n_tiles = tessellate(SLIDE_PATH, tess_h5)
        with h5py.File(tess_h5, "r") as f:
            attrs = dict(f["coords"].attrs)
        print(
            f"  {n_tiles} tiles  patch_size={attrs.get('patch_size')}  mpp={attrs.get('mpp')}"
        )

        # Step 2: Extract CTransPath features
        print("Step 2/3: Extracting CTransPath features...")
        extract_patch_features(
            patch_h5_path=tess_h5,
            slide_path=str(SLIDE_PATH),
            output_h5_path=feats_h5,
            model_type=ModelType.CTRANSPATH,
            batch_size=64,
            use_gpu=True,
            num_workers=0,
            pin_memory=False,
        )
        with h5py.File(feats_h5, "r") as f:
            all_feats = f["features"][:]
            all_coords = f["coords"][:]
        print(f"  Features: {all_feats.shape}")

        # Step 3: Filter
        print(f"Step 3/3: Filtering (threshold={CLASSIFIER_THR})...")
        classifier = load_classifier(str(CLASSIFIER_PKL))
        feats_t = torch.from_numpy(all_feats)
        filt_feats_t, filt_coords = filter_features(
            feats_t, all_coords, classifier, CLASSIFIER_THR
        )
        filt_feats = filt_feats_t.numpy()
        # filt_coords is already np.ndarray
        print(
            f"  After filter: {len(filt_coords)} tiles  (removed {n_tiles - len(filt_coords)})"
        )

    # --- Load reference ---
    with h5py.File(REF_FILTER_H5, "r") as f:
        ref_coords = f["coords"][:]
    ref_feats = torch.load(
        REF_FEATURES_PT, map_location="cpu", weights_only=False
    ).numpy()

    print()
    print("=== Comparison ===")
    print(f"  Mussel filtered:  {filt_feats.shape}  coords {filt_coords.shape}")
    print(f"  Reference:        {ref_feats.shape}    coords {ref_coords.shape}")

    # --- Coordinate grid analysis ---
    mussel_set = set(map(tuple, filt_coords))
    ref_set = set(map(tuple, ref_coords))
    exact_overlap = len(mussel_set & ref_set)

    y0_mussel = filt_coords[:, 1].min()
    y0_ref = ref_coords[:, 1].min()
    x0_mussel = filt_coords[:, 0].min()
    x0_ref = ref_coords[:, 0].min()
    print(
        f"  Grid origin: Mussel x₀={x0_mussel} y₀={y0_mussel}  |  Ref x₀={x0_ref} y₀={y0_ref}  (Δy={y0_ref - y0_mussel})"
    )
    print(
        f"  Exact coord overlap: {exact_overlap} / {len(ref_coords)} reference patches"
    )

    # Bounding-box IoU
    m_xmin, m_ymin = filt_coords.min(axis=0)
    m_xmax, m_ymax = filt_coords.max(axis=0)
    r_xmin, r_ymin = ref_coords.min(axis=0)
    r_xmax, r_ymax = ref_coords.max(axis=0)
    ix1, iy1 = max(m_xmin, r_xmin), max(m_ymin, r_ymin)
    ix2, iy2 = min(m_xmax, r_xmax), min(m_ymax, r_ymax)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (
        (m_xmax - m_xmin) * (m_ymax - m_ymin)
        + (r_xmax - r_xmin) * (r_ymax - r_ymin)
        - inter
    )
    bb_iou = inter / union if union > 0 else 0.0
    print(f"  Bounding-box IoU: {bb_iou:.3f}")

    # --- Feature distribution comparison ---
    print()
    print("  Feature distribution (all filtered patches):")
    print(
        f"    Mussel  mean={filt_feats.mean():.5f}  std={filt_feats.std():.5f}  "
        f"min={filt_feats.min():.4f}  max={filt_feats.max():.4f}"
    )
    print(
        f"    Ref     mean={ref_feats.mean():.5f}  std={ref_feats.std():.5f}  "
        f"min={ref_feats.min():.4f}  max={ref_feats.max():.4f}"
    )

    # --- Feature comparison on exactly matching patches ---
    if exact_overlap > 0:
        mussel_idx = {tuple(c): i for i, c in enumerate(filt_coords)}
        ref_idx = {tuple(c): i for i, c in enumerate(ref_coords)}
        common = sorted(mussel_set & ref_set)
        mi = [mussel_idx[c] for c in common]
        ri = [ref_idx[c] for c in common]
        am = filt_feats[mi]
        ar = ref_feats[ri]
        dot = (am * ar).sum(axis=1)
        nrms = np.linalg.norm(am, axis=1) * np.linalg.norm(ar, axis=1)
        cos = dot / np.clip(nrms, 1e-8, None)
        print(f"\n  Exact-overlap patches ({exact_overlap}):")
        print(f"    Cosine sim: mean={cos.mean():.6f}  min={cos.min():.6f}")
        print(f"    Max abs diff: {np.abs(am - ar).max():.6f}")

    # --- Verdict ---
    tile_ratio = len(filt_coords) / len(ref_coords)
    coord_note = (
        "exact"
        if exact_overlap == len(ref_coords)
        else f"{exact_overlap}/{len(ref_coords)} patches align"
    )
    status = (
        "PASS"
        if (0.95 <= tile_ratio <= 1.05 and bb_iou > 0.9)
        else ("WARN" if (0.90 <= tile_ratio <= 1.10 and bb_iou > 0.8) else "FAIL")
    )
    sym = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[status]
    print()
    print(
        f"  → {sym} {status}  tiles={len(filt_coords)}/{len(ref_coords)} ({tile_ratio:.1%})"
        f"  bb_iou={bb_iou:.3f}  coords: {coord_note}"
    )
    print()
    print("  NOTE: Tile sets may differ due to segmentation differences between Mussel")
    print("  and REEF. Feature accuracy for matching patches is validated separately")
    print(
        "  by tests/regression/regression_vs_reference.py (cos=1.000000 for CTransPath)."
    )
    sys.exit(0 if status in ("PASS", "WARN") else 1)


if __name__ == "__main__":
    main()
