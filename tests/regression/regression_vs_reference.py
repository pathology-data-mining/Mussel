"""Regression check: Mussel features vs reference pipeline output.

Compares Mussel's OPTIMUS and CTRANSPATH feature extraction against
pre-computed reference features from the REEF pipeline, using the same
reference patch H5 (1675 patches, 223 px at 0.5 µm/px → resized to 224).

Usage (from repo root, on a GPU node):
    uv run python tests/regression/regression_vs_reference.py
"""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mussel.models.model_factory import ModelType
from mussel.utils.feature_extract import extract_patch_features

REF_PATCH_H5 = Path(
    "/gpfs/cdsi_ess/foundation/reef/filter_tiles/9481/948176.patch.h5"
)
SLIDE_PATH = REPO / "tests/testdata/948176.svs"

MODELS = [
    (
        ModelType.OPTIMUS,
        Path("/gpfs/cdsi_ess/foundation/reef/features/optimus/9481/948176.features.pt"),
    ),
    (
        ModelType.CTRANSPATH,
        Path("/gpfs/cdsi_ess/foundation/reef/features/ctranspath/9481/948176.features.pt"),
    ),
]


def _run_model(model_type: ModelType, ref_feat_path: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"Model: {model_type.name}")
    print(f"Ref:   {ref_feat_path}")

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        out_h5 = tmp.name

    extract_patch_features(
        patch_h5_path=str(REF_PATCH_H5),
        slide_path=str(SLIDE_PATH),
        output_h5_path=out_h5,
        model_type=model_type,
        batch_size=64,
        use_gpu=True,
        num_workers=0,
        pin_memory=False,
    )

    with h5py.File(out_h5, "r") as f:
        mussel_feats = f["features"][:]
        mussel_coords = f["coords"][:]

    ref_feats = torch.load(ref_feat_path, map_location="cpu", weights_only=False)
    if isinstance(ref_feats, dict):
        ref_feats = next(iter(ref_feats.values()))
    ref_feats = ref_feats.numpy()

    with h5py.File(REF_PATCH_H5, "r") as f:
        ref_coords = f["coords"][:]

    print(f"  Mussel output: {mussel_feats.shape} {mussel_feats.dtype}")
    print(f"  Reference:     {ref_feats.shape}   {ref_feats.dtype}")

    if mussel_feats.shape != ref_feats.shape:
        print(f"  SHAPE MISMATCH ❌")
        return {"model": model_type.name, "status": "SHAPE_MISMATCH"}

    if not np.array_equal(mussel_coords, ref_coords):
        print(f"  COORD MISMATCH ❌ — patches not aligned")
        return {"model": model_type.name, "status": "COORD_MISMATCH"}

    # --- Metrics ---
    dot = (mussel_feats * ref_feats).sum(axis=1)
    norms = np.linalg.norm(mussel_feats, axis=1) * np.linalg.norm(ref_feats, axis=1)
    cos = dot / np.clip(norms, 1e-8, None)

    l2 = np.linalg.norm(mussel_feats - ref_feats, axis=1)
    max_absdiff = np.abs(mussel_feats - ref_feats).max()

    close_tight = np.allclose(mussel_feats, ref_feats, rtol=1e-3, atol=1e-4)
    close_loose = np.allclose(mussel_feats, ref_feats, rtol=1e-2, atol=1e-3)

    print(f"  Cosine sim:  mean={cos.mean():.6f}  min={cos.min():.6f}  p5={np.percentile(cos, 5):.6f}")
    print(f"  L2 distance: mean={l2.mean():.5f}  max={l2.max():.5f}")
    print(f"  Max abs diff: {max_absdiff:.6f}")
    print(f"  allclose(rtol=1e-3, atol=1e-4): {close_tight}")
    print(f"  allclose(rtol=1e-2, atol=1e-3): {close_loose}")

    if cos.mean() > 0.999:
        status = "PASS"
        print(f"  → PASS ✅  mean cosine={cos.mean():.6f} > 0.999")
    elif cos.mean() > 0.99:
        status = "WARN"
        print(f"  → WARN ⚠️  mean cosine={cos.mean():.6f} in [0.99, 0.999]")
    else:
        status = "FAIL"
        print(f"  → FAIL ❌  mean cosine={cos.mean():.6f} < 0.99")

    return {
        "model": model_type.name,
        "status": status,
        "n": len(cos),
        "cos_mean": float(cos.mean()),
        "cos_min": float(cos.min()),
        "cos_p5": float(np.percentile(cos, 5)),
        "l2_mean": float(l2.mean()),
        "l2_max": float(l2.max()),
        "max_absdiff": float(max_absdiff),
    }


def main():
    print(f"Slide:    {SLIDE_PATH}")
    print(f"Patch H5: {REF_PATCH_H5}")

    results = []
    for model_type, ref_path in MODELS:
        r = _run_model(model_type, ref_path)
        results.append(r)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for r in results:
        status_sym = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(r["status"], "❓")
        cos = r.get("cos_mean", float("nan"))
        print(f"  {status_sym} {r['model']:20s}  cos_mean={cos:.6f}  status={r['status']}")
        if r["status"] not in ("PASS", "WARN"):
            all_pass = False

    print()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
