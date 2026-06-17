#!/bin/bash
#SBATCH --job-name=titan-regression-vs-base
#SBATCH --partition=hpc
#SBATCH --qos=premium
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pllimsksparky[1-4]
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/gpfs/mskmind_ess/limr/repos/Mussel-titan-fix/tests/integration/titan-regression-vs-base-%j.log
#SBATCH --export=NONE

set -euo pipefail

PATCHED=/gpfs/mskmind_ess/limr/repos/Mussel-titan-fix
UNPATCHED=/gpfs/mskmind_ess/limr/repos/Mussel-titan-base
SIF=/gpfs/mskmind_ess/limr/repos/mussel-nf/mussel-fastattn.sif
CACHE=/gpfs/cdsi_ess/home/limr/.cache

echo "=== TITAN patched vs unpatched numerical regression ==="
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
echo "Date: $(date)"
echo "Patched:   $PATCHED ($(git -C $PATCHED rev-parse --short HEAD))"
echo "Unpatched: $UNPATCHED ($(git -C $UNPATCHED rev-parse --short HEAD))"

apptainer exec --nv \
  --bind ${PATCHED}:/patched \
  --bind ${UNPATCHED}:/unpatched \
  --bind ${CACHE}:/root/.cache \
  ${SIF} python - << 'PYEOF'
import sys, numpy as np, torch

# Deterministic synthetic input — same as snapshot test
n_patches = 32
rng = np.random.default_rng(42)
patch_dim = 768
fake_features = rng.standard_normal((n_patches, patch_dim)).astype(np.float32)
fake_features /= np.linalg.norm(fake_features, axis=1, keepdims=True) + 1e-8
patch_size = 512
fake_coords = np.stack([
    np.arange(n_patches) * patch_size,
    np.zeros(n_patches, dtype=np.int64),
], axis=1).astype(np.int64)

results = {}
for label, repo in [("unpatched", "/unpatched"), ("patched", "/patched")]:
    sys.path[:] = [p for p in sys.path if "/patched" not in p and "/unpatched" not in p]
    sys.path.insert(0, repo)
    # Clear any cached mussel modules
    for key in list(sys.modules):
        if "mussel" in key:
            del sys.modules[key]

    from mussel.utils.feature_extract import _apply_slide_aggregation
    from mussel.models.model_factory import ModelType

    out = _apply_slide_aggregation(
        features=fake_features,
        aggregation_method="model",
        slide_model_type=ModelType.TITAN_SLIDE,
        use_gpu=True,
        coords=fake_coords,
        patch_size=patch_size,
    )
    results[label] = out
    print(f"{label}: shape={out.shape}, norm={np.linalg.norm(out):.4f}, "
          f"first3={out[:3].tolist()}")

p, u = results["patched"], results["unpatched"]
max_diff   = float(np.max(np.abs(p - u)))
mean_diff  = float(np.mean(np.abs(p - u)))
cos_sim    = float(np.dot(p, u) / (np.linalg.norm(p) * np.linalg.norm(u)))
allclose_1e3 = bool(np.allclose(p, u, rtol=1e-2, atol=1e-3))
allclose_1e2 = bool(np.allclose(p, u, rtol=5e-2, atol=5e-3))

print()
print("=== TITAN patched vs unpatched comparison ===")
print(f"  max abs diff  : {max_diff:.6f}")
print(f"  mean abs diff : {mean_diff:.6f}")
print(f"  cosine sim    : {cos_sim:.6f}")
print(f"  allclose(rtol=1e-2, atol=1e-3): {allclose_1e3}")
print(f"  allclose(rtol=5e-2, atol=5e-3): {allclose_1e2}")

if cos_sim < 0.99:
    print("FAIL: cosine similarity below 0.99 — REGRESSION DETECTED")
    sys.exit(1)
else:
    print("PASS: cosine similarity >= 0.99 — no meaningful regression")
PYEOF

echo "=== Done ==="
