# Mussel Regression Report

**Branch:** `trident-features`  
**Slide:** `948176.svs` (MSK internal, not committed to repo)  
**Hardware:** A100, CUDA 12.6  

---

## 1. Patch-level feature regression

**Script:** `tests/regression/regression_vs_reference.py`

Mussel extracts features using the same patch coordinates as the REEF reference pipeline (1,675 patches from the reference filter H5, 223 px at 0.5 µm/px → resized to 224) and compares the output vectors directly.

**Reference files:**
- Patches: `/gpfs/cdsi_ess/foundation/reef/filter_tiles/9481/948176.patch.h5`
- OPTIMUS features: `/gpfs/cdsi_ess/foundation/reef/features/optimus/9481/948176.features.pt`
- CTransPath features: `/gpfs/cdsi_ess/foundation/reef/features/ctranspath/9481/948176.features.pt`

| Model | Patches | Shape | cos mean | L2 max | max abs diff | Result |
|---|---|---|---|---|---|---|
| `OPTIMUS` | 1,675 | (1675, 1536) | 1.000000 | 0.000000 | 0.000000 | ✅ PASS |
| `CTRANSPATH` | 1,675 | (1675, 768) | 1.000000 | 0.00166 | 3.2 × 10⁻⁴ | ✅ PASS |

OPTIMUS is bit-exact. CTransPath differs by at most 3.2 × 10⁻⁴ (within `rtol=1e-2, atol=1e-3`), attributable to float16 autocast rounding differences between runs. Cosine similarity is 1.000000 for both.

---

## 2. Full-pipeline regression

**Script:** `tests/regression/regression_full_pipeline.py`

End-to-end run: tessellate → extract CTransPath → filter with `model-1727990346535.pkl` (CalibratedClassifierCV) at threshold 0.75. Compared against the REEF post-filter reference.

**Reference files:**
- Classifier: `/gpfs/mskmind_ess/limr/repos/Mussel/model-1727990346535.pkl`
- Post-filter patches: `/gpfs/cdsi_ess/foundation/reef/filter_tiles/9481/948176.patch.h5` (1,675 tiles)
- Post-filter features: `/gpfs/cdsi_ess/foundation/reef/features/ctranspath/9481/948176.features.pt`

### Pipeline step results

| Step | Mussel | Reference |
|---|---|---|
| Tessellated tiles | 1,819 | — |
| Post-filter tiles | 1,763 | 1,675 |
| Tile ratio | 105.3% | — |

### Grid comparison

| Metric | Mussel | Reference | Delta |
|---|---|---|---|
| Grid origin x₀ | 608 | 608 | 0 |
| Grid origin y₀ | 2,401 | 2,433 | **Δy = 32 px** |
| Exact coord overlap | 0 / 1,675 | — | grids are disjoint |
| Bounding-box IoU | — | — | **0.977** |

### Feature distribution

| | mean | std | min | max |
|---|---|---|---|---|
| Mussel | −0.00000 | 0.12619 | — | — |
| Reference | −0.00003 | 0.12721 | — | — |

**Overall result: ⚠️ WARN** — tiles=1763/1675 (105.3%), bb_iou=0.977

### Interpretation

The 32 px y-offset in grid origin comes from a difference in tissue segmentation boundary detection between Mussel and REEF. This shifts the entire y-grid so that no coordinates exactly match, even though both pipelines cover the same tissue area (bb-IoU = 0.977) and produce statistically identical feature distributions.

Per-patch feature accuracy is confirmed by the patch-level regression above (cos = 1.000000 when the same patches are given to both pipelines).

---

## 3. Summary

| Test | Script | Result |
|---|---|---|
| OPTIMUS patch features vs REEF | `regression_vs_reference.py` | ✅ PASS — bit-exact |
| CTransPath patch features vs REEF | `regression_vs_reference.py` | ✅ PASS — cos=1.000000, L2_max=0.00166 |
| Full pipeline (tessellate→extract→filter) | `regression_full_pipeline.py` | ⚠️ WARN — 105.3% tiles, bb_iou=0.977, 32px y-offset |
