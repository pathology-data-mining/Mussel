# Mussel v1.2.0 — New Features & Validation Report

**Branch**: `trident-features` · **Version**: 1.2.0  
**Validation slide**: `948176.svs` (85,656 × 19,917 px, native MPP 0.5026 ≈ 20C�)  
**Reference pipeline**: external WSI patching tool (Otsu segmentation)

---

## Overview

Mussel v1.2.0 adds **11 new foundation models**, three **tessellation quality parameters** (`overlap`, `min_tissue_proportion`, `seg_model`), **batch directory scanning** (`wsi_dir`), and a **pluggable artifact removal hook**.  This report describes each change and validates tessellation correctness by comparing output against an established external WSI patching pipeline on a real clinical slide.

---

## New Features

### 1. Foundation Models

Seven new **patch-level encoders** and four new **slide-level aggregators** are now available via `model_type=` and `slide_model_type=`.

#### New Patch Encoders

| `model_type` | Source | HuggingFace path | Patch size |
|---|---|---|---:|
| `PHIKON` | Owkin | `owkin/phikon` | 224 px |
| `PHIKON_V2` | Owkin | `owkin/phikon-v2` | 224 px |
| `H_OPTIMUS_1` | Bioptimus | `bioptimus/H-optimus-1` | 224 px |
| `H0_MINI` | Bioptimus | `bioptimus/H0-mini` | 224 px |
| `MIDNIGHT12K` | Kaiko AI | `kaiko-ai/midnight` | 224 px |
| `GPFM` | MajiAbo et al. | `majiabo/GPFM` | 224 px |
| `HIBOU_L` | HistAI | `histai/hibou-L` | 224 px |

#### New Slide-Level Aggregators

| `slide_model_type` | Source | Required patch encoder | HuggingFace path |
|---|---|---|---|
| `PRISM_SLIDE` | Paige AI | `VIRCHOW` | `paige-ai/Prism` |
| `FEATHER_SLIDE` | MahmoodLab | `CONCH1_5` | `MahmoodLab/abmil.base.conch_v15.pc108-24k` |
| `CHIEF_SLIDE` | MahmoodLab | `CTRANSPATH` | *(local checkpoint required)* |
| `MADELEINE_SLIDE` | MahmoodLab | `CONCH1_5` | `MahmoodLab/madeleine` |

All models are accessed through the existing `ModelFactory` and `tessellate_extract_features` CLI — no API changes required.

**Example:**
```bash
tessellate_extract_features slide_path=slide.svs model_type=PHIKON_V2
tessellate_extract_features slide_path=slide.svs model_type=VIRCHOW slide_model_type=PRISM_SLIDE
```

---

### 2. Tessellation: `overlap`

Controls patch overlap by deriving `step_size = patch_size − overlap`. Passing both `overlap > 0` and an explicit `step_size` raises a `ValueError`.

```bash
# 64 px overlap → step = 192 px (1.78× more patches than no overlap)
tessellate slide_path=slide.svs seg_config.overlap=64
```

**Validation** — Effect on patch count (256 px patches, 0.5 MPP):

| Condition | Reference | Mussel | Δ% |
|---|---:|---:|---:|
| overlap=0 (baseline) | 1,607 | 1,474 | −8.3% |
| overlap=64 px | 2,872 | 2,586 | −9.9% |

Both pipelines produce ~78% more patches with 64 px overlap, matching the expected (256/192)² ≈ 1.78× grid density increase.

![Overlap comparison](comparison_overlap.png)

---

### 3. Tessellation: `min_tissue_proportion`

Filters patches by the fraction of their area that intersects the tissue polygon. Only patches meeting the threshold are retained. Must be in `[0.0, 1.0]`. Uses `shapely.prepared.prep` for efficient intersection on large slides.

```bash
# Keep only patches where ≥50% of area is tissue
tessellate slide_path=slide.svs seg_config.min_tissue_proportion=0.5
```

**Validation** — Effect on patch count (256 px patches, no overlap):

| Condition | Reference | Mussel | Δ% |
|---|---:|---:|---:|
| mtp=0.0 (no filter) | 1,607 | 1,474 | −8.3% |
| mtp=0.5 | 1,032 | 1,088 | +5.1% |

Both pipelines remove ~30% of patches. Mussel retains slightly more (HSV segmentation produces a slightly more inclusive tissue mask than Otsu).

![min_tissue_proportion](comparison_mtp.png)

---

### 4. Tessellation: `seg_model`

Selects the segmentation backend. Supported values: `"classic"` (default, HSV-based) and `"hest"` (requires the `neural-seg` extra).

The `"hest"` backend uses a learned tissue segmenter from the HEST library, which is more robust on challenging slides (e.g., frozen sections, adipose tissue, necrosis). Unknown values raise `ValueError`; the value is normalised to lowercase before comparison.

> **Install note**: HEST is not on PyPI — it is installed from GitHub via the `neural-seg` extra. It carries heavy transitive dependencies (YOLOv8/ultralytics, scanpy, spatialdata, dask, pytorch-lightning, pyvips) and requires a CUDA GPU for practical performance.

```bash
uv sync --extra neural-seg
tessellate slide_path=slide.svs seg_config.seg_model=hest
```

**Segmenter comparison** — `"classic"` variants vs reference Otsu on the test slide:

| Segmenter | Patches | vs Reference |
|---|---:|---:|
| Reference (Otsu) | 1,607 | — |
| Mussel HSV (`classic`, default) | 1,474 | −8.3% |
| Mussel Otsu (`classic`, `use_otsu=True`) | 1,348 | −16.1% |
| Mussel HEST (`hest`) | *not validated — HEST not installed in CI* | — |

Both `classic` variants are within the 20% tolerance. The HSV segmenter is more conservative (fewer patches at tissue margins) which is generally preferable for downstream tasks. HEST neural segmentation is expected to perform better on challenging tissue types (frozen sections, adipose, necrosis) but has not been benchmarked here due to its heavyweight install requirements.

---

### 5. Batch WSI Discovery: `wsi_dir` + `search_nested`

`tessellate_extract_features` can now scan a directory for WSI files instead of requiring an explicit list.

```bash
# Process all WSIs in a flat directory
tessellate_extract_features wsi_dir=/data/slides output_dir=/data/out model_type=VIRCHOW

# Recursively scan subdirectories
tessellate_extract_features wsi_dir=/data/cohort output_dir=/data/out search_nested=true model_type=UNI
```

`wsi_dir` is mutually exclusive with both `slide_path` and `slide_paths` — mixing them raises `ValueError`. Supported extensions: `.svs`, `.ndpi`, `.tiff`, `.tif`, `.scn`, `.mrxs`, `.vms`, `.vmu`, `.bif`, `.qptiff`, `.czi`.

---

### 6. Artifact Removal Hook: `artifact_remover_fn`

A pluggable callable that receives and returns the binary tissue mask, enabling custom artifact or pen-mark suppression. Only called when `remove_artifacts=True` or `remove_penmarks=True` is also set; warns if the function is provided but neither flag is set.

```python
def remove_blue_pen(mask: np.ndarray) -> np.ndarray:
    # Custom logic to suppress blue pen marks
    ...
    return cleaned_mask

segment_tissue(
    slide_path="slide.svs",
    remove_penmarks=True,
    artifact_remover_fn=remove_blue_pen,
)
```

---

## Tessellation Validation

### Full-Slide Overview

Both pipelines cover the same tissue area. The reference pipeline (Otsu) includes a small border around the tissue; Mussel (HSV) is slightly more conservative, starting at x=608, y=2,401 vs x=256, y=0 for the reference (blank background margin).

![Full-slide overview](comparison_overview.png)

### Tissue Mask

Top row: Mussel tissue polygon (HSV segmentation). Bottom row: Reference pipeline covered area (Otsu). Both delineate the same primary tissue regions across left, centre, and right crops.

![Tissue mask](comparison_tissue_mask.png)

### Patch Grid (Baseline)

At 4× downsample (64 px per patch), the grids are near-identical. Minor differences at tissue boundaries reflect the segmenter difference.

![Patch grid](comparison_patch_grid.png)

### Patch Agreement

Green = patches present in both pipelines · Blue = Mussel-only · Red = Reference-only.  
The vast majority of patches are shared; differences are concentrated at tissue edges.

![Differential](comparison_diff.png)

---

## Quantitative Summary

All tests use `seg_model="classic"` (HSV). HEST neural segmentation was not benchmarked (not installed in CI).

| Parameter condition | Reference | Mussel | Δ% | Pass (≤20%)? |
|---|---:|---:|---:|:---:|
| Baseline: HSV, overlap=0, mtp=0 | 1,607 | 1,474 | −8.3% | ✅ |
| Baseline: Otsu, overlap=0, mtp=0 | 1,607 | 1,348 | −16.1% | ✅ |
| HSV, overlap=64 px | 2,872 | 2,586 | −9.9% | ✅ |
| HSV, min_tissue_proportion=0.5 | 1,032 | 1,088 | +5.1% | ✅ |
| HEST neural seg | — | *not run* | — | — |

Coordinate space: both pipelines use level-0 px. Coordinate span agreement:

| Axis | Reference span | Mussel span | Δ | Tolerance (±20% slide dim) |
|---|---:|---:|---:|---|
| x | 82,688 px | 81,345 px | −1,343 px | ±17,131 px ✅ |
| y | 17,920 px | 15,300 px | −2,620 px | ±3,983 px ✅ |

---

## Bug Fixed: Coordinate dtype (`float64` → `int64`)

Discovered during validation: `segment_tissue()` was saving coordinates as `float64` (Shapely `exterior.coords[0]` returns floats). Fixed by explicitly casting:

```python
asset_dict = {"coords": np.array(coords, dtype=np.int64)}
```

---

## Known Limitations

- **`tissue_area_threshold` scaling**: The default threshold (100) is scaled by the segmentation-level downsample factor, which on slides with coarse tissue can exceed actual contour areas → zero tissue found. Workaround: `tissue_area_threshold=1`. Pre-existing issue.
- **`CHIEF_SLIDE`**: Requires a locally downloaded checkpoint path; raises `NotImplementedError` if no path is supplied.
- **`seg_model="hest"`**: Requires `uv sync --extra neural-seg` (installs HEST from GitHub). Heavy deps: ultralytics, scanpy, spatialdata, dask, pytorch-lightning. Needs a CUDA GPU.

---

## Test Coverage

- **172 unit tests** pass (`uv run pytest tests/ -m "not slow"`)
- **Comparison tests** (`@pytest.mark.slow`): patch count tolerance, coordinate bounds, no duplicates, overlap correctness, min_tissue_proportion behaviour
