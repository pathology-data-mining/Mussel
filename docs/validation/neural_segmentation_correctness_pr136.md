# Mussel Neural Segmentation Correctness Validation - PR #136 on main

## Summary

Validation target: fixed neural segmentation behavior on `main`, exercised with PR #136 (`codex/fix-circular-import`) applied on top.

Conclusion: PASS. The neural segmentation path completed on a 10-slide panel spanning multiple formats, slide sizes, MPP values, and tissue burdens. All generated HDF5 outputs had valid coordinates, correct `seg_model` attributes, non-empty patch sets, and neural/classic patch-count ratios inside the configured sanity range.

## Code Under Validation

Repository: `pathology-data-mining/Mussel`

Base branch: `origin/main`

Validation branch: `codex/fix-circular-import`

Head commit: `cc68d57 fix circular utils import`

Relevant upstream fixes already present on `main`:

- `fb35ace [codex] fix tessellate neural segmentation (#135)`
- `2230233 [codex] add batch mode to tessellate CLI (#134)`
- `86b2270 feat: add abmil_benchmark CLI for precision benchmarking (#124)`

## Acceptance Criteria

The validation fails if any slide has:

- neural segmentation exception or `segment_tissue(..., seg_model="neural")` returns `None`
- zero neural patches
- missing or invalid HDF5 output
- HDF5 patch count mismatch against returned coordinates
- HDF5 `coords.attrs["seg_model"]` not equal to `"neural"`
- any generated coordinate outside slide bounds
- neural/classic patch-count ratio outside `[0.10, 10.0]`

The ratio check is intentionally broad. It is not an equivalence test between classic HSV and neural segmentation; it is a bug-catcher for empty masks, pathological over-segmentation, wrong pyramid level selection, or broken rescaling.

## Validation Harness

Added reusable validation entrypoints:

- [neural_segmentation_correctness_panel.py](/gpfs/mskmind_ess/limr/repos/Mussel-2/tests/validation/neural_segmentation_correctness_panel.py)
- [run_neural_segmentation_correctness_panel.sh](/gpfs/mskmind_ess/limr/repos/Mussel-2/tests/slurm/run_neural_segmentation_correctness_panel.sh)

Panel command:

```bash
sbatch --qos=premium tests/slurm/run_neural_segmentation_correctness_panel.sh
```

The SLURM wrapper runs:

```bash
uv sync --extra torch-gpu
uv run python tests/validation/neural_segmentation_correctness_panel.py \
    --output-dir /gpfs/cdsi_ess/home/limr/logs/slurm/neural_seg_panel_${SLURM_JOB_ID} \
    --device cuda \
    --batch-size 8
```

## Results

### 10-slide neural segmentation panel

SLURM job: `3563436`

Node: `pllimsksparky3`

State: `COMPLETED`

Exit code: `0:0`

Elapsed: `00:27:00`

Max RSS: `6738208K`

Log:

`/gpfs/cdsi_ess/home/limr/logs/slurm/neural_seg_panel_3563436.out`

Machine-readable outputs:

- `/gpfs/cdsi_ess/home/limr/logs/slurm/neural_seg_panel_3563436/neural_segmentation_correctness_panel.csv`
- `/gpfs/cdsi_ess/home/limr/logs/slurm/neural_seg_panel_3563436/neural_segmentation_correctness_panel.json`

Result: `10/10 passed`

| # | Slide | Format/source | Dimensions | MPP | Classic patches | Neural patches | Ratio | Status |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `948176` | SVS testdata | 85656 x 19917 | 0.5026 | 1515 | 1560 | 1.030 | PASS |
| 2 | `8471385` | SVS large local slide | 117749 x 77677 | 0.262882 | 9617 | 15038 | 1.564 | PASS |
| 3 | `1065626` | SVS small local slide | 11952 x 10249 | 0.5026 | 9 | 30 | 3.333 | PASS |
| 4 | `1005517` | SVS workflow slide | 89640 x 33960 | 0.5021 | 9491 | 18386 | 1.937 | PASS |
| 5 | `PROV-000-000001` | NDPI GigaPath sample | 145920 x 20736 | 0.23016 | 1015 | 1004 | 0.989 | PASS |
| 6 | `TCGA-AZ-4313` | TCGA SVS | 115109 x 62467 | 0.252 | 14916 | 15600 | 1.046 | PASS |
| 7 | `TCGA-86-8074` | TCGA SVS | 91128 x 71141 | 0.252 | 16863 | 13651 | 0.810 | PASS |
| 8 | `001d865e...` | PANDA TIFF | 28672 x 34560 | 0.452018 | 941 | 941 | 1.000 | PASS |
| 9 | `TCGA-RM-A68W` | TCGA SVS | 57767 x 50248 | 0.2527 | 5234 | 5231 | 0.999 | PASS |
| 10 | `1079807` | SVS local slide | 53784 x 39410 | 0.5026 | 4651 | 6649 | 1.430 | PASS |

Observed ratio range: `0.810` to `3.333`.

Observed neural patch-count range: `30` to `18386`.

All failures fields were empty in the CSV summary.

### Existing neural/artifact integration suite

SLURM array job: `3563392`

Tasks: `34-37`

State: all `COMPLETED`

Exit code: all `0:0`

Covered tests:

- `test_neural_segmentation_produces_valid_patches`
- `test_neural_segmentation_patch_count_close_to_hsv`
- `test_grandqc_artifact_remover_runs_on_real_slide`
- `test_grandqc_artifact_remover_integrated_with_segment_tissue`

Logs:

- `/gpfs/cdsi_ess/home/limr/logs/slurm/test_integration_3563392_34.out`
- `/gpfs/cdsi_ess/home/limr/logs/slurm/test_integration_3563392_35.out`
- `/gpfs/cdsi_ess/home/limr/logs/slurm/test_integration_3563392_36.out`
- `/gpfs/cdsi_ess/home/limr/logs/slurm/test_integration_3563392_37.out`

### Tessellate/extract GPU E2E

SLURM job: `3563398`

State: `COMPLETED`

Exit code: `0:0`

Elapsed: `00:18:19`

Max RSS: `4860184K`

Log:

`/gpfs/cdsi_ess/home/limr/logs/slurm/test_tef_e2e_pr136_3563398.out`

Result: `6 passed, 11 deselected in 1088.87s`

This specifically confirms that the prior DataLoader spawned-worker circular import failure no longer reproduces when running the tessellate/extract E2E path on PR #136.

### Import-order regression

Command:

```bash
uv run pytest tests/mussel/test_import_order.py -q
```

Result: `3 passed in 38.77s`

## Risk Coverage

This validation materially reduces the chance of bugs slipping through by covering:

- 10 slides instead of a single fixture
- SVS, NDPI, and TIFF inputs
- slide widths from 11952 to 145920 pixels
- slide heights from 10249 to 77677 pixels
- MPP values from approximately 0.230 to 0.503
- patch-count outputs from sparse tissue (`30` neural patches) to dense tissue (`18386` neural patches)
- neural segmentation HDF5 persistence and coordinate validity
- comparison against classic segmentation as a broad sanity check
- tessellate/extract E2E execution with DataLoader workers enabled

## Limitations

This is a correctness and regression validation, not a pathologist-reviewed segmentation-quality study. The classic-vs-neural ratio is used only as a broad anomaly detector.

The panel includes local filesystem slides outside the repository. Re-running on another machine requires those paths or an equivalent manifest.

## Final Assessment

The fixed neural segmentation behavior passes the expanded correctness validation on PR #136 over current `main`.

Recommendation: treat PR #136 as ready to support the neural segmentation validation path, with the 10-slide panel retained as the broader regression check for future changes to segmentation level selection, neural mask rescaling, HDF5 coordinate output, or DataLoader/import behavior.
