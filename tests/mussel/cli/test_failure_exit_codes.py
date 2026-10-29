"""Failures fail: CLI runs with a failed slide, model, file or algorithm raise (exit non-zero).

Batch runs still process the other slides/models/files first; then they raise,
naming what failed. Segmentation, extraction and aggregation are mocked.
"""

import os
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

import mussel.cli.clustering_benchmark as cb
import mussel.cli.convert as convert_module
import mussel.cli.tessellate as tessellate_module
from mussel.cli.convert import ConvertConfig
from mussel.cli.tessellate import SegConfig, TessellateConfig
from mussel.cli.tessellate_extract_features import (
    TessellateExtractFeaturesConfig, main)
from mussel.models import ModelType
from mussel.utils import feature_extract as fe
from mussel.utils import save_torch_tensor
from mussel.utils.converter import AnyToTiffConverter

TEF = "mussel.cli.tessellate_extract_features"
COMMON = "mussel.cli.tessellate_extract_features_common"

_COORDS = np.array([[0, 0], [256, 0], [0, 256]], dtype=np.int64)
_PATCH_FEATURES = np.random.rand(3, 8).astype(np.float32)
_SLIDE_FEATURES = np.random.rand(1, 8).astype(np.float32)


def _segment_tissue(failing=()):
    """segment_tissue mock: no tissue (None) for slide IDs in ``failing``, else three tiles."""

    def fake(slide_path, slide_id, output_h5_path, **kwargs):
        if slide_id in failing:
            return None
        os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
        with h5py.File(output_h5_path, "w") as f:
            f.create_dataset("coords", data=_COORDS)
        return (MagicMock(), MagicMock(), _COORDS.tolist(), None)

    return fake


def _extract(patch_h5_paths, slide_paths, output_h5_paths, **kwargs):
    for path in output_h5_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            f.create_dataset("features", data=_PATCH_FEATURES)
            f.create_dataset("coords", data=_COORDS)


def _aggregate(patch_features_h5_paths, output_h5_paths=None, output_pt_paths=None, **kwargs):
    for path in output_pt_paths or []:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_torch_tensor(path, torch.from_numpy(_SLIDE_FEATURES))
    for path in output_h5_paths or []:
        with h5py.File(path, "w") as f:
            f.create_dataset("features", data=_SLIDE_FEATURES)


def _batch_cfg(tmp_path, slide_ids=("bad", "good"), **overrides):
    slide_paths = [str(tmp_path / f"{s}.svs") for s in slide_ids]
    return OmegaConf.create(TessellateExtractFeaturesConfig(
        slide_paths=slide_paths,
        slide_ids=list(slide_ids),
        output_dir=str(tmp_path / "out"),
        prefilter_model_type=ModelType.RESNET50,
        seg_config=SegConfig(),
        use_gpu=False,
        num_workers=0,
        **overrides,
    ))


def _run(cfg, segment=None, extract=_extract, aggregate=_aggregate):
    with (
        patch(f"{TEF}._build_segmentation_runtime", return_value=(None, None)),
        patch(f"{COMMON}.segment_tissue", side_effect=segment or _segment_tissue()),
        patch(f"{TEF}.extract_patch_features_batch", side_effect=extract),
        patch(f"{TEF}.aggregate_slide_features_batch", side_effect=aggregate),
    ):
        main(cfg)


# ── tessellate_extract_features ───────────────────────────────────────────────


def test_tef_single_slide_without_tissue_raises(tmp_path):
    cfg = OmegaConf.create(TessellateExtractFeaturesConfig(
        slide_path=str(tmp_path / "bad.svs"), slide_id="bad", output_dir=str(tmp_path / "out"),
        prefilter_model_type=ModelType.RESNET50, seg_config=SegConfig(), use_gpu=False,
    ))
    with pytest.raises(RuntimeError, match="Tessellation failed"):
        _run(cfg, segment=_segment_tissue(failing={"bad"}))


def test_tef_batch_tessellation_failure_processes_the_others_then_raises(tmp_path):
    cfg = _batch_cfg(tmp_path)
    with pytest.raises(RuntimeError, match="1 of 2 slide.*bad: tessellation failed") as info:
        _run(cfg, segment=_segment_tissue(failing={"bad"}))
    assert "good" not in str(info.value)
    out = tmp_path / "out"
    assert (out / "pt" / "good.features.pt").exists()
    assert (out / "h5" / "good.features.h5").exists()
    assert not (out / "pt" / "bad.features.pt").exists()


def test_tef_batch_all_tessellations_fail_raises(tmp_path):
    cfg = _batch_cfg(tmp_path)
    with pytest.raises(RuntimeError, match="2 of 2 slide"):
        _run(cfg, segment=_segment_tissue(failing={"bad", "good"}))


def test_tef_batch_extraction_error_raises(tmp_path):
    """Phase 2 used to log "Continuing with available results..." and exit 0."""
    cfg = _batch_cfg(tmp_path, aggregation_method="mean")
    with pytest.raises(RuntimeError, match="model failed to load"):
        _run(cfg, extract=MagicMock(side_effect=RuntimeError("model failed to load")))


def test_tef_batch_aggregation_error_raises(tmp_path):
    """Phase 3 used to swallow aggregate_slide_features_batch's errors, even "all slides failed"."""
    cfg = _batch_cfg(tmp_path, aggregation_method="mean")
    with pytest.raises(RuntimeError, match="All 2 slide"):
        _run(cfg, aggregate=MagicMock(side_effect=RuntimeError("All 2 slide(s) failed during 'mean' aggregation")))


@pytest.mark.parametrize("aggregation_method", ["identity", "mean"])
def test_tef_batch_one_slide_failing_extraction_keeps_the_others(tmp_path, aggregation_method):
    cfg = _batch_cfg(tmp_path, aggregation_method=aggregation_method)
    aggregated = []

    def extract(patch_h5_paths, slide_paths, output_h5_paths, **kwargs):
        _extract(patch_h5_paths[1:], slide_paths[1:], output_h5_paths[1:])
        raise fe.SlideBatchError("patch feature extraction", {0: "corrupt tile"}, slide_paths,
                                 [None] + list(output_h5_paths[1:]))

    def aggregate(patch_features_h5_paths, **kwargs):
        aggregated.extend(patch_features_h5_paths)
        _aggregate(patch_features_h5_paths, **kwargs)

    with pytest.raises(RuntimeError, match="1 of 2 slide.*bad: patch feature extraction failed: corrupt tile"):
        _run(cfg, extract=extract, aggregate=aggregate)
    out = tmp_path / "out"
    assert (out / "pt" / "good.features.pt").exists()
    assert not (out / "pt" / "bad.features.pt").exists()
    if aggregation_method == "mean":
        assert [os.path.basename(p) for p in aggregated] == ["good.patch.h5"]


@pytest.mark.parametrize("aggregation_method", ["identity", "mean"])
def test_tef_batch_slide_without_tiles_is_skipped_not_failed(tmp_path, caplog, aggregation_method):
    """Extraction writes nothing for a slide without tiles: skipped with a warning, and the run exits 0."""
    cfg = _batch_cfg(tmp_path, slide_ids=("empty", "good"), aggregation_method=aggregation_method)

    def extract(patch_h5_paths, slide_paths, output_h5_paths, **kwargs):
        _extract(patch_h5_paths[1:], slide_paths[1:], output_h5_paths[1:])

    _run(cfg, extract=extract, aggregate=fe.aggregate_slide_features_batch)  # no raise
    out = tmp_path / "out"
    assert (out / "pt" / "good.features.pt").exists()
    assert not (out / "pt" / "empty.features.pt").exists()
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("1 of 2 slide(s) skipped (no tiles)" in m for m in warnings)


def test_tef_multi_model_failure_runs_the_other_models_then_raises(tmp_path):
    cfg = _batch_cfg(tmp_path, model_type=["RESNET50", "CLIP"])
    seen = []

    def fake_main_batch(cfg_copy, patch_output_dir=None):
        seen.append(cfg_copy.model_type)
        if cfg_copy.model_type == ModelType.RESNET50:
            raise RuntimeError("out of memory")

    with patch(f"{TEF}._main_batch", side_effect=fake_main_batch):
        with pytest.raises(RuntimeError, match="1 of 2 model.*RESNET50: out of memory"):
            main(cfg)
    assert sorted(m.name for m in seen) == ["CLIP", "RESNET50"]


def test_tef_classifier_found_in_model_dir_is_used(tmp_path):
    """classifier.pkl in model_dir turned filtering on, but load_classifier was given None."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "classifier.pkl").write_bytes(b"")
    cfg = OmegaConf.create(TessellateExtractFeaturesConfig(
        slide_path=str(tmp_path / "s.svs"), slide_id="s", output_dir=str(tmp_path / "out"),
        prefilter_model_type=ModelType.RESNET50, seg_config=SegConfig(), use_gpu=False,
        model_dir=str(model_dir),
    ))

    class Loaded(Exception):
        pass

    with (
        patch(f"{TEF}._build_segmentation_runtime", return_value=(None, None)),
        patch(f"{COMMON}.segment_tissue", side_effect=_segment_tissue()),
        patch(f"{COMMON}.save_features"),
        patch(f"{COMMON}.load_classifier", side_effect=Loaded) as load_classifier,
    ):
        with pytest.raises(Loaded):
            main(cfg)
    load_classifier.assert_called_once_with(str(model_dir / "classifier.pkl"))


# ── tessellate ────────────────────────────────────────────────────────────────


def test_tessellate_single_slide_without_tissue_raises(tmp_path):
    out = tmp_path / "s.patch.h5"
    cfg = OmegaConf.structured(TessellateConfig(slide_path=str(tmp_path / "s.svs"), output_h5_path=str(out),
                                                   seg_config=SegConfig()))
    with patch.object(tessellate_module, "_run_tessellation", return_value={(224, 0.5): None}):
        with pytest.raises(RuntimeError, match="no patch H5"):
            tessellate_module._main(cfg)
    assert not out.exists()


# ── convert ───────────────────────────────────────────────────────────────────


def _png(path):
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(path)
    return str(path)


def test_convert_single_file_failure_raises(tmp_path):
    cfg = OmegaConf.structured(ConvertConfig(input_path=_png(tmp_path / "s.png"),
                                             output_dir=str(tmp_path / "out"), mpp=0.5))
    with patch.object(AnyToTiffConverter, "_try_pyvips_convert", side_effect=OSError("corrupt")):
        with pytest.raises(OSError, match="corrupt"):
            convert_module.main(cfg)


def test_convert_single_file_applies_downscale_by(tmp_path):
    cfg = OmegaConf.structured(ConvertConfig(input_path=_png(tmp_path / "s.png"),
                                             output_dir=str(tmp_path / "out"), mpp=0.25, downscale_by=2))
    with patch.object(AnyToTiffConverter, "process_file") as process_file:
        convert_module.main(cfg)
    assert process_file.call_args.kwargs["zoom"] == 0.5


def test_convert_rejects_downscale_by_below_one(tmp_path):
    cfg = OmegaConf.structured(ConvertConfig(input_path=_png(tmp_path / "s.png"),
                                             output_dir=str(tmp_path / "out"), mpp=0.25, downscale_by=0))
    with pytest.raises(ValueError, match="downscale_by"):
        convert_module.main(cfg)


def test_convert_batch_failure_converts_the_others_then_raises(tmp_path):
    wsis = tmp_path / "wsis"
    wsis.mkdir()
    _png(wsis / "bad.png")
    _png(wsis / "good.png")
    csv = tmp_path / "mpp.csv"
    csv.write_text("wsi,mpp\nbad.png,0.5\ngood.png,0.5\n")
    converted = []

    def fake_convert(self, input_file, save_path, zoom, output_mpp):
        if "bad" in input_file:
            raise OSError("corrupt")
        converted.append(os.path.basename(input_file))
        return True

    cfg = OmegaConf.structured(ConvertConfig(input_path=str(wsis), output_dir=str(tmp_path / "out"),
                                             mpp_csv=str(csv), num_workers=1))
    with patch.object(AnyToTiffConverter, "_try_pyvips_convert", fake_convert):
        with pytest.raises(RuntimeError, match=r"1 of 2 file.*bad\.png"):
            convert_module.main(cfg)
    assert converted == ["good.png"]


# ── clustering_benchmark ──────────────────────────────────────────────────────


def _clustering_cfg(tmp_path, algorithms):
    rng = np.random.default_rng(0)
    rows = 40
    df = {
        "slide_id": [f"s{i % 4}" for i in range(rows)],
        "annotation": [1 + i % 2 for i in range(rows)],
        "overlap_area": [0.9] * rows,
        "tile_area": [1.0] * rows,
    }
    for j in range(4):
        df[f"feature_{j}"] = rng.standard_normal(rows)

    parquet = tmp_path / "f.parquet"
    pd.DataFrame(df).to_parquet(parquet)
    return OmegaConf.structured(cb.ClusteringBenchmarkConfig(
        features_annotation_parquet_path=str(parquet),
        output_metrics_csv=str(tmp_path / "metrics.csv"),
        output_summary_json=str(tmp_path / "summary.json"),
        output_umap_png=str(tmp_path / "umap.png"),
        algorithms=algorithms,
        n_clusters=2,
    ))


def test_clustering_unknown_algorithm_raises_before_any_output(tmp_path):
    cfg = _clustering_cfg(tmp_path, ["kmeans", "kmeanz"])
    with pytest.raises(ValueError, match="Unknown algorithm 'kmeanz'"):
        cb.main.__wrapped__(cfg)
    assert not os.path.exists(cfg.output_metrics_csv)


def test_clustering_no_algorithm_succeeding_raises(tmp_path):
    cfg = _clustering_cfg(tmp_path, ["kmeans"])
    with patch.object(cb, "_fit_predict", side_effect=MemoryError("too big")):
        with pytest.raises(RuntimeError, match="No clustering algorithm succeeded.*kmeans"):
            cb.main.__wrapped__(cfg)
    assert not os.path.exists(cfg.output_metrics_csv)
