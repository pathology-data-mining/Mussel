import json
import math
import os
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from shapely.geometry import box

from mussel.cli.linear_probe_benchmark import (
    LinearProbeBenchmarkConfig,
    _bootstrap_ci_auc,
    _compute_metrics,
    _plot_gs_heatmap,
    _split_by_slide,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parquet(tmp_path, n_slides=20, tiles_per_slide=10, n_features=8, seed=0):
    """Create a synthetic GeoParquet file with feature columns and annotations."""
    rng = np.random.default_rng(seed)
    rows = []
    for slide_idx in range(n_slides):
        slide_id = f"slide_{slide_idx:03d}"
        annotation = 2 if slide_idx < n_slides // 2 else 1
        for tile_idx in range(tiles_per_slide):
            feature_vals = rng.standard_normal(n_features)
            row = {
                "slide_id": slide_id,
                "annotation": annotation,
                "overlap_area": 0.8,
                "tile_area": 1.0,
                "geometry": box(tile_idx * 256, 0, (tile_idx + 1) * 256, 256),
            }
            for i, v in enumerate(feature_vals):
                row[f"feature_{i}"] = v
            rows.append(row)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry")
    path = str(tmp_path / "features.parquet")
    gdf.to_parquet(path)
    return path


def _make_df(n_slides=20, tiles_per_slide=10, n_features=4, seed=0):
    """Return an in-memory GeoDataFrame (no file I/O) for unit tests."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_slides):
        slide_id = f"slide_{s:03d}"
        annotation = 2 if s < n_slides // 2 else 1
        for t in range(tiles_per_slide):
            row = {
                "slide_id": slide_id,
                "annotation": annotation,
                "overlap_area": 0.8,
                "tile_area": 1.0,
                "geometry": box(t * 256, 0, (t + 1) * 256, 256),
            }
            for i, v in enumerate(rng.standard_normal(n_features)):
                row[f"feature_{i}"] = v
            rows.append(row)
    return gpd.GeoDataFrame(rows, geometry="geometry")


def _cfg(tmp_path, parquet_path, **overrides):
    defaults = dict(
        features_annotation_parquet_path=parquet_path,
        output_csv=str(tmp_path / "report_val.csv"),
        output_png=str(tmp_path / "cm_val.png"),
        output_test_csv=str(tmp_path / "report_test.csv"),
        output_test_png=str(tmp_path / "cm_test.png"),
        output_roc_png=str(tmp_path / "roc.png"),
        output_pr_png=str(tmp_path / "pr.png"),
        output_gs_heatmap_png=str(tmp_path / "gs_heatmap.png"),
        output_feature_importance_png=str(tmp_path / "feature_importance.png"),
        output_calibration_png=str(tmp_path / "calibration.png"),
        output_cv_results_csv=str(tmp_path / "cv_results.csv"),
        output_summary_json=str(tmp_path / "results.json"),
        cv=2,
        C_values=[0.1, 1.0],
        penalties=["l2"],
        max_iter=200,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        annotation_percent_filter_threshold=0.5,
        n_seeds=1,
        n_bootstrap=50,
        n_top_features=4,
    )
    defaults.update(overrides)
    return OmegaConf.structured(LinearProbeBenchmarkConfig(**defaults))


# ---------------------------------------------------------------------------
# Tests — main() integration
# ---------------------------------------------------------------------------


def test_main_runs_and_outputs_files(tmp_path):
    """main() produces all expected output files."""
    parquet_path = _make_parquet(tmp_path)
    cfg = _cfg(tmp_path, parquet_path)
    main(cfg)

    expected = [
        cfg.output_csv,
        cfg.output_png,
        cfg.output_test_csv,
        cfg.output_test_png,
        cfg.output_roc_png,
        cfg.output_pr_png,
        cfg.output_gs_heatmap_png,
        cfg.output_feature_importance_png,
        cfg.output_calibration_png,
        cfg.output_cv_results_csv,
        cfg.output_summary_json,
    ]
    for path in expected:
        assert os.path.exists(path), f"expected output not written: {path}"


def test_summary_json_structure(tmp_path):
    """results.json has the expected top-level keys and metric structure."""
    parquet_path = _make_parquet(tmp_path)
    cfg = _cfg(tmp_path, parquet_path)
    main(cfg)

    with open(cfg.output_summary_json) as f:
        summary = json.load(f)

    assert "n_seeds" in summary
    assert "best_params" in summary
    assert "best_cv_auc" in summary
    for split in ("val", "test"):
        assert split in summary
        for key in ("tile_f1", "tile_auc_roc", "tile_average_precision"):
            assert key in summary[split], f"{key} missing from summary[{split}]"
            assert "mean" in summary[split][key]
            assert "std" in summary[split][key]
    # Bootstrap CI stored on test tile_auc_roc
    assert "bootstrap_ci_95" in summary["test"]["tile_auc_roc"]
    lo, hi = summary["test"]["tile_auc_roc"]["bootstrap_ci_95"]
    assert 0.0 <= lo <= hi <= 1.0


def test_summary_json_slide_metrics(tmp_path):
    """results.json includes slide-level metrics when both classes are present."""
    parquet_path = _make_parquet(tmp_path, n_slides=30)
    cfg = _cfg(tmp_path, parquet_path)
    main(cfg)

    with open(cfg.output_summary_json) as f:
        summary = json.load(f)

    for split in ("val", "test"):
        assert "slide_auc_roc" in summary[split], f"slide_auc_roc missing from {split}"


def test_cv_results_csv(tmp_path):
    """cv_results.csv is written and contains expected GridSearchCV columns."""
    parquet_path = _make_parquet(tmp_path)
    cfg = _cfg(tmp_path, parquet_path)
    main(cfg)

    cv_df = pd.read_csv(cfg.output_cv_results_csv)
    assert "mean_test_score" in cv_df.columns
    assert "param_clf__C" in cv_df.columns
    assert len(cv_df) == len(cfg.C_values) * len(cfg.penalties)


def test_classification_report_contains_extra_rows(tmp_path):
    """Val and test CSV reports contain auc_roc and average_precision rows."""
    parquet_path = _make_parquet(tmp_path)
    cfg = _cfg(tmp_path, parquet_path)
    main(cfg)

    for path in (cfg.output_csv, cfg.output_test_csv):
        report = pd.read_csv(path, index_col=0)
        assert "auc_roc" in report.index, f"auc_roc missing in {path}"
        assert "average_precision" in report.index, f"average_precision missing in {path}"


def test_multi_seed_aggregation(tmp_path):
    """With n_seeds>1, summary JSON std values are populated (non-NaN)."""
    parquet_path = _make_parquet(tmp_path, n_slides=30)
    cfg = _cfg(tmp_path, parquet_path, n_seeds=3)
    main(cfg)

    with open(cfg.output_summary_json) as f:
        summary = json.load(f)

    assert summary["n_seeds"] == 3
    std = summary["test"]["tile_auc_roc"]["std"]
    assert std >= 0.0  # std is defined (not NaN) across seeds


def test_grid_search_heatmap_multi_penalty(tmp_path):
    """Grid search heatmap is written for a multi-penalty search."""
    parquet_path = _make_parquet(tmp_path, n_slides=30)
    cfg = _cfg(tmp_path, parquet_path, penalties=["l1", "l2"], C_values=[0.1, 1.0])
    main(cfg)
    assert os.path.getsize(cfg.output_gs_heatmap_png) > 0


# ---------------------------------------------------------------------------
# Tests — unit helpers
# ---------------------------------------------------------------------------


def test_compute_metrics_returns_expected_keys():
    """_compute_metrics returns tile- and slide-level metric keys."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = _make_df(n_slides=10, tiles_per_slide=10, n_features=4)
    df = df.assign(y=(df.annotation == 2).astype(int))

    X = df.filter(regex="feature_").values
    y = df["y"].values
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(X, y)
    y_prob = pipe.predict_proba(X)[:, 1]

    metrics = _compute_metrics(df, y_prob, "test")

    for key in ("tile_f1", "tile_auc_roc", "tile_average_precision"):
        assert key in metrics, f"{key} missing"
        assert 0.0 <= metrics[key] <= 1.0

    # slide metrics present when both classes exist
    for key in ("slide_f1", "slide_auc_roc", "slide_average_precision"):
        assert key in metrics, f"{key} missing"


def test_split_by_slide_no_overlap():
    """_split_by_slide produces non-overlapping slide sets."""
    df = _make_df(n_slides=20, tiles_per_slide=5)
    df = df.assign(y=(df.annotation == 2).astype(int))

    train_df, val_df, test_df = _split_by_slide(df, test_size=0.2, val_size=0.1, random_state=0)

    train_slides = set(train_df["slide_id"])
    val_slides = set(val_df["slide_id"])
    test_slides = set(test_df["slide_id"])

    assert train_slides.isdisjoint(val_slides), "train/val overlap"
    assert train_slides.isdisjoint(test_slides), "train/test overlap"
    assert val_slides.isdisjoint(test_slides), "val/test overlap"
    assert train_slides | val_slides | test_slides == set(df["slide_id"])


def test_bootstrap_ci_auc_bounds():
    """_bootstrap_ci_auc returns a valid [lo, hi] interval within [0, 1]."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4))
    y = (X[:, 0] > 0).astype(int)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(X, y)
    y_prob = pipe.predict_proba(X)[:, 1]

    lo, hi = _bootstrap_ci_auc(y_prob, y, n_bootstrap=100, random_state=0)
    assert 0.0 <= lo <= hi <= 1.0


def test_positive_annotation_label_one(tmp_path):
    """positive_annotation_label=1 treats annotation==1 as positive (class-mapped 0/1 data)."""
    # Simulate post-class-mapping parquet: annotation is already 0 or 1
    rng = np.random.default_rng(7)
    rows = []
    for s in range(20):
        slide_id = f"slide_{s:03d}"
        annotation = 1 if s < 10 else 0  # 0/1 labels, not 1/2
        for t in range(10):
            row = {
                "slide_id": slide_id,
                "annotation": annotation,
                "overlap_area": 0.8,
                "tile_area": 1.0,
                "geometry": box(t * 256, 0, (t + 1) * 256, 256),
            }
            for i, v in enumerate(rng.standard_normal(8)):
                row[f"feature_{i}"] = v
            rows.append(row)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry")
    parquet_path = str(tmp_path / "features_01.parquet")
    gdf.to_parquet(parquet_path)

    cfg = _cfg(tmp_path, parquet_path, positive_annotation_label=1, n_seeds=1, n_bootstrap=10)
    main(cfg)

    with open(cfg.output_summary_json) as f:
        summary = json.load(f)

    # Both classes should be present, so AUC is meaningful (> 0)
    assert summary["test"]["tile_auc_roc"]["mean"] > 0.0


# ---------------------------------------------------------------------------
# Helpers — multiclass
# ---------------------------------------------------------------------------


def _make_multiclass_parquet(
    tmp_path,
    n_classes=3,
    n_slides_per_class=12,
    tiles_per_slide=10,
    n_features=8,
    seed=0,
    with_background=False,
):
    """Create a GeoParquet with *n_classes* non-zero annotation values.

    Each class gets its own dedicated slides.  If *with_background* is True,
    an extra slide whose tiles all carry annotation=0 is appended so that
    background-exclusion tests can verify that those rows are dropped.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cls in range(1, n_classes + 1):
        for slide_idx in range(n_slides_per_class):
            slide_id = f"slide_c{cls}_{slide_idx:03d}"
            for tile_idx in range(tiles_per_slide):
                row = {
                    "slide_id": slide_id,
                    "annotation": cls,
                    "overlap_area": 0.8,
                    "tile_area": 1.0,
                    "geometry": box(tile_idx * 256, 0, (tile_idx + 1) * 256, 256),
                }
                for i, v in enumerate(rng.standard_normal(n_features)):
                    row[f"feature_{i}"] = v
                rows.append(row)
    if with_background:
        for tile_idx in range(tiles_per_slide):
            row = {
                "slide_id": "slide_background_000",
                "annotation": 0,
                "overlap_area": 0.8,
                "tile_area": 1.0,
                "geometry": box(tile_idx * 256, 0, (tile_idx + 1) * 256, 256),
            }
            for i, v in enumerate(rng.standard_normal(n_features)):
                row[f"feature_{i}"] = v
            rows.append(row)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry")
    path = str(tmp_path / "multiclass_features.parquet")
    gdf.to_parquet(path)
    return path


def _cfg_multiclass(tmp_path, parquet_path, **overrides):
    """Return a config with multiclass=True."""
    return _cfg(tmp_path, parquet_path, multiclass=True, **overrides)


# ---------------------------------------------------------------------------
# Tests — multiclass integration
# ---------------------------------------------------------------------------


def test_main_multiclass_runs_and_outputs_files(tmp_path):
    """main() with multiclass=True produces all expected output files."""
    parquet_path = _make_multiclass_parquet(tmp_path)
    cfg = _cfg_multiclass(tmp_path, parquet_path)
    main(cfg)

    expected = [
        cfg.output_csv,
        cfg.output_png,
        cfg.output_test_csv,
        cfg.output_test_png,
        cfg.output_roc_png,
        cfg.output_pr_png,
        cfg.output_gs_heatmap_png,
        cfg.output_feature_importance_png,
        cfg.output_calibration_png,
        cfg.output_cv_results_csv,
        cfg.output_summary_json,
    ]
    for path in expected:
        assert os.path.exists(path), f"expected output not written: {path}"


def test_main_multiclass_summary_json_structure(tmp_path):
    """multiclass summary JSON has tile metrics but omits binary-only keys."""
    parquet_path = _make_multiclass_parquet(tmp_path)
    cfg = _cfg_multiclass(tmp_path, parquet_path)
    main(cfg)

    with open(cfg.output_summary_json) as f:
        summary = json.load(f)

    for split in ("val", "test"):
        assert split in summary
        assert "tile_f1" in summary[split]
        assert "tile_auc_roc" in summary[split]
        # binary-only metrics must be absent in multiclass mode
        assert "tile_average_precision" not in summary[split], (
            f"tile_average_precision should not appear in multiclass {split}"
        )
        assert "slide_auc_roc" not in summary[split], (
            f"slide_auc_roc should not appear in multiclass {split}"
        )

    # Bootstrap CI must be present on the test split
    ci = summary["test"]["tile_auc_roc"].get("bootstrap_ci_95")
    assert ci is not None, "bootstrap_ci_95 missing from multiclass test AUC"
    lo, hi = ci
    # lo/hi serialise as null when bootstrap found no valid samples; otherwise check bounds
    if lo is not None and hi is not None:
        assert 0.0 <= lo <= hi <= 1.0


def test_main_multiclass_report_csv_omits_average_precision(tmp_path):
    """multiclass confusion-matrix CSV includes auc_roc but not average_precision."""
    parquet_path = _make_multiclass_parquet(tmp_path)
    cfg = _cfg_multiclass(tmp_path, parquet_path)
    main(cfg)

    for path in (cfg.output_csv, cfg.output_test_csv):
        report = pd.read_csv(path, index_col=0)
        assert "auc_roc" in report.index, f"auc_roc missing in {path}"
        assert "average_precision" not in report.index, (
            f"average_precision should not appear in multiclass report {path}"
        )


def test_main_multiclass_excludes_background_tiles(tmp_path):
    """Background tiles (annotation == 0) are excluded from the training matrix."""
    from sklearn.model_selection import GridSearchCV

    parquet_path = _make_multiclass_parquet(tmp_path, with_background=True)
    cfg = _cfg_multiclass(tmp_path, parquet_path)

    captured = {}
    original_fit = GridSearchCV.fit

    def capturing_fit(self, X, y, **kwargs):
        if "y_train" not in captured:
            captured["y_train"] = y.copy()
        return original_fit(self, X, y, **kwargs)

    with patch.object(GridSearchCV, "fit", capturing_fit):
        main(cfg)

    assert "y_train" in captured, "GridSearchCV.fit was never called"
    assert 0 not in np.unique(captured["y_train"]), (
        "background class 0 must be excluded from the training set"
    )


def test_main_multiclass_multi_seed(tmp_path):
    """multiclass mode with n_seeds=2 aggregates metrics across seeds correctly."""
    parquet_path = _make_multiclass_parquet(tmp_path)
    cfg = _cfg_multiclass(tmp_path, parquet_path, n_seeds=2)
    main(cfg)

    with open(cfg.output_summary_json) as f:
        summary = json.load(f)

    assert summary["n_seeds"] == 2
    for split in ("val", "test"):
        assert summary[split]["tile_auc_roc"]["std"] >= 0.0


# ---------------------------------------------------------------------------
# Tests — multiclass unit helpers
# ---------------------------------------------------------------------------


def test_compute_metrics_multiclass_returns_expected_keys():
    """_compute_metrics with 2-D y_prob returns tile keys only; no slide/AP keys."""
    rng = np.random.default_rng(5)
    n = 90
    y = np.concatenate(
        [np.ones(30, dtype=int), np.full(30, 2, dtype=int), np.full(30, 3, dtype=int)]
    )
    classes = np.array([1, 2, 3])

    # Build an informative probability matrix: each class gets a boost on its column.
    y_prob = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    for i, c in enumerate(classes):
        y_prob[y == c, i] += 1.0
    y_prob /= y_prob.sum(axis=1, keepdims=True)

    df = pd.DataFrame({"slide_id": [f"s{i}" for i in range(n)], "y": y})
    metrics = _compute_metrics(df, y_prob, "test", classes=classes)

    assert "tile_f1" in metrics
    assert "tile_auc_roc" in metrics
    assert 0.0 <= metrics["tile_f1"] <= 1.0
    assert 0.0 <= metrics["tile_auc_roc"] <= 1.0
    # binary-only metrics must not appear in multiclass output
    assert "tile_average_precision" not in metrics
    assert "slide_f1" not in metrics
    assert "slide_auc_roc" not in metrics
    assert "slide_average_precision" not in metrics


def test_compute_metrics_multiclass_absent_class():
    """_compute_metrics handles a split where one class is entirely absent."""
    from sklearn.metrics import roc_auc_score as _roc_auc

    rng = np.random.default_rng(6)
    n = 60
    # y has only classes 1 and 2; class 3 is absent from this split
    y = np.concatenate([np.ones(30, dtype=int), np.full(30, 2, dtype=int)])
    classes = np.array([1, 2, 3])

    # 3-column probability matrix as would come from a model trained on all 3 classes
    y_prob = rng.dirichlet([1.0, 1.0, 0.1], size=n)

    df = pd.DataFrame({"slide_id": [f"s{i}" for i in range(n)], "y": y})
    metrics = _compute_metrics(df, y_prob, "test", classes=classes)

    assert "tile_f1" in metrics
    assert "tile_auc_roc" in metrics
    assert not math.isnan(metrics["tile_auc_roc"])

    # AUC must be the macro mean of OvR AUCs for the *present* classes only (1 and 2).
    expected_auc = float(
        np.mean(
            [
                _roc_auc((y == 1).astype(int), y_prob[:, 0]),
                _roc_auc((y == 2).astype(int), y_prob[:, 1]),
            ]
        )
    )
    assert abs(metrics["tile_auc_roc"] - expected_auc) < 1e-6


def test_bootstrap_ci_auc_multiclass():
    """_bootstrap_ci_auc with 3-class 2-D probabilities returns valid bounds."""
    rng = np.random.default_rng(7)
    n = 90
    y = np.concatenate(
        [np.ones(30, dtype=int), np.full(30, 2, dtype=int), np.full(30, 3, dtype=int)]
    )
    classes = np.array([1, 2, 3])

    # Informative probability matrix
    y_prob = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    for i, c in enumerate(classes):
        y_prob[y == c, i] += 1.0
    y_prob /= y_prob.sum(axis=1, keepdims=True)

    lo, hi = _bootstrap_ci_auc(y_prob, y, n_bootstrap=100, random_state=0, classes=classes)
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_auc_multiclass_absent_class():
    """When a class is entirely absent from y, bootstrap CI returns (nan, nan).

    The current implementation requires all *n_classes* classes to appear in every
    bootstrap sample.  When fewer than n_classes unique labels exist in y, every
    bootstrap draw fails the uniqueness check, scores is empty, and (nan, nan) is
    returned.  This test documents that known behaviour.
    """
    rng = np.random.default_rng(8)
    n = 60
    # y has only classes 1 and 2; class 3 is absent
    y = np.concatenate([np.ones(30, dtype=int), np.full(30, 2, dtype=int)])
    classes = np.array([1, 2, 3])
    y_prob = rng.dirichlet([1.0, 1.0, 0.1], size=n)

    lo, hi = _bootstrap_ci_auc(y_prob, y, n_bootstrap=100, random_state=0, classes=classes)
    assert math.isnan(lo) and math.isnan(hi)
