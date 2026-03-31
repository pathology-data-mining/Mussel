import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from shapely.geometry import box

from mussel.cli.linear_probe_benchmark import (
    LinearProbeBenchmarkConfig,
    _eval_split,
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


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def _cfg(tmp_path, parquet_path, **overrides):
    defaults = dict(
        features_annotation_parquet_path=parquet_path,
        output_csv=str(tmp_path / "report_val.csv"),
        output_png=str(tmp_path / "cm_val.png"),
        output_test_csv=str(tmp_path / "report_test.csv"),
        output_test_png=str(tmp_path / "cm_test.png"),
        cv=2,
        C_values=[0.1, 1.0],
        penalties=["l2"],
        max_iter=200,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        annotation_percent_filter_threshold=0.5,
    )
    defaults.update(overrides)
    return OmegaConf.structured(LinearProbeBenchmarkConfig(**defaults))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_main_runs_and_outputs_files(tmp_path):
    """main() produces val and test CSV/PNG outputs."""
    parquet_path = _make_parquet(tmp_path)
    cfg = _cfg(tmp_path, parquet_path)
    main(cfg)

    assert os.path.exists(cfg.output_csv), "val classification report not written"
    assert os.path.exists(cfg.output_png), "val confusion matrix not written"
    assert os.path.exists(cfg.output_test_csv), "test classification report not written"
    assert os.path.exists(cfg.output_test_png), "test confusion matrix not written"


def test_test_set_metrics_in_report(tmp_path):
    """Test CSV report contains AUC-ROC and average precision rows."""
    parquet_path = _make_parquet(tmp_path)
    cfg = _cfg(tmp_path, parquet_path)
    main(cfg)

    report = pd.read_csv(cfg.output_test_csv, index_col=0)
    assert "auc_roc" in report.index, "auc_roc missing from test report"
    assert "average_precision" in report.index, "average_precision missing from test report"


def test_val_metrics_in_report(tmp_path):
    """Val CSV report contains AUC-ROC and average precision rows."""
    parquet_path = _make_parquet(tmp_path)
    cfg = _cfg(tmp_path, parquet_path)
    main(cfg)

    report = pd.read_csv(cfg.output_csv, index_col=0)
    assert "auc_roc" in report.index
    assert "average_precision" in report.index


def test_stratified_split_preserves_both_classes(tmp_path):
    """Both classes should appear in val and test splits after stratification."""
    parquet_path = _make_parquet(tmp_path, n_slides=30)
    cfg = _cfg(tmp_path, parquet_path)
    main(cfg)

    # If splits were unstratified we could end up with single-class splits;
    # verify reports have both class columns (0 and 1).
    for csv_path in [cfg.output_csv, cfg.output_test_csv]:
        report = pd.read_csv(csv_path, index_col=0)
        assert "0" in report.columns or 0 in report.columns, f"class 0 missing in {csv_path}"
        assert "1" in report.columns or 1 in report.columns, f"class 1 missing in {csv_path}"


def test_eval_split_returns_metrics(tmp_path):
    """_eval_split returns a dict with the three expected metric keys."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 4))
    y = (X[:, 0] > 0).astype(int)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(X, y)

    metrics = _eval_split(
        pipe, X, y,
        str(tmp_path / "r.csv"),
        str(tmp_path / "cm.png"),
        "test",
    )

    assert set(metrics.keys()) == {"f1", "auc_roc", "average_precision"}
    assert 0.0 <= metrics["auc_roc"] <= 1.0
    assert 0.0 <= metrics["average_precision"] <= 1.0


def test_grid_search_selects_best_params(tmp_path):
    """GridSearchCV should find a best_params_ entry — smoke test that it ran."""
    parquet_path = _make_parquet(tmp_path, n_slides=20)
    cfg = _cfg(tmp_path, parquet_path, C_values=[0.01, 1.0], penalties=["l2"])
    # If grid search ran, both CSV files should exist and be non-empty
    main(cfg)
    for p in [cfg.output_csv, cfg.output_test_csv]:
        assert os.path.getsize(p) > 0, f"{p} is empty"
