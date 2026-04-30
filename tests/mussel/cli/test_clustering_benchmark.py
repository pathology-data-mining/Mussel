import json
import math
import os
import sys
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from shapely.geometry import box

from mussel.cli.clustering_benchmark import (
    ClusteringBenchmarkConfig,
    _cluster_purity,
    _compute_cluster_metrics,
    _fit_predict,
    _sanitize_for_json,
    _slide_level_metrics,
    main,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


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


def _make_df(n_slides=20, tiles_per_slide=10, n_features=4, seed=0, multiclass=False):
    """Return an in-memory DataFrame (no file I/O) for unit tests."""
    rng = np.random.default_rng(seed)
    rows = []
    n_classes = 3 if multiclass else 2
    for s in range(n_slides):
        slide_id = f"slide_{s:03d}"
        annotation = (s % n_classes) + 1  # 1, 2, or 3 (no zero)
        for t in range(tiles_per_slide):
            row = {
                "slide_id": slide_id,
                "annotation": annotation,
                "overlap_area": 0.8,
                "tile_area": 1.0,
            }
            for i, v in enumerate(rng.standard_normal(n_features)):
                row[f"feature_{i}"] = v
            rows.append(row)
    df = pd.DataFrame(rows)
    df["y"] = df["annotation"].astype(int)
    return df


def _cfg(tmp_path, parquet_path, **overrides):
    defaults = dict(
        features_annotation_parquet_path=parquet_path,
        output_metrics_csv=str(tmp_path / "metrics.csv"),
        output_summary_json=str(tmp_path / "results.json"),
        output_umap_png=str(tmp_path / "umap.png"),
        annotation_percent_filter_threshold=0.5,
        positive_annotation_label=2,
        multiclass=False,
        algorithms=["kmeans"],
        n_clusters=2,
        dbscan_eps=0.5,
        dbscan_min_samples=5,
        umap_n_neighbors=5,
        umap_min_dist=0.1,
        umap_n_components=2,
        umap_subsample=50,
        random_state=42,
    )
    defaults.update(overrides)
    return OmegaConf.structured(ClusteringBenchmarkConfig(**defaults))


def _fake_umap_module(n_rows: int) -> MagicMock:
    """Return a mock umap module whose UMAP().fit_transform() returns zeros."""
    fake = MagicMock()
    fake.UMAP.return_value.fit_transform.return_value = np.zeros((n_rows, 2))
    return fake


# ── Unit tests: _cluster_purity ───────────────────────────────────────────────


def test_cluster_purity_perfect():
    """Purity is 1.0 when each cluster contains only one class."""
    lt = np.array([0, 0, 0, 1, 1, 1])
    lp = np.array([0, 0, 0, 1, 1, 1])
    assert _cluster_purity(lt, lp) == pytest.approx(1.0)


def test_cluster_purity_random():
    """Purity lies in [0, 1] for a random assignment."""
    rng = np.random.default_rng(0)
    lt = rng.integers(0, 2, size=100)
    lp = rng.integers(0, 3, size=100)
    p = _cluster_purity(lt, lp)
    assert 0.0 <= p <= 1.0


def test_cluster_purity_excludes_noise():
    """DBSCAN noise tiles (label -1) are excluded from purity."""
    lt = np.array([0, 0, 1, 1])
    lp = np.array([-1, 0, 1, -1])  # two noise points
    # Only (lt=0, lp=0) and (lt=1, lp=1) contribute; both match → purity = 1.0
    assert _cluster_purity(lt, lp) == pytest.approx(1.0)


def test_cluster_purity_all_noise():
    """Purity is NaN when every tile is labelled noise."""
    lt = np.array([0, 1, 0])
    lp = np.array([-1, -1, -1])
    assert math.isnan(_cluster_purity(lt, lp))


def test_cluster_purity_multiclass():
    """Purity works correctly with labels 1, 2, 3 (non-zero multiclass)."""
    lt = np.array([1, 1, 2, 2, 3, 3])
    lp = np.array([0, 0, 1, 1, 2, 2])  # perfect 1-to-1 cluster-class mapping
    assert _cluster_purity(lt, lp) == pytest.approx(1.0)


# ── Unit tests: _compute_cluster_metrics ─────────────────────────────────────


def test_compute_cluster_metrics_binary():
    """Metrics dict contains expected keys and values in range for a simple binary case."""
    rng = np.random.default_rng(1)
    n = 100
    X = rng.standard_normal((n, 8)).astype(np.float32)
    lt = (rng.random(n) > 0.5).astype(int)
    lp = lt.copy()  # perfect clustering
    m = _compute_cluster_metrics(lt, lp, X, "tile", "test_algo")
    assert "nmi" in m and "ari" in m and "purity" in m and "silhouette" in m
    assert m["nmi"] == pytest.approx(1.0, abs=1e-6)
    assert m["ari"] == pytest.approx(1.0, abs=1e-6)
    assert m["purity"] == pytest.approx(1.0, abs=1e-6)


def test_compute_cluster_metrics_all_noise():
    """When all predictions are -1 (DBSCAN noise), metrics are NaN."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    lt = np.ones(20, dtype=int)
    lp = np.full(20, -1)
    m = _compute_cluster_metrics(lt, lp, X, "tile", "dbscan")
    assert math.isnan(m["nmi"])
    assert math.isnan(m["ari"])
    assert math.isnan(m["purity"])
    assert m["n_clusters_found"] == 0


# ── Unit tests: _slide_level_metrics ─────────────────────────────────────────


def test_slide_level_metrics_basic():
    """Slide-level metrics keys are present and values are in valid range."""
    df = _make_df(n_slides=10, tiles_per_slide=5, n_features=4)
    # Cluster 0 for first 5 slides, cluster 1 for last 5
    labels = np.array([0] * 25 + [1] * 25)
    m = _slide_level_metrics(df, labels, "kmeans")
    assert "slide_nmi" in m
    assert -1.0 <= m["slide_ari"] <= 1.0
    assert 0.0 <= m["slide_purity"] <= 1.0


def test_slide_level_metrics_single_class():
    """Returns empty dict when all slides have the same label (degenerate case)."""
    df = _make_df(n_slides=10, tiles_per_slide=5, n_features=4)
    df["y"] = 1  # override: single class
    labels = np.zeros(len(df), dtype=int)
    m = _slide_level_metrics(df, labels, "kmeans")
    assert m == {}


# ── Unit tests: _fit_predict ──────────────────────────────────────────────────


@pytest.mark.parametrize("algo", ["kmeans", "hierarchical", "dbscan"])
def test_fit_predict_output_shape(algo):
    """fit_predict returns an integer array with the same length as input."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((50, 8)).astype(np.float32)
    cfg = ClusteringBenchmarkConfig(
        features_annotation_parquet_path="dummy",
        n_clusters=2,
        dbscan_eps=1.0,
        dbscan_min_samples=3,
    )
    labels = _fit_predict(algo, X, cfg)
    assert labels.shape == (50,)
    assert labels.dtype.kind in ("i", "u")  # integer type


def test_fit_predict_unknown_algo():
    """Unknown algorithm name raises ValueError."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    cfg = ClusteringBenchmarkConfig(features_annotation_parquet_path="dummy")
    with pytest.raises(ValueError, match="Unknown algorithm"):
        _fit_predict("neural_chaos", X, cfg)


# ── Unit tests: _sanitize_for_json ───────────────────────────────────────────


def test_sanitize_nan_inf():
    obj = {"a": float("nan"), "b": float("inf"), "c": 1.0, "d": [float("nan"), 2.0]}
    clean = _sanitize_for_json(obj)
    assert clean["a"] is None
    assert clean["b"] is None
    assert clean["c"] == pytest.approx(1.0)
    assert clean["d"][0] is None
    assert clean["d"][1] == pytest.approx(2.0)


# ── Unit tests: validation ────────────────────────────────────────────────────


def test_main_raises_on_invalid_umap_components(tmp_path):
    """main() raises ValueError when umap_n_components != 2."""
    parquet_path = _make_parquet(tmp_path)
    cfg = _cfg(tmp_path, parquet_path, umap_n_components=3)
    import mussel.cli.clustering_benchmark as cb

    with pytest.raises(ValueError, match="umap_n_components must be 2"):
        cb.main.__wrapped__(cfg)


# ── Integration tests: main() ─────────────────────────────────────────────────


def test_main_end_to_end_kmeans(tmp_path):
    """Full main() invocation with kmeans produces all expected output files."""
    parquet_path = _make_parquet(tmp_path)
    n_tiles = 200  # 20 slides × 10 tiles
    cfg = _cfg(tmp_path, parquet_path, algorithms=["kmeans"], umap_subsample=0)
    import mussel.cli.clustering_benchmark as cb

    with patch.dict(sys.modules, {"umap": _fake_umap_module(n_tiles)}):
        cb.main.__wrapped__(cfg)

    assert os.path.exists(cfg.output_metrics_csv)
    assert os.path.exists(cfg.output_summary_json)

    with open(cfg.output_summary_json) as f:
        data = json.load(f)
    assert "kmeans" in data
    assert "nmi" in data["kmeans"]


def test_main_end_to_end_dbscan(tmp_path):
    """DBSCAN path runs without errors and produces outputs."""
    parquet_path = _make_parquet(tmp_path)
    n_tiles = 200
    cfg = _cfg(
        tmp_path,
        parquet_path,
        algorithms=["dbscan"],
        dbscan_eps=2.0,
        dbscan_min_samples=2,
        umap_subsample=0,
    )
    import mussel.cli.clustering_benchmark as cb

    with patch.dict(sys.modules, {"umap": _fake_umap_module(n_tiles)}):
        cb.main.__wrapped__(cfg)

    assert os.path.exists(cfg.output_summary_json)
    with open(cfg.output_summary_json) as f:
        data = json.load(f)
    assert "dbscan" in data


def test_main_end_to_end_multiclass(tmp_path):
    """Multiclass mode runs and produces outputs."""
    rng = np.random.default_rng(42)
    rows = []
    n_slides = 30
    for s in range(n_slides):
        annotation = (s % 3) + 1  # classes 1, 2, 3
        for t in range(10):
            row = {
                "slide_id": f"slide_{s:03d}",
                "annotation": annotation,
                "overlap_area": 0.8,
                "tile_area": 1.0,
                "geometry": box(t * 256, 0, (t + 1) * 256, 256),
            }
            for i, v in enumerate(rng.standard_normal(8)):
                row[f"feature_{i}"] = v
            rows.append(row)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry")
    parquet_path = str(tmp_path / "multi.parquet")
    gdf.to_parquet(parquet_path)

    n_tiles = 300
    cfg = _cfg(
        tmp_path,
        parquet_path,
        algorithms=["kmeans"],
        n_clusters=3,
        multiclass=True,
        umap_subsample=0,
    )
    import mussel.cli.clustering_benchmark as cb

    with patch.dict(sys.modules, {"umap": _fake_umap_module(n_tiles)}):
        cb.main.__wrapped__(cfg)

    assert os.path.exists(cfg.output_summary_json)
    with open(cfg.output_summary_json) as f:
        data = json.load(f)
    assert "kmeans" in data


def test_main_umap_subsample(tmp_path):
    """When umap_subsample > 0, UMAP receives a subsampled feature matrix."""
    parquet_path = _make_parquet(tmp_path)
    subsample = 50
    cfg = _cfg(tmp_path, parquet_path, algorithms=["kmeans"], umap_subsample=subsample)
    import mussel.cli.clustering_benchmark as cb

    fake = _fake_umap_module(subsample)
    with patch.dict(sys.modules, {"umap": fake}):
        cb.main.__wrapped__(cfg)

    # UMAP.fit_transform should have been called with `subsample` rows
    call_args = fake.UMAP.return_value.fit_transform.call_args
    assert call_args is not None
    X_passed = call_args[0][0]
    assert X_passed.shape[0] == subsample

