import json
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ClusteringBenchmarkConfig:
    """
    features_annotation_parquet_path (str): Path to the GeoParquet file produced by
        merge_annotation_features (must contain slide_id, annotation, overlap_area,
        tile_area, and feature_* columns).
    output_metrics_csv (str): Per-algorithm metrics table (CSV).
    output_summary_json (str): All scalar metrics as a nested dict (JSON).
    output_umap_png (str): UMAP scatter-plot grid (PNG); one figure with columns
        [cluster coloring, annotation coloring] for each algorithm row.
    annotation_percent_filter_threshold (float): Min overlap fraction to include a tile
        (default 0.50).
    positive_annotation_label (int): Annotation value treated as the positive class
        in binary mode (default 2).
    multiclass (bool): If True, use all non-zero annotation values as class labels.
        Background (annotation == 0) tiles are excluded.
    algorithms (List[str]): Clustering algorithms to run.  Supported: "kmeans",
        "hierarchical", "dbscan".  Default: ["kmeans", "hierarchical"].
    n_clusters (int): Number of clusters for kmeans and hierarchical (default 2).
    dbscan_eps (float): DBSCAN neighbourhood radius (default 0.5).
    dbscan_min_samples (int): DBSCAN minimum samples per core point (default 5).
    umap_n_neighbors (int): UMAP n_neighbors (default 15).
    umap_min_dist (float): UMAP min_dist (default 0.1).
    umap_n_components (int): UMAP output dimensionality — must be 2 for plotting
        (default 2).
    umap_subsample (int): Maximum number of tiles used for UMAP projection (random
        subsample for speed; default 10000). Use 0 to disable subsampling.
    random_state (int): Random seed (default 42).
    """

    features_annotation_parquet_path: str = MISSING
    output_metrics_csv: str = "clustering_metrics.csv"
    output_summary_json: str = "clustering_results.json"
    output_umap_png: str = "umap.png"
    annotation_percent_filter_threshold: float = 0.50
    positive_annotation_label: int = 2
    multiclass: bool = False
    algorithms: List[str] = field(default_factory=lambda: ["kmeans", "hierarchical"])
    n_clusters: int = 2
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 2
    umap_subsample: int = 10_000
    random_state: int = 42


desc_doc = """== ${hydra.help.app_name} ==
Evaluate feature quality by clustering tile-level embeddings and comparing the
resulting cluster assignments to annotation labels.  Produces supervised external
metrics (NMI, ARI, purity) at both tile and slide level, plus UMAP scatter plots.
"""

parameter_doc = f"""
== Available Parameters ==
{ClusteringBenchmarkConfig.__doc__}
"""

cs = ConfigStore.instance()
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=desc_doc, footer=parameter_doc)),
    provider="hydra",
)
cs.store(name="clustering_benchmark_config", node=ClusteringBenchmarkConfig)


# ── Metric helpers ─────────────────────────────────────────────────────────────


def _cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Fraction of tiles matching the majority annotation in their assigned cluster.

    Tiles labelled -1 by DBSCAN (noise) are excluded from the calculation.
    Returns NaN when no tiles remain after exclusion.
    """
    mask = labels_pred != -1
    if not mask.any():
        return float("nan")
    lt = labels_true[mask]
    lp = labels_pred[mask]
    total = len(lt)
    correct = 0
    for cluster_id in np.unique(lp):
        in_cluster = lt[lp == cluster_id]
        correct += int(np.bincount(in_cluster).max())
    return correct / total


def _compute_cluster_metrics(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    X_scaled: np.ndarray,
    split_name: str,
    algo_name: str,
    random_state: int = 42,
) -> dict:
    """Compute NMI, ARI, purity, and silhouette for one (algorithm, split) pair.

    DBSCAN noise points (label == -1) are excluded from all metrics.
    Silhouette is skipped when fewer than 2 distinct non-noise clusters are found.
    """
    noise_mask = labels_pred != -1
    lt_clean = labels_true[noise_mask]
    lp_clean = labels_pred[noise_mask]

    if len(lp_clean) == 0:
        logger.warning("[%s/%s] all tiles are noise — metrics set to NaN", algo_name, split_name)
        return {
            "nmi": float("nan"),
            "ari": float("nan"),
            "purity": float("nan"),
            "silhouette": float("nan"),
            "n_noise": int((~noise_mask).sum()),
            "n_clusters_found": 0,
        }

    n_clusters_found = len(np.unique(lp_clean))
    nmi = float(normalized_mutual_info_score(lt_clean, lp_clean, average_method="arithmetic"))
    ari = float(adjusted_rand_score(lt_clean, lp_clean))
    purity = _cluster_purity(labels_true, labels_pred)

    sil: Optional[float] = None
    if n_clusters_found >= 2 and len(lp_clean) >= n_clusters_found + 1:
        try:
            sil = float(silhouette_score(X_scaled[noise_mask], lp_clean, sample_size=min(5000, len(lp_clean)), random_state=random_state))
        except Exception as exc:
            logger.warning("[%s/%s] silhouette failed: %s", algo_name, split_name, exc)

    metrics = {
        "nmi": nmi,
        "ari": ari,
        "purity": purity,
        "silhouette": sil if sil is not None else float("nan"),
        "n_noise": int((~noise_mask).sum()),
        "n_clusters_found": n_clusters_found,
    }
    logger.info(
        "[%s/%s]  NMI=%.4f  ARI=%.4f  Purity=%.4f  Silhouette=%s  clusters=%d  noise=%d",
        algo_name,
        split_name,
        nmi,
        ari,
        purity,
        f"{sil:.4f}" if sil is not None else "n/a",
        n_clusters_found,
        metrics["n_noise"],
    )
    return metrics


def _slide_level_metrics(
    df: pd.DataFrame,
    labels_pred: np.ndarray,
    algo_name: str,
) -> dict:
    """Majority-vote cluster label per slide → NMI, ARI, purity vs slide annotation.

    Slides where all tiles are DBSCAN noise (label == -1) are excluded.
    """
    tmp = df[["slide_id", "y"]].copy()
    tmp["cluster"] = labels_pred

    def _majority_label(x):
        clean = x[x != -1].astype(int)
        if len(clean) == 0:
            return -1
        return int(np.argmax(np.bincount(clean)))

    slide_df = (
        tmp.groupby("slide_id")
        .agg(y=("y", _majority_label), cluster=("cluster", _majority_label))
        .reset_index()
    )
    slide_df = slide_df[slide_df["cluster"] != -1]

    if len(slide_df) == 0 or slide_df["y"].nunique() < 2:
        logger.warning("[%s/slide] not enough distinct classes — slide metrics skipped", algo_name)
        return {}

    lt_s = slide_df["y"].values
    lp_s = slide_df["cluster"].values
    metrics = {
        "slide_nmi": float(normalized_mutual_info_score(lt_s, lp_s, average_method="arithmetic")),
        "slide_ari": float(adjusted_rand_score(lt_s, lp_s)),
        "slide_purity": _cluster_purity(lt_s, lp_s),
    }
    logger.info(
        "[%s/slide]  n=%d  NMI=%.4f  ARI=%.4f  Purity=%.4f",
        algo_name,
        len(slide_df),
        metrics["slide_nmi"],
        metrics["slide_ari"],
        metrics["slide_purity"],
    )
    return metrics


# ── Clustering helpers ─────────────────────────────────────────────────────────


def _fit_predict(algo_name: str, X: np.ndarray, cfg: ClusteringBenchmarkConfig) -> np.ndarray:
    """Fit a clustering algorithm and return integer cluster labels."""
    if algo_name == "kmeans":
        model = KMeans(n_clusters=cfg.n_clusters, random_state=cfg.random_state, n_init=10)
    elif algo_name == "hierarchical":
        model = AgglomerativeClustering(n_clusters=cfg.n_clusters)
    elif algo_name == "dbscan":
        model = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples)
    else:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Supported: kmeans, hierarchical, dbscan.")
    return model.fit_predict(X)


# ── Plot helpers ───────────────────────────────────────────────────────────────


def _plot_umap(
    embedding: np.ndarray,
    subsample_idx: np.ndarray,
    labels_by_algo: dict,
    true_labels: np.ndarray,
    output_path: str,
    algo_names: List[str],
) -> None:
    """UMAP scatter-plot grid.

    Rows = algorithms, columns = [cluster coloring, annotation coloring].
    """
    n_algos = len(algo_names)
    fig, axes = plt.subplots(n_algos, 2, figsize=(10, 4 * n_algos), squeeze=False)

    for row, algo in enumerate(algo_names):
        cluster_labels = labels_by_algo[algo][subsample_idx]
        annot_labels = true_labels[subsample_idx]

        # Left: cluster coloring
        ax_cl = axes[row][0]
        scatter = ax_cl.scatter(
            embedding[:, 0], embedding[:, 1],
            c=cluster_labels, cmap="tab20", s=2, alpha=0.5, rasterized=True,
        )
        plt.colorbar(scatter, ax=ax_cl, label="Cluster")
        ax_cl.set_title(f"{algo} — clusters")
        ax_cl.set_xlabel("UMAP-1")
        ax_cl.set_ylabel("UMAP-2")

        # Right: annotation coloring
        ax_an = axes[row][1]
        scatter2 = ax_an.scatter(
            embedding[:, 0], embedding[:, 1],
            c=annot_labels, cmap="tab10", s=2, alpha=0.5, rasterized=True,
        )
        plt.colorbar(scatter2, ax=ax_an, label="Annotation")
        ax_an.set_title(f"{algo} — annotation")
        ax_an.set_xlabel("UMAP-1")
        ax_an.set_ylabel("UMAP-2")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved UMAP plot → %s", output_path)


def _sanitize_for_json(obj):
    """Recursively replace float NaN/Inf with None so json.dump produces valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra.main(
    version_base=None, config_path=".", config_name="clustering_benchmark_config"
)
def main(cfg: ClusteringBenchmarkConfig):
    """Benchmark clustering algorithms on extracted WSI tile features."""
    import pyarrow.parquet as pq

    _schema_names = pq.read_schema(cfg.features_annotation_parquet_path).names
    _cols = [c for c in _schema_names if c != "geometry"]
    df = pd.read_parquet(cfg.features_annotation_parquet_path, columns=_cols)

    df_filtered = df.query(
        f"overlap_area > {cfg.annotation_percent_filter_threshold} * tile_area"
    ).reset_index(drop=True)

    if cfg.multiclass:
        df_filtered = df_filtered[df_filtered["annotation"] != 0].copy()
        df_filtered["y"] = df_filtered["annotation"].astype(int)
        n_classes = df_filtered["y"].nunique()
        logger.info(
            "Multiclass mode: %d classes → %s", n_classes, sorted(df_filtered["y"].unique())
        )
    else:
        df_filtered["y"] = (df_filtered["annotation"] == cfg.positive_annotation_label).astype(int)

    feature_cols = [c for c in df_filtered.columns if c.startswith("feature_")]
    if df_filtered.empty:
        raise ValueError(
            "No tiles remain after filtering. Check annotation_percent_filter_threshold "
            "and that the parquet file is non-empty."
        )
    if not feature_cols:
        raise ValueError(
            "No feature columns (starting with 'feature_') found in the parquet file."
        )
    if cfg.umap_n_components != 2:
        raise ValueError(
            f"umap_n_components must be 2 for scatter plotting, got {cfg.umap_n_components}."
        )
    logger.info(
        "%d tiles  %d slides  %d features",
        len(df_filtered),
        df_filtered["slide_id"].nunique(),
        len(feature_cols),
    )

    df_filtered = df_filtered.reset_index(drop=True)
    X_raw = df_filtered[feature_cols].values.astype(np.float32)
    y_true = df_filtered["y"].values

    # Scale once; all algorithms work in the same normalised feature space.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    algo_names = list(cfg.algorithms)
    summary: dict = {}
    rows: list = []

    # Per-algorithm labels stored for UMAP coloring
    labels_by_algo: dict = {}

    for algo in algo_names:
        logger.info("── %s " + "─" * 50, algo)
        try:
            labels = _fit_predict(algo, X_scaled, cfg)
        except Exception as exc:
            logger.error("Algorithm '%s' failed: %s", algo, exc)
            continue

        labels_by_algo[algo] = labels

        tile_m = _compute_cluster_metrics(y_true, labels, X_scaled, "tile", algo, random_state=cfg.random_state)
        slide_m = _slide_level_metrics(df_filtered, labels, algo)

        summary[algo] = {**tile_m, **slide_m}
        row = {"algorithm": algo}
        row.update(tile_m)
        row.update(slide_m)
        rows.append(row)

    # ── Save tabular outputs ───────────────────────────────────────────────────
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(cfg.output_metrics_csv, index=False)
    logger.info("Saved metrics CSV → %s", cfg.output_metrics_csv)

    with open(cfg.output_summary_json, "w") as f:
        json.dump(_sanitize_for_json(summary), f, indent=2)
    logger.info("Saved summary JSON → %s", cfg.output_summary_json)

    # ── UMAP visualisation ─────────────────────────────────────────────────────
    if not labels_by_algo:
        logger.warning("No algorithms produced results — skipping UMAP plot.")
        return

    try:
        import umap as umap_module

        n = len(X_scaled)
        if cfg.umap_subsample > 0 and n > cfg.umap_subsample:
            rng = np.random.RandomState(cfg.random_state)
            subsample_idx = rng.choice(n, cfg.umap_subsample, replace=False)
        else:
            subsample_idx = np.arange(n)

        X_sub = X_scaled[subsample_idx]
        reducer = umap_module.UMAP(
            n_neighbors=cfg.umap_n_neighbors,
            min_dist=cfg.umap_min_dist,
            n_components=cfg.umap_n_components,
            random_state=cfg.random_state,
        )
        embedding = reducer.fit_transform(X_sub)

        _plot_umap(
            embedding,
            subsample_idx,
            labels_by_algo,
            y_true,
            cfg.output_umap_png,
            list(labels_by_algo.keys()),
        )
    except ImportError:
        logger.warning("umap-learn is not installed — skipping UMAP plot.")

    logger.info("Done.")
