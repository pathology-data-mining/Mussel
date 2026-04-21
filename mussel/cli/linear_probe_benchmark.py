import json
import logging
import math
from dataclasses import dataclass, field
from typing import List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class LinearProbeBenchmarkConfig:
    """
    output_csv (str): Validation classification report (CSV).
    output_png (str): Validation confusion matrix (PNG).
    output_test_csv (str): Test classification report (CSV).
    output_test_png (str): Test confusion matrix (PNG).
    output_roc_png (str): ROC curves for val and test splits (PNG).
    output_pr_png (str): Precision-recall curves for val and test splits (PNG).
    output_gs_heatmap_png (str): Grid search CV AUC heatmap (PNG).
    output_feature_importance_png (str): Top features by |coefficient| (PNG).
    output_calibration_png (str): Calibration curves for val and test splits (PNG).
    output_cv_results_csv (str): Full GridSearchCV cv_results_ table (CSV).
    output_summary_json (str): All scalar metrics, mean +/- std across seeds (JSON).
    positive_annotation_label (int): Annotation value treated as the positive class (default 2
        for raw BMP annotations; set to 1 when class_mapping has already remapped to 0/1).
    annotation_percent_filter_threshold (float): Min overlap fraction to include a tile.
    test_size (float): Fraction of slides held out as test set.
    val_size (float): Fraction of all slides held out as validation set.
    random_state (int): Primary seed; seeds random_state ... random_state+n_seeds-1 are used.
    cv (int): Cross-validation folds for GridSearchCV.
    C_values (List[float]): Regularisation strengths to search.
    penalties (List[str]): Penalty types to search ('l1', 'l2', 'elasticnet', 'none').
    max_iter (int): Maximum solver iterations.
    n_top_features (int): Number of features shown in the importance plot.
    n_seeds (int): Number of random seeds; mean +/- std reported across seeds.
    n_bootstrap (int): Bootstrap resamples for 95% CI on test AUC-ROC (primary seed).
    multiclass (bool): If True, treat annotation values directly as class labels (multiclass
        logistic regression). Background (annotation == 0) tiles are excluded.
        Use with a class_mapping that emits ≥ 3 distinct non-zero labels.
    """

    output_csv: str = "classification_report.csv"
    output_png: str = "confusion_matrix.png"
    output_test_csv: str = "classification_report_test.csv"
    output_test_png: str = "confusion_matrix_test.png"
    output_roc_png: str = "roc_curve.png"
    output_pr_png: str = "pr_curve.png"
    output_gs_heatmap_png: str = "grid_search_heatmap.png"
    output_feature_importance_png: str = "feature_importance.png"
    output_calibration_png: str = "calibration_curve.png"
    output_cv_results_csv: str = "cv_results.csv"
    output_summary_json: str = "results.json"
    features_annotation_parquet_path: str = MISSING
    annotation_percent_filter_threshold: float = 0.50
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    cv: int = 5
    C_values: List[float] = field(default_factory=lambda: [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0])
    penalties: List[str] = field(default_factory=lambda: ["l2"])
    max_iter: int = 5000
    n_top_features: int = 20
    n_seeds: int = 5
    n_bootstrap: int = 1000
    positive_annotation_label: int = 2
    multiclass: bool = False


desc_doc = """== ${hydra.help.app_name} ==
This script benchmarks a linear probe classifier on features extracted from whole slide images.
"""

parameter_doc = f"""
== Available Parameters ==
{LinearProbeBenchmarkConfig.__doc__}
"""

cs = ConfigStore.instance()
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=desc_doc, footer=parameter_doc)),
    provider="hydra",
)
cs.store(name="linear_probe_benchmark_config", node=LinearProbeBenchmarkConfig)


# ── Data helpers ──────────────────────────────────────────────────────────────


def _split_by_slide(df, test_size, val_size, random_state):
    """Random train / val / test split at slide level.

    Splits are done at the slide level so all tiles from a given slide land
    in the same partition, preventing data leakage from slide-specific
    artefacts (staining, scanner, tissue prep).  Tile-level labels are not
    used for stratification — the split is purely random over slides.
    """
    # Sort before permuting so the split is reproducible regardless of the order
    # slides appear in the DataFrame (which depends on parquet write/read order).
    slide_ids = np.sort(df["slide_id"].unique())
    rng = np.random.RandomState(random_state)
    slide_ids = rng.permutation(slide_ids)

    n = len(slide_ids)
    n_test = max(1, int(round(n * test_size)))
    n_val  = max(1, int(round(n * val_size)))
    n_train = n - n_test - n_val

    logger.info(
        "Slide split: total=%d  train=%d  val=%d  test=%d", n, n_train, n_val, n_test
    )

    test_ids  = set(slide_ids[:n_test])
    val_ids   = set(slide_ids[n_test : n_test + n_val])
    train_ids = set(slide_ids[n_test + n_val :])

    return (
        df[df["slide_id"].isin(train_ids)],
        df[df["slide_id"].isin(val_ids)],
        df[df["slide_id"].isin(test_ids)],
    )


def _log_split_stats(train_df, val_df, test_df, is_multiclass=False):
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if is_multiclass:
            counts = d["y"].value_counts().sort_index().to_dict()
            logger.info(
                f"  {name:5s}: {len(d):6d} tiles  {d['slide_id'].nunique():3d} slides  class_counts={counts}"
            )
        else:
            logger.info(
                f"  {name:5s}: {len(d):6d} tiles  {d['slide_id'].nunique():3d} slides  "
                f"pos_rate={d['y'].mean():.3f}"
            )


# ── Metrics helpers ───────────────────────────────────────────────────────────


def _compute_metrics(df, y_prob, split_name, classes=None):
    """Compute tile-level and slide-level metrics from pre-computed probabilities.

    y_prob: 1-D array (binary) or 2-D array shape (n, n_classes) (multiclass).
    classes: ndarray of class labels corresponding to y_prob columns (multiclass only).
    """
    y = df["y"].values
    is_multiclass = y_prob.ndim == 2

    if is_multiclass:
        y_pred = classes[np.argmax(y_prob, axis=1)]

        # Compute OvR AUC manually (one binary AUC per class vs rest, then macro-average).
        # This avoids sklearn's "probabilities must sum to 1" requirement when a rare class
        # is absent from a split and its column is dropped.
        unique_y = np.unique(y)
        aucs = []
        for i, cls in enumerate(classes):
            if cls not in unique_y:
                logger.warning(
                    "[%s] class %s absent from split — excluded from OvR AUC", split_name, cls
                )
                continue
            aucs.append(roc_auc_score((y == cls).astype(int), y_prob[:, i]))
        tile_auc = float(np.mean(aucs)) if aucs else float("nan")

        metrics = {
            "tile_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
            "tile_auc_roc": tile_auc,
        }
        logger.info(
            f"[{split_name} tiles]  F1={metrics['tile_f1']:.4f}  "
            f"AUC(OvR)={metrics['tile_auc_roc']:.4f}"
        )
        # Slide-level aggregation is ambiguous for multiclass — skip.
        return metrics

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "tile_f1": float(f1_score(y, y_pred, average="weighted")),
        "tile_auc_roc": float(roc_auc_score(y, y_prob)),
        "tile_average_precision": float(average_precision_score(y, y_prob)),
    }
    logger.info(
        f"[{split_name} tiles]  F1={metrics['tile_f1']:.4f}  "
        f"AUC={metrics['tile_auc_roc']:.4f}  AP={metrics['tile_average_precision']:.4f}"
    )

    # Slide-level: aggregate tile probabilities -> mean per slide
    slide_df = (
        df.assign(prob=y_prob)
        .groupby("slide_id")
        .agg(y=("y", "max"), mean_prob=("prob", "mean"))
        .reset_index()
    )
    if slide_df["y"].nunique() >= 2:
        y_s = slide_df["y"].values
        p_s = slide_df["mean_prob"].values
        metrics.update(
            {
                "slide_f1": float(
                    f1_score(y_s, (p_s >= 0.5).astype(int), average="weighted")
                ),
                "slide_auc_roc": float(roc_auc_score(y_s, p_s)),
                "slide_average_precision": float(average_precision_score(y_s, p_s)),
            }
        )
        logger.info(
            f"[{split_name} slides] n={len(slide_df)}  "
            f"F1={metrics['slide_f1']:.4f}  AUC={metrics['slide_auc_roc']:.4f}  "
            f"AP={metrics['slide_average_precision']:.4f}"
        )
    else:
        logger.warning(
            f"[{split_name} slide-level] only one class present — slide metrics skipped"
        )

    return metrics


def _save_confusion_matrix(y, y_pred, y_prob, output_csv, output_png, split_name, metrics):
    is_multiclass = np.asarray(y_prob).ndim == 2

    report = pd.DataFrame(classification_report(y, y_pred, output_dict=True))
    report.loc["auc_roc"] = np.nan
    report.at["auc_roc", "weighted avg"] = metrics["tile_auc_roc"]
    if not is_multiclass:
        report.loc["average_precision"] = np.nan
        report.at["average_precision", "weighted avg"] = metrics["tile_average_precision"]
    report.to_csv(output_csv)

    title_suffix = f"F1={metrics['tile_f1']:.3f}  AUC={'(OvR) ' if is_multiclass else ''}{metrics['tile_auc_roc']:.3f}"
    if not is_multiclass:
        title_suffix += f"  AP={metrics['tile_average_precision']:.3f}"

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=ax)
    ax.set_title(f"{split_name} — {title_suffix}")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bootstrap_ci_auc(y_prob, y, n_bootstrap, random_state, classes=None):
    """95% bootstrap CI for AUC-ROC; resamples pre-computed probabilities.

    For multiclass, y_prob is 2-D and OvR macro AUC is used.
    """
    is_multiclass = np.asarray(y_prob).ndim == 2
    n_classes = len(classes) if classes is not None else 2
    rng = np.random.RandomState(random_state)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < (n_classes if is_multiclass else 2):
            continue
        if is_multiclass:
            unique_idx = np.unique(y[idx])
            aucs_b = []
            for i, cls in enumerate(classes):
                if cls not in unique_idx:
                    continue
                aucs_b.append(roc_auc_score((y[idx] == cls).astype(int), y_prob[idx][:, i]))
            if len(aucs_b) >= 2:
                scores.append(float(np.mean(aucs_b)))
        else:
            scores.append(roc_auc_score(y[idx], y_prob[idx]))
    if not scores:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# -- Plot helpers -----------------------------------------------------------



def _plot_roc_curves(val_y, val_prob, test_y, test_prob, output_path, classes=None):
    """ROC curves for val and test.  OvR per-class curves for multiclass."""
    is_multiclass = np.asarray(val_prob).ndim == 2
    if is_multiclass:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (split_name, y, prob) in zip(
            axes, [("val", val_y, val_prob), ("test", test_y, test_prob)]
        ):
            for i, cls in enumerate(classes):
                RocCurveDisplay.from_predictions(
                    (y == cls).astype(int), prob[:, i], ax=ax, name=f"class {cls}"
                )
            ax.set_title(f"ROC Curves — {split_name} (OvR)")
        fig.suptitle("One-vs-Rest ROC Curves")
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, y, prob in [("val", val_y, val_prob), ("test", test_y, test_prob)]:
            RocCurveDisplay.from_predictions(y, prob, ax=ax, name=name)
        ax.set_title("ROC Curve")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pr_curves(val_y, val_prob, test_y, test_prob, output_path, classes=None):
    """Precision-recall curves for val and test.  OvR per-class for multiclass."""
    is_multiclass = np.asarray(val_prob).ndim == 2
    if is_multiclass:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (split_name, y, prob) in zip(
            axes, [("val", val_y, val_prob), ("test", test_y, test_prob)]
        ):
            for i, cls in enumerate(classes):
                PrecisionRecallDisplay.from_predictions(
                    (y == cls).astype(int), prob[:, i], ax=ax, name=f"class {cls}"
                )
            ax.set_title(f"PR Curves — {split_name} (OvR)")
        fig.suptitle("One-vs-Rest Precision-Recall Curves")
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, y, prob in [("val", val_y, val_prob), ("test", test_y, test_prob)]:
            PrecisionRecallDisplay.from_predictions(y, prob, ax=ax, name=name)
        ax.set_title("Precision-Recall Curve")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_gs_heatmap(search, output_path):
    """Heatmap of mean CV AUC by C x penalty; falls back to a line plot for a single penalty."""
    results = pd.DataFrame(search.cv_results_)
    results["param_clf__C"] = results["param_clf__C"].astype(float)
    results["param_clf__penalty"] = results["param_clf__penalty"].astype(str)
    penalties = results["param_clf__penalty"].unique()

    if len(penalties) > 1:
        pivot = results.pivot_table(
            index="param_clf__penalty",
            columns="param_clf__C",
            values="mean_test_score",
        )
        fig, ax = plt.subplots(
            figsize=(max(6, len(pivot.columns) * 1.2), max(3, len(pivot) * 1.2))
        )
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.3g}" for c in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("C")
        ax.set_ylabel("Penalty")
        ax.set_title("Grid Search — Mean CV AUC-ROC")
        plt.colorbar(im, ax=ax, label="Mean CV AUC-ROC")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(
                    j,
                    i,
                    f"{pivot.values[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
    else:
        c_vals = sorted(results["param_clf__C"].unique())
        scores = [
            results.loc[results["param_clf__C"] == c, "mean_test_score"].values[0]
            for c in c_vals
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogx(c_vals, scores, marker="o")
        ax.axvline(
            search.best_params_["clf__C"],
            color="r",
            linestyle="--",
            label=f"best C={search.best_params_['clf__C']:.3g}",
        )
        ax.set_xlabel("C")
        ax.set_ylabel("Mean CV AUC-ROC")
        ax.set_title(f"Grid Search — Penalty={penalties[0]}")
        ax.legend()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(pipeline, feature_names, n_top, output_path):
    """Horizontal bar chart of top features by absolute logistic-regression coefficient.

    For multiclass (coef_ shape (n_classes, n_features)), the L2 norm across classes
    is used so a single importance score summarises all classes.
    """
    clf = pipeline.named_steps["clf"]
    if clf.coef_.shape[0] == 1:
        coef = clf.coef_[0]
        title = f"Top {n_top} Feature Importances by |Coefficient|"
    else:
        coef = np.linalg.norm(clf.coef_, axis=0)
        title = f"Top {n_top} Feature Importances by ‖Coefficient‖₂ (multiclass)"

    top_idx = np.argsort(coef)[::-1][:n_top]
    top_names = [feature_names[i] for i in top_idx][::-1]
    top_coef = coef[top_idx][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, n_top * 0.35)))
    colors = ["tab:red" if c > 0 else "tab:blue" for c in top_coef]
    ax.barh(range(len(top_names)), top_coef, color=colors)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (scaled feature space)")
    ax.set_title(title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration(val_y, val_prob, test_y, test_prob, output_path):
    """Calibration curves for val and test on the same axes (binary only)."""
    if np.asarray(val_prob).ndim == 2:
        logger.info("Calibration curve skipped for multiclass mode — writing placeholder.")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "Calibration curve\nnot available for multiclass",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y, prob in [("val", val_y, val_prob), ("test", test_y, test_prob)]:
        CalibrationDisplay.from_predictions(y, prob, ax=ax, name=name, n_bins=10)
    ax.set_title("Calibration Curve")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    version_base=None, config_path=".", config_name="linear_probe_benchmark_config"
)
def main(cfg: LinearProbeBenchmarkConfig):
    """Benchmark a linear probe classifier on extracted WSI features."""
    import pyarrow.parquet as pq
    _schema_names = pq.read_schema(cfg.features_annotation_parquet_path).names
    _cols = [c for c in _schema_names if c != "geometry"]
    df = pd.read_parquet(cfg.features_annotation_parquet_path, columns=_cols)
    df_filtered = df.query(
        f"overlap_area > {cfg.annotation_percent_filter_threshold} * tile_area"
    )

    if cfg.multiclass:
        # Exclude background (annotation == 0); use raw annotation value as y.
        df_filtered = df_filtered[df_filtered["annotation"] != 0]
        df_filtered = df_filtered.assign(y=df_filtered["annotation"].astype(int))
        n_classes = df_filtered["y"].nunique()
        logger.info(
            "Multiclass mode: %d classes → %s", n_classes, sorted(df_filtered["y"].unique())
        )
    else:
        df_filtered = df_filtered.assign(
            y=(df_filtered.annotation == cfg.positive_annotation_label).astype(int)
        )

    feature_cols = [c for c in df_filtered.columns if c.startswith("feature_")]

    # Pre-cast feature matrix to float32 once — halves memory bandwidth for all
    # subsequent matrix operations (StandardScaler, LogisticRegression).
    # Reset index first so df_filtered.index matches positional indices in the array.
    df_filtered = df_filtered.reset_index(drop=True)
    feature_arr_f32 = df_filtered[feature_cols].values.astype(np.float32)
    df_filtered = df_filtered.drop(columns=feature_cols)  # free float64 memory
    df_filtered = df_filtered.copy()  # defragment after drop

    # Auto-select solver: lbfgs converges in ~100 L-BFGS steps for L2; saga is
    # needed only for L1.  lbfgs is 3-10× faster on dense data.
    if "elasticnet" in list(cfg.penalties):
        raise ValueError(
            "penalty='elasticnet' requires l1_ratio, which this benchmark does not "
            "configure in the GridSearchCV param grid. Remove 'elasticnet' from "
            "cfg.penalties or add l1_ratio support."
        )
    _needs_saga = any(p in ("l1",) for p in cfg.penalties)
    _solver = "saga" if _needs_saga else "lbfgs"
    _max_iter = cfg.max_iter if _needs_saga else min(cfg.max_iter, 1000)
    logger.info("solver=%s  max_iter=%d", _solver, _max_iter)

    # Multiclass: OvR AUC; binary: standard AUC.
    _scoring = "roc_auc_ovr" if cfg.multiclass else "roc_auc"

    seeds = [cfg.random_state + i for i in range(cfg.n_seeds)]
    all_val_metrics: list = []
    all_test_metrics: list = []
    primary_search = primary_best = primary_val_df = primary_test_df = None
    best_params = None  # set after primary seed; reused for remaining seeds

    for seed in seeds:
        logger.info(f"── seed={seed} " + "─" * 40)
        train_df, val_df, test_df = _split_by_slide(
            df_filtered, cfg.test_size, cfg.val_size, seed
        )
        _log_split_stats(train_df, val_df, test_df, is_multiclass=cfg.multiclass)

        train_idx = train_df.index
        val_idx   = val_df.index
        test_idx  = test_df.index

        X_train_raw = feature_arr_f32[train_idx]
        X_val_raw   = feature_arr_f32[val_idx]
        X_test_raw  = feature_arr_f32[test_idx]
        y_train = train_df["y"].values

        if best_params is None:
            # Primary seed: full grid search with slide-level group constraints to
            # prevent tiles from the same slide appearing in both train and validation
            # folds of the inner CV loop.
            search = GridSearchCV(
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(solver=_solver, max_iter=_max_iter)),
                ]),
                {"clf__C": list(cfg.C_values), "clf__penalty": list(cfg.penalties)},
                cv=StratifiedGroupKFold(n_splits=cfg.cv),
                scoring=_scoring,
                n_jobs=-1,
                return_train_score=True,
            )
            search.fit(X_train_raw, y_train, groups=train_df["slide_id"].values)
            best_params = search.best_params_
            best = search.best_estimator_
            logger.info(f"  best_params={best_params}  CV AUC={search.best_score_:.4f}")
        else:
            # Subsequent seeds: reuse best_params, single refit (no grid search).
            best = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(solver=_solver, max_iter=_max_iter)),
            ])
            best.set_params(**best_params)
            best.fit(X_train_raw, y_train)
            logger.info(f"  reusing best_params={best_params} (no grid search)")
            search = None

        # Compute predictions once per split; all helpers reuse these arrays.
        # Multiclass: full probability matrix (n × n_classes); binary: 1-D vector.
        y_val = val_df["y"].values
        y_test = test_df["y"].values
        _classes = best.named_steps["clf"].classes_ if cfg.multiclass else None
        if cfg.multiclass:
            prob_val  = best.predict_proba(X_val_raw)
            prob_test = best.predict_proba(X_test_raw)
        else:
            prob_val  = best.predict_proba(X_val_raw)[:, 1]
            prob_test = best.predict_proba(X_test_raw)[:, 1]

        val_m  = _compute_metrics(val_df,  prob_val,  "val",  classes=_classes)
        test_m = _compute_metrics(test_df, prob_test, "test", classes=_classes)
        all_val_metrics.append(val_m)
        all_test_metrics.append(test_m)

        if seed == cfg.random_state:
            primary_search, primary_best = search, best
            primary_val_df, primary_test_df = val_df, test_df
            primary_val_y, primary_val_prob = y_val, prob_val
            primary_test_y, primary_test_prob = y_test, prob_test
            if cfg.multiclass:
                y_pred_val  = _classes[np.argmax(prob_val,  axis=1)]
                y_pred_test = _classes[np.argmax(prob_test, axis=1)]
            else:
                y_pred_val  = (prob_val  >= 0.5).astype(int)
                y_pred_test = (prob_test >= 0.5).astype(int)
            _save_confusion_matrix(y_val,  y_pred_val,  prob_val,  cfg.output_csv,      cfg.output_png,      "val",  val_m)
            _save_confusion_matrix(y_test, y_pred_test, prob_test, cfg.output_test_csv, cfg.output_test_png, "test", test_m)

    # ── Multi-seed summary ────────────────────────────────────────────────────
    all_keys = sorted({k for m in all_val_metrics + all_test_metrics for k in m})
    summary: dict = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "best_params": best_params,
        "best_cv_auc": float(primary_search.best_score_),
    }
    for split_name, records in [("val", all_val_metrics), ("test", all_test_metrics)]:
        summary[split_name] = {}
        for key in all_keys:
            vals = [r[key] for r in records if key in r]
            if not vals:
                continue
            mean, std = float(np.mean(vals)), float(np.std(vals))
            summary[split_name][key] = {"mean": mean, "std": std}
            logger.info(f"[{split_name}] {key}: {mean:.4f} +/- {std:.4f}")

    # ── Bootstrap CI on test AUC (primary seed) ───────────────────────────────
    _primary_classes = primary_best.named_steps["clf"].classes_ if cfg.multiclass else None
    ci_lo, ci_hi = _bootstrap_ci_auc(
        primary_test_prob, primary_test_y, cfg.n_bootstrap, cfg.random_state,
        classes=_primary_classes,
    )
    summary["test"]["tile_auc_roc"]["bootstrap_ci_95"] = [ci_lo, ci_hi]
    logger.info(f"[test] tile_auc_roc bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ── Save tabular outputs ──────────────────────────────────────────────────
    pd.DataFrame(primary_search.cv_results_).to_csv(cfg.output_cv_results_csv, index=False)
    with open(cfg.output_summary_json, "w") as f:
        json.dump(_sanitize_for_json(summary), f, indent=2)

    # ── Plots (primary seed) ──────────────────────────────────────────────────
    _plot_roc_curves(primary_val_y, primary_val_prob, primary_test_y, primary_test_prob, cfg.output_roc_png, classes=_primary_classes)
    _plot_pr_curves(primary_val_y, primary_val_prob, primary_test_y, primary_test_prob, cfg.output_pr_png, classes=_primary_classes)
    _plot_gs_heatmap(primary_search, cfg.output_gs_heatmap_png)
    _plot_feature_importance(
        primary_best, feature_cols, cfg.n_top_features, cfg.output_feature_importance_png
    )
    _plot_calibration(primary_val_y, primary_val_prob, primary_test_y, primary_test_prob, cfg.output_calibration_png)

    logger.info("Done.")
