import logging
from dataclasses import dataclass, field
from typing import List, Optional

import geopandas as gpd
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class LinearProbeBenchmarkConfig:
    """
    output_csv (str): Path to save the validation classification report in CSV format.
    output_png (str): Path to save the validation confusion matrix as a PNG image.
    output_test_csv (str): Path to save the test classification report in CSV format.
    output_test_png (str): Path to save the test confusion matrix as a PNG image.
    features_annotation_parquet_path (str): Path to the parquet file containing features and annotations.
    annotation_percent_filter_threshold (float): Threshold for filtering annotations based on overlap area.
    test_size (float): Proportion of the dataset to include in the test split.
    val_size (float): Proportion of the training dataset to include in the validation split.
    random_state (int): Random seed for reproducibility.
    cv (int): Number of cross-validation folds for GridSearchCV.
    C_values (List[float]): C values to search over in the logistic regression grid search.
    penalties (List[str]): Regularization penalties to search over ('l1', 'l2', 'elasticnet', 'none').
    max_iter (int): Maximum number of iterations for the logistic regression solver.
    """

    output_csv: str = "classification_report.csv"
    output_png: str = "confusion_matrix.png"
    output_test_csv: str = "classification_report_test.csv"
    output_test_png: str = "confusion_matrix_test.png"
    features_annotation_parquet_path: str = MISSING
    annotation_percent_filter_threshold: float = 0.50
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    cv: int = 5
    C_values: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0])
    penalties: List[str] = field(default_factory=lambda: ["l2"])
    max_iter: int = 5000


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


def _eval_split(clf, X, y, output_csv, output_png, split_name):
    """Evaluate a fitted classifier on a data split, save report and confusion matrix."""
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]

    f1 = f1_score(y, y_pred, average="weighted")
    auc = roc_auc_score(y, y_prob)
    avg_prec = average_precision_score(y, y_prob)

    report = pd.DataFrame(classification_report(y, y_pred, output_dict=True))
    report.loc["auc_roc"] = np.nan
    report.loc["average_precision"] = np.nan
    report.at["auc_roc", "weighted avg"] = auc
    report.at["average_precision", "weighted avg"] = avg_prec
    report.to_csv(output_csv)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=ax)
    ax.set_title(f"{split_name} — F1={f1:.3f}  AUC={auc:.3f}  AP={avg_prec:.3f}")
    fig.savefig(output_png)
    plt.close(fig)

    logger.info(
        f"[{split_name}] F1={f1:.4f}  AUC-ROC={auc:.4f}  AvgPrec={avg_prec:.4f}"
    )
    return {"f1": f1, "auc_roc": auc, "average_precision": avg_prec}


@hydra.main(
    version_base=None, config_path=".", config_name="linear_probe_benchmark_config"
)
def main(cfg: LinearProbeBenchmarkConfig):
    """Benchmark a linear probe classifier on extracted features."""
    df = gpd.read_parquet(cfg.features_annotation_parquet_path)

    df_filtered = df.query(
        f"overlap_area > {cfg.annotation_percent_filter_threshold} * tile_area"
    )
    df_filtered = df_filtered.assign(y=(df_filtered.annotation == 2).astype(int))

    # One label per slide (1 if the slide has any positive tile, else 0) for stratification
    slide_labels = (
        df_filtered.groupby("slide_id")["y"].max().reset_index()
    )
    slide_ids = slide_labels["slide_id"].values
    strat = slide_labels["y"].values

    train_ids, test_ids, train_strat, _ = train_test_split(
        slide_ids, strat, test_size=cfg.test_size, random_state=cfg.random_state, stratify=strat
    )
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=cfg.val_size / (1 - cfg.test_size),
        random_state=cfg.random_state,
        stratify=train_strat,
    )

    train_df = df_filtered[df_filtered["slide_id"].isin(train_ids)]
    val_df = df_filtered[df_filtered["slide_id"].isin(val_ids)]
    test_df = df_filtered[df_filtered["slide_id"].isin(test_ids)]

    X_train = train_df.filter(regex="feature_").values
    y_train = train_df["y"].values
    X_val = val_df.filter(regex="feature_").values
    y_val = val_df["y"].values
    X_test = test_df.filter(regex="feature_").values
    y_test = test_df["y"].values

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="saga", max_iter=cfg.max_iter)),
    ])
    param_grid = {
        "clf__C": list(cfg.C_values),
        "clf__penalty": list(cfg.penalties),
    }
    search = GridSearchCV(
        pipeline, param_grid, cv=cfg.cv, scoring="roc_auc", n_jobs=-1
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    logger.info(f"Best params: {search.best_params_}  CV AUC={search.best_score_:.4f}")

    _eval_split(best, X_val, y_val, cfg.output_csv, cfg.output_png, "val")
    _eval_split(best, X_test, y_test, cfg.output_test_csv, cfg.output_test_png, "test")
