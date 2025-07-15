from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import MISSING, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

import hydra


@dataclass
class LinearProbeBenchmarkConfig:
    """
    output_csv (str): Path to save the classification report in CSV format.
    output_png (str): Path to save the confusion matrix as a PNG image.
    features_annotation_parquet_path (str): Path to the parquet file containing features and annotations.
    annotation_percent_filter_threshold (float): Threshold for filtering annotations based on overlap area.
    test_size (float): Proportion of the dataset to include in the test split.
    val_size (float): Proportion of the training dataset to include in the validation split.
    random_state (int): Random seed for reproducibility.
    penalty (Optional[str]): Regularization penalty to use in the logistic regression ('l1', 'l2', 'elasticnet' or None).
    C (float): Inverse of regularization strength; smaller values specify stronger regularization.
    max_iter (int): Maximum number of iterations for the logistic regression solver.
    """

    output_csv: str = "classification_report.csv"
    output_png: str = "confusion_matrix.png"
    features_annotation_parquet_path: str = MISSING
    annotation_percent_filter_threshold: float = 0.50
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    penalty: Optional[str] = None  # 'l1', 'l2', 'elasticnet' or None
    C: float = 1.0
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


@hydra.main(
    version_base=None, config_path=".", config_name="linear_probe_benchmark_config"
)
def main(cfg: LinearProbeBenchmarkConfig):
    df = gpd.read_parquet(cfg.features_annotation_parquet_path)

    df_filtered = df.query(
        f"overlap_area > {cfg.annotation_percent_filter_threshold} * tile_area"
    )

    df_filtered = df_filtered.assign(y=(df_filtered.annotation == 2).astype(int))

    slide_ids = df_filtered.query("annotation == 2")["slide_id"].unique()

    # 20% test size, 10% validation, 70% train size
    train_ids, test_ids = train_test_split(
        slide_ids, test_size=cfg.test_size, random_state=cfg.random_state
    )
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=cfg.val_size / (1 - cfg.test_size),
        random_state=cfg.random_state,
    )  # 0.25 * 0.8 = 0.2

    train_df = df_filtered[df_filtered["slide_id"].isin(train_ids)]
    val_df = df_filtered[df_filtered["slide_id"].isin(val_ids)]
    test_df = df_filtered[df_filtered["slide_id"].isin(test_ids)]

    X_train = train_df.filter(regex="feature_").values
    y_train = train_df["y"].values

    X_val = val_df.filter(regex="feature_").values
    y_val = val_df["y"].values

    X_test = test_df.filter(regex="feature_").values
    y_test = test_df["y"].values

    clf = LogisticRegression(
        penalty=cfg.penalty, C=cfg.C, max_iter=cfg.max_iter, solver="lbfgs"
    )
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average="weighted")

    cls_report_df = pd.DataFrame(
        classification_report(y_val, y_val_pred, output_dict=True)
    )
    cls_report_df.to_csv(cfg.output_csv)

    ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred)
    plt.savefig(cfg.output_png)

    logger.info(f"C_value : {cfg.C} \t F1_score: {val_f1}")
