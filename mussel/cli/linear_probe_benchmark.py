from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import MISSING, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


@dataclass
class LinearProbeBenchmarkConfig:
    features_annotation_parquet_path: str = MISSING
    annotation_percent_filter_threshold: float = 0.5
    annotation_mean_threshold: float = 0.5

cs = ConfigStore.instance()
cs.store(name="linear_probe_benchmark_config", node=LinearProbeBenchmarkConfig)

@hydra.main(version_base=None, config_path=".", config_name="linear_probe_benchmark_config")
def main(cfg: LinearProbeBenchmarkConfig):
    df = gpd.read_parquet(cfg.features_annotation_parquet_path)

    df_filtered = df.query(f"annotation_count > {cfg.annotation_percent_filter_threshold} * patch_size * patch_size")

    df_filtered['y'] = (df_filtered.annotation_mean > cfg.annotation_mean_threshold).astype(int)

    slide_ids = df_filtered['slide_id'].unique()

    train_ids, test_ids = train_test_split(slide_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.125, random_state=42)  # 0.25 * 0.8 = 0.2

    train_df = df_filtered[df_filtered['slide_id'].isin(train_ids)]
    val_df = df_filtered[df_filtered['slide_id'].isin(val_ids)]
    test_df = df_filtered[df_filtered['slide_id'].isin(test_ids)]

    X_train = train_df.filter(regex='feature_').values
    y_train = train_df['y'].values

    X_val = val_df.filter(regex='feature_').values
    y_val = val_df['y'].values

    X_test = test_df.filter(regex='feature_').values
    y_test = test_df['y'].values

    C_values = [0.01, 0.1, 1, 10, 100]

    best_f1 = 0
    best_model = None
    best_C = None

    for C in C_values:
        # Initialize logistic regression model
        clf = LogisticRegression(C=C, max_iter=5000, solver='lbfgs')
        # Train the model
        clf.fit(X_train, y_train)
        # Validation predictions
        y_val_pred = clf.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        print(f"C_value : {C} \t F1_score: {val_f1}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = clf
            best_C = C
