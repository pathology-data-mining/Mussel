"""Tests for mussel.cli.abmil_benchmark."""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from mussel.cli.abmil_benchmark import (
    AbmilBenchmarkConfig,
    AbmilClassifier,
    SlideDataset,
    _bootstrap_ci_auc,
    _collate_fn,
    _load_features,
    _split_by_slide,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_h5_features(path: Path, n_tiles: int = 50, feature_dim: int = 64, dtype: str = "float32", seed: int = 0):
    """Write a synthetic H5 feature file."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n_tiles, feature_dim)).astype(dtype)
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=arr)


def _make_features_dir(tmp_path: Path, n_slides: int = 10, feature_dim: int = 64, seed: int = 0) -> Path:
    """Create a directory of per-slide H5 feature files."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    rng = np.random.default_rng(seed)
    for i in range(n_slides):
        n_tiles = rng.integers(30, 80)
        _make_h5_features(features_dir / f"slide_{i:03d}.h5", int(n_tiles), feature_dim, seed=i)
    return features_dir


def _make_labels_parquet(tmp_path: Path, n_slides: int = 10, target_col: str = "label", seed: int = 0) -> Path:
    """Create a synthetic labels parquet."""
    rng = np.random.default_rng(seed)
    rows = [
        {"slide_id": f"slide_{i:03d}", target_col: int(rng.integers(0, 2))}
        for i in range(n_slides)
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "labels.parquet"
    df.to_parquet(path)
    return path


# ---------------------------------------------------------------------------
# Unit tests: data loading
# ---------------------------------------------------------------------------


def test_load_features_float32(tmp_path):
    h5_path = tmp_path / "slide.h5"
    _make_h5_features(h5_path, n_tiles=40, feature_dim=16, dtype="float32")
    features = _load_features(h5_path, torch.float32)
    assert features.shape == (40, 16)
    assert features.dtype == torch.float32


def test_load_features_cast_to_float16(tmp_path):
    h5_path = tmp_path / "slide.h5"
    _make_h5_features(h5_path, n_tiles=40, feature_dim=16, dtype="float32")
    features = _load_features(h5_path, torch.float16)
    assert features.dtype == torch.float16
    # Values should differ from float32 due to reduced precision.
    features_f32 = _load_features(h5_path, torch.float32)
    assert not torch.equal(features.float(), features_f32)


def test_slide_dataset(tmp_path):
    features_dir = _make_features_dir(tmp_path, n_slides=5, feature_dim=32)
    slide_ids = [f"slide_{i:03d}" for i in range(5)]
    labels = np.array([0, 1, 0, 1, 0])
    ds = SlideDataset(slide_ids, labels, features_dir, cast_dtype=torch.float32)
    assert len(ds) == 5
    features, label = ds[2]
    assert features.ndim == 2
    assert features.shape[1] == 32
    assert label.item() == 0


def test_collate_fn_pads_correctly():
    """Batch of slides with different tile counts should be padded correctly."""
    feat_dim = 8
    b1 = torch.ones(10, feat_dim)
    b2 = torch.ones(20, feat_dim) * 2
    b3 = torch.ones(15, feat_dim) * 3
    labels = [torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0)]
    batch = [(b1, labels[0]), (b2, labels[1]), (b3, labels[2])]
    padded, lbls, mask = _collate_fn(batch)
    assert padded.shape == (3, 20, feat_dim)
    assert mask.shape == (3, 20)
    assert padded.dtype == torch.float32
    assert mask[0, :10].all() and not mask[0, 10:].any()
    assert mask[1, :20].all()
    assert mask[2, :15].all() and not mask[2, 15:].any()
    # Padding entries must be zero.
    assert (padded[0, 10:] == 0).all()


# ---------------------------------------------------------------------------
# Unit tests: model
# ---------------------------------------------------------------------------


def test_abmil_classifier_forward():
    model = AbmilClassifier(feature_dim=32, head_dim=16, n_heads=2)
    model.eval()
    B, N, D = 4, 25, 32
    x = torch.randn(B, N, D)
    mask = torch.ones(B, N, dtype=torch.bool)
    with torch.no_grad():
        logits = model(x, mask)
    assert logits.shape == (B,)


def test_abmil_classifier_with_padding():
    model = AbmilClassifier(feature_dim=16, head_dim=8, n_heads=1)
    model.eval()
    B, N_max, D = 2, 30, 16
    x = torch.randn(B, N_max, D)
    # Slide 0 has 20 valid tiles, slide 1 has 30.
    mask = torch.zeros(B, N_max, dtype=torch.bool)
    mask[0, :20] = True
    mask[1, :30] = True
    with torch.no_grad():
        logits = model(x, mask)
    assert logits.shape == (B,)


# ---------------------------------------------------------------------------
# Unit tests: splits
# ---------------------------------------------------------------------------


def test_split_by_slide_no_leakage():
    slide_ids = [f"slide_{i:03d}" for i in range(30)]
    labels = [i % 2 for i in range(30)]
    df = pd.DataFrame({"slide_id": slide_ids, "y": labels})
    train_df, val_df, test_df = _split_by_slide(df, test_size=0.2, val_size=0.1, seed=42)
    all_ids = set(train_df["slide_id"]) | set(val_df["slide_id"]) | set(test_df["slide_id"])
    assert all_ids == set(slide_ids)
    # No overlap between splits.
    assert not (set(train_df["slide_id"]) & set(val_df["slide_id"]))
    assert not (set(train_df["slide_id"]) & set(test_df["slide_id"]))
    assert not (set(val_df["slide_id"]) & set(test_df["slide_id"]))


# ---------------------------------------------------------------------------
# Unit tests: bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_auc():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 100)
    labels = (probs > 0.5).astype(int)
    lo, hi = _bootstrap_ci_auc(probs, labels, n_bootstrap=200, seed=0)
    assert lo < hi
    assert 0.5 <= lo <= 1.0
    assert 0.5 <= hi <= 1.0


# ---------------------------------------------------------------------------
# Integration test: end-to-end (CPU, tiny dataset)
# ---------------------------------------------------------------------------


def test_abmil_benchmark_end_to_end(tmp_path, monkeypatch):
    """Smoke test: run one seed of training on a tiny synthetic dataset."""
    n_slides = 12
    feature_dim = 16
    features_dir = _make_features_dir(tmp_path, n_slides=n_slides, feature_dim=feature_dim)
    labels_path = _make_labels_parquet(tmp_path, n_slides=n_slides)
    output_path = str(tmp_path / "results.json")

    from mussel.cli.abmil_benchmark import AbmilBenchmarkConfig, _run_one_seed

    cfg = AbmilBenchmarkConfig(
        features_dir=str(features_dir),
        labels_parquet=str(labels_path),
        target_col="label",
        output_summary_json=output_path,
        n_seeds=1,
        random_state=0,
        n_epochs=2,
        batch_size=4,
        dtype="float32",
        head_dim=8,
        n_heads=1,
    )

    df = pd.read_parquet(labels_path)
    df = df.rename(columns={"label": "y"})
    df["y"] = df["y"].astype(int)

    metrics, test_probs, test_labels = _run_one_seed(
        cfg, df, features_dir, torch.device("cpu"), seed=0
    )
    assert "auroc" in metrics["val"]
    assert "auroc" in metrics["test"]
    # With random labels, AUROC might be anywhere; just check it's a valid float.
    val_auc = metrics["val"]["auroc"]
    test_auc = metrics["test"]["auroc"]
    assert isinstance(val_auc, float)
    assert isinstance(test_auc, float)
    assert test_probs.shape == test_labels.shape
