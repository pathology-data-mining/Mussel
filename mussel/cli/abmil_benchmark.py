"""ABMIL benchmark: evaluate WSI classification AUROC from patch features.

Measures the effect of feature storage precision (float32/float16/bfloat16) on
slide-level classification performance, using Attention-Based MIL (ABMIL) as the
aggregation model.

Usage::

    abmil_benchmark \\
        features_dir=/path/to/feature/h5s \\
        labels_parquet=/path/to/labels.parquet \\
        target_col=STK11_ONCOGENIC \\
        dtype=float16 \\
        output_summary_json=results_float16.json
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from mussel.models.abmil import ABMIL
from mussel.utils.feature_extract import _numpy_to_torch

logger = logging.getLogger(__name__)

_VALID_DTYPES = ("float32", "float16", "bfloat16")
_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class AbmilBenchmarkConfig:
    """
    features_dir (str): Directory of per-slide HDF5 feature files ({slide_id}.h5).
        Each file must contain a dataset named 'features' with shape (n_tiles, feature_dim).
    labels_parquet (str): Parquet file with columns: slide_id, <target_col>, and optionally
        a split column. Rows without a matching H5 file are silently dropped.
    target_col (str): Column name of the binary target label (integer 0/1).
    split_col (Optional[str]): Column containing 'train'/'val'/'test' split labels.
        If None, a random split is computed using test_size and val_size.
    output_summary_json (str): Output path for the JSON metrics summary.
    n_seeds (int): Number of random seeds. Mean +/- std is reported across seeds.
    random_state (int): Primary seed; seeds random_state ... random_state+n_seeds-1 are run.
    test_size (float): Fraction of slides held out as test (only used when split_col is None).
    val_size (float): Fraction of slides held out as val (only used when split_col is None).
    n_epochs (int): Number of training epochs per seed.
    lr (float): Adam learning rate.
    batch_size (int): Number of slides per batch. Tiles are zero-padded to the longest in batch.
    dtype (str): Dtype to cast features to at load time; simulates low-precision storage.
        Options: 'float32' (default), 'float16', 'bfloat16'.
        The model always trains in float32 regardless of this setting.
    head_dim (int): Hidden dimension for each ABMIL attention head.
    n_heads (int): Number of independent attention heads in ABMIL.
    dropout (float): Dropout probability within each attention head.
    gated (bool): If True, use gated ABMIL (sigmoid gating on attention).
    n_bootstrap (int): Bootstrap resamples for 95% CI on test AUROC (primary seed only).
    """

    features_dir: str = MISSING
    labels_parquet: str = MISSING
    target_col: str = MISSING
    split_col: Optional[str] = None
    output_summary_json: str = "results.json"
    n_seeds: int = 3
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.1
    n_epochs: int = 20
    lr: float = 1e-4
    batch_size: int = 8
    dtype: str = "float32"
    head_dim: int = 256
    n_heads: int = 8
    dropout: float = 0.0
    gated: bool = False
    n_bootstrap: int = 1000


desc_doc = """== ${hydra.help.app_name} ==
Benchmark ABMIL classification on patch features extracted from whole slide images.
"""

parameter_doc = f"""
== Available Parameters ==
{AbmilBenchmarkConfig.__doc__}
"""

cs = ConfigStore.instance()
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=desc_doc, footer=parameter_doc)),
    provider="hydra",
)
cs.store(name="abmil_benchmark_config", node=AbmilBenchmarkConfig)


# ── Data ──────────────────────────────────────────────────────────────────────


def _load_features(h5_path: Path, cast_dtype: torch.dtype) -> torch.Tensor:
    """Load patch features from an HDF5 file and cast to the requested dtype.

    Args:
        h5_path: Path to the ``.h5`` file. Must contain a ``'features'`` dataset
            of shape ``(n_tiles, feature_dim)``.
        cast_dtype: Target torch dtype. Features are cast to this dtype after loading,
            which may reduce precision (e.g. float32 → float16).

    Returns:
        Tensor of shape ``(n_tiles, feature_dim)`` in the requested dtype.
    """
    with h5py.File(h5_path, "r") as f:
        arr = np.array(f["features"])
    features = _numpy_to_torch(arr)  # handles bfloat16 |V2 transparently
    return features.to(cast_dtype)


class SlideDataset(Dataset):
    """Dataset of per-slide patch feature tensors loaded from HDF5 files.

    Args:
        slide_ids: Ordered list of slide identifiers.
        labels: Integer labels (0/1) aligned with ``slide_ids``.
        features_dir: Directory containing ``{slide_id}.h5`` files.
        cast_dtype: Torch dtype to cast features to at load time.
    """

    def __init__(
        self,
        slide_ids: List[str],
        labels: np.ndarray,
        features_dir: Path,
        cast_dtype: torch.dtype,
    ):
        self.slide_ids = slide_ids
        self.labels = labels
        self.features_dir = features_dir
        self.cast_dtype = cast_dtype

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = _load_features(
            self.features_dir / f"{self.slide_ids[idx]}.h5", self.cast_dtype
        )
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label


def _collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length tile sequences into a padded batch.

    Returns:
        padded: ``(B, N_max, D)`` zero-padded feature tensor (float32).
        labels: ``(B,)`` label tensor.
        mask: ``(B, N_max)`` boolean mask; True entries are valid tiles.
    """
    features_list, labels_list = zip(*batch)
    max_tiles = max(f.shape[0] for f in features_list)
    feature_dim = features_list[0].shape[1]

    padded = torch.zeros(len(features_list), max_tiles, feature_dim, dtype=torch.float32)
    mask = torch.zeros(len(features_list), max_tiles, dtype=torch.bool)
    for i, f in enumerate(features_list):
        n = f.shape[0]
        padded[i, :n] = f.float()  # upcast to float32; model always trains in float32
        mask[i, :n] = True

    return padded, torch.stack(labels_list), mask


# ── Model ─────────────────────────────────────────────────────────────────────


class AbmilClassifier(nn.Module):
    """ABMIL attention pooling followed by a binary classification head.

    Args:
        feature_dim: Input feature dimension.
        head_dim: Hidden dimension for each ABMIL attention head.
        n_heads: Number of independent attention heads.
        dropout: Dropout inside attention heads.
        gated: Enable sigmoid gating (Gated-ABMIL).
    """

    def __init__(
        self,
        feature_dim: int,
        head_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.0,
        gated: bool = False,
    ):
        super().__init__()
        self.abmil = ABMIL(
            feature_dim=feature_dim,
            head_dim=head_dim,
            n_heads=n_heads,
            dropout=dropout,
            n_branches=1,
            gated=gated,
        )
        self.head = nn.Linear(feature_dim, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, N, D)`` patch feature tensor.
            mask: ``(B, N)`` boolean mask; True = valid tile.

        Returns:
            ``(B,)`` logit tensor.
        """
        aggregated, _ = self.abmil(x, attn_mask=mask)  # (B, 1, D)
        aggregated = aggregated.squeeze(1)  # (B, D)
        return self.head(aggregated).squeeze(-1)  # (B,)


# ── Splits ────────────────────────────────────────────────────────────────────


def _split_by_slide(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random train / val / test split at slide level.

    All rows for a given slide land in the same partition to prevent data leakage.
    """
    slide_ids = np.sort(df["slide_id"].unique())
    rng = np.random.RandomState(seed)
    slide_ids = rng.permutation(slide_ids)

    n = len(slide_ids)
    n_test = max(1, int(round(n * test_size)))
    n_val = max(1, int(round(n * val_size)))

    test_ids = set(slide_ids[:n_test])
    val_ids = set(slide_ids[n_test : n_test + n_val])
    train_ids = set(slide_ids[n_test + n_val :])

    logger.info(
        "Split (seed=%d): train=%d  val=%d  test=%d slides",
        seed,
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )
    return (
        df[df["slide_id"].isin(train_ids)],
        df[df["slide_id"].isin(val_ids)],
        df[df["slide_id"].isin(test_ids)],
    )


# ── Evaluation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def _eval_auc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute slide-level AUROC over a dataloader."""
    model.eval()
    all_probs, all_labels = [], []
    for features, labels, mask in loader:
        features = features.to(device)
        mask = mask.to(device)
        logits = model(features, mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
        logger.warning("AUROC undefined: only one class present in labels.")
        return float("nan")
    return float(roc_auc_score(all_labels, all_probs))


def _bootstrap_ci_auc(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float]:
    """Bootstrap 95% CI for AUROC."""
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(labels)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_bs, p_bs = labels[idx], probs[idx]
        if y_bs.sum() in (0, n):
            continue
        aucs.append(roc_auc_score(y_bs, p_bs))
    if not aucs:
        return float("nan"), float("nan")
    lo, hi = float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
    return lo, hi


# ── Training ──────────────────────────────────────────────────────────────────


def _run_one_seed(
    cfg: AbmilBenchmarkConfig,
    df: pd.DataFrame,
    features_dir: Path,
    device: torch.device,
    seed: int,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Train ABMIL for one seed. Returns metrics dict and primary-seed arrays.

    Returns:
        metrics: dict with 'val' and 'test' sub-dicts containing 'auroc'.
        test_probs: predicted probabilities on the test set.
        test_labels: ground-truth labels on the test set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if cfg.split_col is not None:
        train_df = df[df[cfg.split_col] == "train"].reset_index(drop=True)
        val_df = df[df[cfg.split_col] == "val"].reset_index(drop=True)
        test_df = df[df[cfg.split_col] == "test"].reset_index(drop=True)
        logger.info(
            "Using split_col=%r: train=%d  val=%d  test=%d slides",
            cfg.split_col,
            len(train_df),
            len(val_df),
            len(test_df),
        )
    else:
        train_df, val_df, test_df = _split_by_slide(df, cfg.test_size, cfg.val_size, seed)

    cast_dtype = _TORCH_DTYPE[cfg.dtype]

    def _make_loader(split_df: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = SlideDataset(
            split_df["slide_id"].tolist(),
            split_df["y"].values,
            features_dir,
            cast_dtype,
        )
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            collate_fn=_collate_fn,
            num_workers=min(4, len(split_df)),
            pin_memory=device.type == "cuda",
        )

    # Infer feature_dim from the first slide in the dataset.
    first_h5 = features_dir / f"{df['slide_id'].iloc[0]}.h5"
    with h5py.File(first_h5, "r") as f:
        feature_dim = f["features"].shape[1]
    logger.info("feature_dim=%d  dtype=%s  device=%s", feature_dim, cfg.dtype, device)

    model = AbmilClassifier(
        feature_dim=feature_dim,
        head_dim=cfg.head_dim,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        gated=cfg.gated,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = _make_loader(train_df, shuffle=True)
    val_loader = _make_loader(val_df, shuffle=False)
    test_loader = _make_loader(test_df, shuffle=False)

    best_val_auc = float("-inf")
    # Initialise with current weights so load_state_dict never receives None.
    best_state: Dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(cfg.n_epochs):
        model.train()
        epoch_loss = 0.0
        for features, labels, mask in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            logits = model(features, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_auc = _eval_auc(model, val_loader, device)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        logger.debug(
            "epoch=%d/%d  loss=%.4f  val_auc=%.4f  best=%.4f",
            epoch + 1,
            cfg.n_epochs,
            epoch_loss / max(1, len(train_loader)),
            val_auc,
            best_val_auc,
        )

    model.load_state_dict(best_state)
    test_auc = _eval_auc(model, test_loader, device)
    logger.info("seed=%d  best_val_auc=%.4f  test_auc=%.4f", seed, best_val_auc, test_auc)

    # Collect raw probabilities and labels for bootstrap CI (caller uses primary seed only).
    test_probs, test_labels = [], []
    model.eval()
    with torch.no_grad():
        for features, labels, mask in test_loader:
            logits = model(features.to(device), mask.to(device))
            test_probs.append(torch.sigmoid(logits).cpu().numpy())
            test_labels.append(labels.numpy())
    test_probs = np.concatenate(test_probs)
    test_labels = np.concatenate(test_labels)

    return (
        {"val": {"auroc": best_val_auc}, "test": {"auroc": test_auc}},
        test_probs,
        test_labels,
    )


# ── JSON helpers ──────────────────────────────────────────────────────────────


def _sanitize_for_json(obj):
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path=".", config_name="abmil_benchmark_config")
def main(cfg: AbmilBenchmarkConfig):
    """Benchmark ABMIL on WSI patch features for the given dtype and target."""
    if cfg.dtype not in _VALID_DTYPES:
        raise ValueError(
            f"dtype={cfg.dtype!r} is not supported. Choose from {_VALID_DTYPES}."
        )

    features_dir = Path(cfg.features_dir)
    if not features_dir.is_dir():
        raise FileNotFoundError(f"features_dir does not exist: {features_dir}")

    # Load labels; drop slides without a corresponding H5 file.
    df = pd.read_parquet(cfg.labels_parquet, columns=_label_columns(cfg))
    available = {p.stem for p in features_dir.glob("*.h5")}
    before = len(df)
    df = df[df["slide_id"].isin(available)].reset_index(drop=True)
    if len(df) < before:
        logger.warning(
            "Dropped %d slides from labels parquet (no matching .h5 in features_dir).",
            before - len(df),
        )
    if df.empty:
        raise RuntimeError(
            "No slides remain after filtering. Check features_dir and labels_parquet."
        )

    # Rename target column to 'y'.
    df = df.rename(columns={cfg.target_col: "y"})
    df["y"] = df["y"].astype(int)
    pos_rate = df["y"].mean()
    logger.info(
        "Dataset: %d slides  pos_rate=%.3f  dtype=%s", len(df), pos_rate, cfg.dtype
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    seeds = [cfg.random_state + i for i in range(cfg.n_seeds)]
    all_val_metrics: List[Dict] = []
    all_test_metrics: List[Dict] = []
    primary_test_probs: Optional[np.ndarray] = None
    primary_test_labels: Optional[np.ndarray] = None

    for seed in seeds:
        logger.info("── seed=%d " + "─" * 40, seed)
        metrics, test_probs, test_labels = _run_one_seed(
            cfg, df, features_dir, device, seed
        )
        all_val_metrics.append(metrics["val"])
        all_test_metrics.append(metrics["test"])
        if seed == cfg.random_state:
            primary_test_probs = test_probs
            primary_test_labels = test_labels

    # ── Multi-seed summary ─────────────────────────────────────────────────────
    all_keys = sorted({k for m in all_val_metrics + all_test_metrics for k in m})
    summary: Dict = {
        "dtype": cfg.dtype,
        "target_col": cfg.target_col,
        "n_slides": int(len(df)),
        "pos_rate": float(pos_rate),
        "n_seeds": len(seeds),
        "seeds": seeds,
    }
    for split_name, records in [("val", all_val_metrics), ("test", all_test_metrics)]:
        summary[split_name] = {}
        for key in all_keys:
            vals = [r[key] for r in records if key in r]
            if not vals:
                continue
            mean, std = float(np.mean(vals)), float(np.std(vals))
            summary[split_name][key] = {"mean": mean, "std": std}
            logger.info("[%s] %s: %.4f +/- %.4f", split_name, key, mean, std)

    # ── Bootstrap CI on test AUROC (primary seed) ──────────────────────────────
    if primary_test_probs is not None and not np.isnan(
        summary["test"].get("auroc", {}).get("mean", float("nan"))
    ):
        ci_lo, ci_hi = _bootstrap_ci_auc(
            primary_test_probs,
            primary_test_labels,
            cfg.n_bootstrap,
            cfg.random_state,
        )
        summary["test"]["auroc"]["bootstrap_ci_95"] = [ci_lo, ci_hi]
        logger.info("[test] auroc bootstrap 95%% CI: [%.4f, %.4f]", ci_lo, ci_hi)

    with open(cfg.output_summary_json, "w") as f:
        json.dump(_sanitize_for_json(summary), f, indent=2)
    logger.info("Results written to %s", cfg.output_summary_json)


def _label_columns(cfg: AbmilBenchmarkConfig) -> List[str]:
    """Determine which parquet columns to load."""
    cols = ["slide_id", cfg.target_col]
    if cfg.split_col is not None:
        cols.append(cfg.split_col)
    return cols
