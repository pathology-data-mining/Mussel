"""Integration tests for fastattn-based models (requires ``--extra fastattn``).

Run via SLURM (see ``tests/slurm/test_fastattn.sh``) or manually after installing
the fastattn extra:

    UV_PROJECT_ENVIRONMENT=.venv-fastattn uv sync --extra fastattn
    UV_PROJECT_ENVIRONMENT=.venv-fastattn uv run pytest \\
        tests/mussel/models/test_fastattn_models.py --use-gpu -v

These tests are skipped automatically when gigapath is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

gigapath = pytest.importorskip("gigapath", reason="fastattn extra not installed")

from mussel.models.model_factory import ModelType, get_required_patch_encoder
from mussel.utils.feature_extract import extract_patch_features, _apply_slide_aggregation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TESTDATA = Path(__file__).parent.parent.parent / "testdata"
_SLIDE_PATH = str(_TESTDATA / "948176.svs")
_PATCH_H5 = str(_TESTDATA / "948176.patch.h5")

_GIGAPATH_PATCH_DIM = 1536


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_fastattn
@pytest.mark.timeout(600)
def test_gigapath_patch_encoder_extracts_features(tmp_path, use_gpu):
    """Load Prov-GigaPath patch encoder, run inference on the test slide.

    Checks:
    - Output HDF5 contains ``features`` and ``coords`` datasets.
    - Feature dimension is 1536.
    - Features are finite and non-zero.
    """
    output_h5 = str(tmp_path / "GIGAPATH.h5")

    extract_patch_features(
        patch_h5_path=_PATCH_H5,
        slide_path=_SLIDE_PATH,
        output_h5_path=output_h5,
        model_type=ModelType.GIGAPATH,
        batch_size=16,
        use_gpu=use_gpu,
        num_workers=0,
        pin_memory=False,
        is_test_run=True,
    )

    import h5py

    with h5py.File(output_h5, "r") as f:
        features = f["features"][:]
        coords = f["coords"][:]

    n, d = features.shape
    assert n > 0
    assert d == _GIGAPATH_PATCH_DIM, f"expected dim {_GIGAPATH_PATCH_DIM}, got {d}"
    assert np.all(np.isfinite(features))
    assert not np.all(features == 0)
    assert coords.shape == (n, 2)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_fastattn
@pytest.mark.timeout(300)
def test_gigapath_slide_encoder_aggregates_features(tmp_path, use_gpu):
    """Load Prov-GigaPath slide encoder, aggregate synthetic patch features.

    Checks:
    - Output is a 1-D finite non-zero numpy array.
    """
    rng = np.random.default_rng(42)
    n_patches, patch_size_native = 32, 512
    fake_features = rng.standard_normal((n_patches, _GIGAPATH_PATCH_DIM)).astype(np.float32)
    fake_features /= np.linalg.norm(fake_features, axis=1, keepdims=True) + 1e-8
    fake_coords = np.stack(
        [np.arange(n_patches) * patch_size_native, np.zeros(n_patches, dtype=np.int64)],
        axis=1,
    ).astype(np.int64)

    result = _apply_slide_aggregation(
        features=fake_features,
        aggregation_method="model",
        slide_model_type=ModelType.GIGAPATH_SLIDE,
        use_gpu=use_gpu,
        coords=fake_coords,
        patch_size=patch_size_native,
    )

    assert isinstance(result, np.ndarray)
    assert result.ndim >= 1 and result.size > 0
    assert np.all(np.isfinite(result))
    assert not np.all(result == 0)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_fastattn
@pytest.mark.timeout(900)
def test_gigapath_end_to_end(tmp_path, use_gpu):
    """End-to-end: patch encode with GIGAPATH then slide aggregate with GIGAPATH_SLIDE."""
    output_h5 = str(tmp_path / "GIGAPATH_e2e.h5")

    extract_patch_features(
        patch_h5_path=_PATCH_H5,
        slide_path=_SLIDE_PATH,
        output_h5_path=output_h5,
        model_type=ModelType.GIGAPATH,
        batch_size=16,
        use_gpu=use_gpu,
        num_workers=0,
        pin_memory=False,
        is_test_run=True,
    )

    import h5py

    with h5py.File(output_h5, "r") as f:
        features = f["features"][:]
        coords = f["coords"][:]

    result = _apply_slide_aggregation(
        features=features,
        aggregation_method="model",
        slide_model_type=ModelType.GIGAPATH_SLIDE,
        use_gpu=use_gpu,
        coords=coords,
        patch_size=512,
    )

    assert isinstance(result, np.ndarray)
    assert result.ndim >= 1 and result.size > 0
    assert np.all(np.isfinite(result))
    assert not np.all(result == 0)
