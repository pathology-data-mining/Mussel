"""Integration tests for TensorFlow-based models (requires ``--extra tensorflow-gpu/cpu``).

Run via SLURM (see ``tests/slurm/test_tensorflow.sh``) or manually after installing
the tensorflow extra:

    UV_PROJECT_ENVIRONMENT=.venv-tensorflow uv sync --extra tensorflow-gpu
    UV_PROJECT_ENVIRONMENT=.venv-tensorflow uv run pytest \\
        tests/mussel/models/test_tensorflow_models.py --use-gpu -v

These tests are skipped automatically when TensorFlow is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

tensorflow = pytest.importorskip("tensorflow", reason="tensorflow extra not installed")

from mussel.models.model_factory import ModelType
from mussel.utils.feature_extract import extract_patch_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TESTDATA = Path(__file__).parent.parent.parent / "testdata"
_SLIDE_PATH = str(_TESTDATA / "948176.svs")
_PATCH_H5 = str(_TESTDATA / "948176.patch.h5")

_TF_PATCH_ENCODERS = [
    ModelType.GOOGLEPATH,
]

_EXPECTED_DIM = {
    ModelType.GOOGLEPATH: 1024,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_tensorflow
@pytest.mark.timeout(600)
@pytest.mark.parametrize("model_type", _TF_PATCH_ENCODERS, ids=lambda m: m.name)
def test_tensorflow_patch_encoder_extracts_features(tmp_path, model_type, use_gpu):
    """Load each TensorFlow patch encoder, run inference on the test slide.

    Checks:
    - Output HDF5 contains ``features`` and ``coords`` datasets.
    - ``features`` shape is ``(N, D)`` with N > 0 and D == expected dimension.
    - Features are finite and non-zero.
    """
    output_h5 = str(tmp_path / f"{model_type.name}.h5")

    extract_patch_features(
        patch_h5_path=_PATCH_H5,
        slide_path=_SLIDE_PATH,
        output_h5_path=output_h5,
        model_type=model_type,
        batch_size=16,
        use_gpu=use_gpu,
        num_workers=0,
        pin_memory=False,
        is_test_run=True,
    )

    import h5py

    with h5py.File(output_h5, "r") as f:
        assert "features" in f, "Output HDF5 missing 'features'"
        assert "coords" in f, "Output HDF5 missing 'coords'"
        features = f["features"][:]
        coords = f["coords"][:]

    n, d = features.shape
    assert n > 0, f"{model_type.name}: expected at least one patch, got 0"
    assert np.all(np.isfinite(features)), f"{model_type.name}: features contain NaN or Inf"
    assert not np.all(features == 0), f"{model_type.name}: all features are zero"
    assert coords.shape == (n, 2), f"{model_type.name}: coords shape {coords.shape} != ({n}, 2)"

    if model_type in _EXPECTED_DIM:
        assert d == _EXPECTED_DIM[model_type], (
            f"{model_type.name}: expected dim {_EXPECTED_DIM[model_type]}, got {d}"
        )
