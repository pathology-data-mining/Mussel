"""Integration tests for patch and slide encoders on a real WSI.

These tests load actual model weights (from HuggingFace or local paths) and run
inference on the test slide ``tests/testdata/948176.svs``.

Tests are parametrized over all supported model types. Each test skips
gracefully when a model is unavailable (gated HF repo, missing local
checkpoint, network error, etc.).

Run (GPU recommended):
    uv run pytest tests/mussel/models/test_encoder_integration.py -m integration --use-gpu -v
    uv run pytest tests/mussel/models/test_encoder_integration.py -k RESNET50 --use-gpu -v
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from mussel.models.model_factory import (
    ModelType,
    SLIDE_ENCODER_COMPATIBILITY,
    get_required_patch_encoder,
)
from mussel.utils.feature_extract import extract_patch_features, _apply_slide_aggregation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TESTDATA = Path(__file__).parent.parent.parent / "testdata"
_SLIDE_PATH = str(_TESTDATA / "948176.svs")
_PATCH_H5 = str(_TESTDATA / "948176.patch.h5")

# All patch encoder model types (excludes slide encoders)
_SLIDE_ENCODERS = set(SLIDE_ENCODER_COMPATIBILITY.keys())
_PATCH_ENCODER_TYPES = [mt for mt in ModelType if mt not in _SLIDE_ENCODERS]
_SLIDE_ENCODER_TYPES = list(_SLIDE_ENCODERS)

# Expected feature dimension per patch encoder (used to synthesise fake features
# for slide-encoder-only tests).
_PATCH_ENCODER_DIM: dict[ModelType, int] = {
    ModelType.RESNET50:    1024,   # custom baseline stops at layer3 (256*4=1024)
    ModelType.CTRANSPATH:   768,
    ModelType.GIGAPATH:    1536,
    ModelType.VIRCHOW:     2560,
    ModelType.VIRCHOW2:    2560,
    ModelType.OPTIMUS:     1536,
    ModelType.CLIP:         512,
    ModelType.GOOGLEPATH:  1024,
    ModelType.CONCH1_5:     768,
    ModelType.UNI:         1024,
    ModelType.UNI2:        1536,
    ModelType.PHIKON:       768,
    ModelType.PHIKON_V2:   1024,   # DINOv2 ViT-L/14
    ModelType.H_OPTIMUS_1: 1536,
    ModelType.H0_MINI:     1152,
    ModelType.MIDNIGHT12K: 1536,   # DINOv2 ViT-g/14
    ModelType.GPFM:        1024,   # ViT-L/14 (embed_dim=1024, num_classes=0)
    ModelType.HIBOU_L:     1024,
}

# Per-slide-encoder override for the INPUT patch feature dimension.
# Used when a slide encoder was trained on a different patch encoder than the
# one listed in SLIDE_ENCODER_COMPATIBILITY (e.g. MADELEINE was trained on
# CONCH v1.0 at 512-dim, but SLIDE_ENCODER_COMPATIBILITY maps it to CONCH1_5
# which now outputs 768-dim).
_SLIDE_ENCODER_INPUT_DIM: dict[ModelType, int] = {
    ModelType.MADELEINE_SLIDE: 512,  # trained on CONCH v1.0 (512-dim), not CONCH1.5 (768-dim)
}



def _skip_on_load_failure(fn):
    """Decorator: run fn and pytest.skip on any model-load/network error."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            # HF gated model, missing file, network issues, or dynamic module download failures → skip
            if any(kw in msg for kw in [
                "401", "403", "access", "gated", "permission", "not found",
                "no such file", "checkpoint", "notimplemented", "cannot open",
                "connection", "timeout", "huggingface", "hf hub",
                "no module named 'transformers_modules",
                "no module named 'tensorflow",
                "no module named 'fastattn",
                "no module named 'gigapath",
            ]):
                pytest.skip(f"Model unavailable: {exc}")
            raise  # unexpected error → real failure

    return wrapper


# ---------------------------------------------------------------------------
# Patch encoder integration tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
@pytest.mark.parametrize("model_type", _PATCH_ENCODER_TYPES, ids=lambda m: m.name)
def test_patch_encoder_extracts_features(tmp_path, model_type, use_gpu):
    """Load each patch encoder, run inference on the test slide, verify output.

    Checks:
    - Output HDF5 contains ``features`` and ``coords`` datasets.
    - ``features`` shape is ``(N, D)`` with N > 0 and D > 0.
    - ``coords`` shape is ``(N, 2)``.
    - Features are finite and non-zero (model produced real output).
    - Coordinates match patch-level expectations.
    """
    output_h5 = str(tmp_path / f"{model_type.name}.h5")

    @_skip_on_load_failure
    def run():
        extract_patch_features(
            patch_h5_path=_PATCH_H5,
            slide_path=_SLIDE_PATH,
            output_h5_path=output_h5,
            model_type=model_type,
            batch_size=16,
            use_gpu=use_gpu,
            num_workers=0,
            pin_memory=False,
            is_test_run=True,  # process only first 3 batches → fast
        )

    run()

    import h5py
    with h5py.File(output_h5, "r") as f:
        assert "features" in f, "Output HDF5 missing 'features'"
        assert "coords" in f, "Output HDF5 missing 'coords'"
        features = f["features"][:]
        coords = f["coords"][:]

    n, d = features.shape
    assert n > 0, f"{model_type.name}: expected at least one patch, got 0"
    assert d > 0, f"{model_type.name}: feature dimension is 0"
    assert coords.shape == (n, 2), (
        f"{model_type.name}: coords shape {coords.shape} != ({n}, 2)"
    )
    assert np.all(np.isfinite(features)), (
        f"{model_type.name}: features contain NaN or Inf"
    )
    assert not np.all(features == 0), (
        f"{model_type.name}: all features are zero (model not producing output)"
    )

    # If expected dimension is known, validate it
    if model_type in _PATCH_ENCODER_DIM:
        expected_d = _PATCH_ENCODER_DIM[model_type]
        assert d == expected_d, (
            f"{model_type.name}: expected dim {expected_d}, got {d}"
        )


# ---------------------------------------------------------------------------
# Slide encoder integration tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(300)
@pytest.mark.parametrize("slide_model_type", _SLIDE_ENCODER_TYPES, ids=lambda m: m.name)
def test_slide_encoder_aggregates_features(tmp_path, slide_model_type, use_gpu):
    """Load each slide encoder, run aggregation on synthetic patch features, verify output.

    Synthetic patch features of the correct dimension for the required patch encoder
    are used so that this test doesn't depend on patch encoder availability.

    Checks:
    - Slide embedding is a 1-D numpy array of finite, non-zero values.
    - Output can be saved and reloaded as a torch tensor.
    """
    required_patch_enc = get_required_patch_encoder(slide_model_type)
    # Use per-slide-encoder input dim override if available (e.g. MADELEINE was trained
    # on CONCH v1.0 at 512-dim, not the current CONCH1.5 at 768-dim).
    patch_dim = _SLIDE_ENCODER_INPUT_DIM.get(slide_model_type) or _PATCH_ENCODER_DIM.get(required_patch_enc)
    if patch_dim is None:
        pytest.skip(f"Feature dim for {required_patch_enc.name} not in _PATCH_ENCODER_DIM")

    n_patches = 32
    rng = np.random.default_rng(42)
    # Use realistic feature scale (unit-normalised embeddings)
    fake_features = rng.standard_normal((n_patches, patch_dim)).astype(np.float32)
    fake_features /= np.linalg.norm(fake_features, axis=1, keepdims=True) + 1e-8

    # Some slide encoders need coordinates
    patch_size_native = 512  # 256 px at 0.5 mpp
    fake_coords = np.stack([
        np.arange(n_patches) * patch_size_native,
        np.zeros(n_patches, dtype=np.int64),
    ], axis=1).astype(np.int64)

    @_skip_on_load_failure
    def run():
        return _apply_slide_aggregation(
            features=fake_features,
            aggregation_method="model",
            slide_model_type=slide_model_type,
            use_gpu=use_gpu,
            coords=fake_coords,
            patch_size=patch_size_native,
        )

    result = run()

    assert isinstance(result, np.ndarray), (
        f"{slide_model_type.name}: expected np.ndarray, got {type(result)}"
    )
    assert result.ndim >= 1, f"{slide_model_type.name}: result has 0 dims"
    assert result.size > 0, f"{slide_model_type.name}: result is empty"
    assert np.all(np.isfinite(result)), (
        f"{slide_model_type.name}: result contains NaN or Inf"
    )
    assert not np.all(result == 0), (
        f"{slide_model_type.name}: all result values are zero"
    )

    # Verify the embedding can be saved as a torch tensor
    output_pt = str(tmp_path / f"{slide_model_type.name}.pt")
    torch.save(torch.tensor(result), output_pt)
    loaded = torch.load(output_pt, weights_only=True)
    assert loaded.shape == torch.tensor(result).shape


# ---------------------------------------------------------------------------
# End-to-end: full pipeline for fast models (patch encode → slide aggregate)
# ---------------------------------------------------------------------------

_E2E_PAIRS = [
    # (patch_encoder, slide_encoder) — pairs usable without HF auth
    # Extend this list when more public models are available.
]

# Conditionally add GIGAPATH pair if token is available (used in existing tests)
# We don't hard-require it — skip if auth fails.
_E2E_PAIRS_ALL = [
    (get_required_patch_encoder(se), se)
    for se in _SLIDE_ENCODER_TYPES
]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "patch_model_type,slide_model_type",
    _E2E_PAIRS_ALL,
    ids=lambda m: m.name,
)
def test_end_to_end_patch_then_slide_encode(
    tmp_path, patch_model_type, slide_model_type, use_gpu
):
    """Full end-to-end: tessellated slide → patch encode → slide encode.

    Uses extract_patch_features with is_test_run=True (3 batches only) then
    _apply_slide_aggregation with the real slide encoder model.
    """
    patch_h5_out = str(tmp_path / "patch_features.h5")

    @_skip_on_load_failure
    def run_patch():
        extract_patch_features(
            patch_h5_path=_PATCH_H5,
            slide_path=_SLIDE_PATH,
            output_h5_path=patch_h5_out,
            model_type=patch_model_type,
            batch_size=8,
            use_gpu=use_gpu,
            num_workers=0,
            pin_memory=False,
            is_test_run=True,
        )

    run_patch()

    import h5py
    with h5py.File(patch_h5_out, "r") as f:
        features = f["features"][:]
        coords = f["coords"][:]

    assert features.shape[0] > 0, "No patches extracted"

    @_skip_on_load_failure
    def run_slide():
        return _apply_slide_aggregation(
            features=features,
            aggregation_method="model",
            slide_model_type=slide_model_type,
            use_gpu=use_gpu,
            coords=coords,
            patch_size=512,
        )

    result = run_slide()

    assert isinstance(result, np.ndarray)
    assert result.size > 0
    assert np.all(np.isfinite(result)), f"{slide_model_type.name}: non-finite values in result"
