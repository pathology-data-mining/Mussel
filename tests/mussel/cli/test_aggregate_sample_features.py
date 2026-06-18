import h5py
import ml_dtypes
import numpy as np
import pytest
import torch

from mussel.utils.feature_extract import subsample_tiles


def _make_data(n_tiles, dim=4):
    rng = np.random.default_rng(0)
    features = rng.random((n_tiles, dim)).astype(np.float32)
    coords = rng.integers(0, 1000, (n_tiles, 2))
    return features, coords


def _write_fake_h5(path, n_tiles, dim=4, seed=0, feature_dtype=np.float32):
    """Write a synthetic feature H5 with 'features' and 'coords' datasets."""
    rng = np.random.default_rng(seed)
    features = rng.random((n_tiles, dim)).astype(feature_dtype)
    coords = rng.integers(0, 1000, (n_tiles, 2))
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=features)
        f.create_dataset("coords", data=coords)
    return features, coords


# =============================================================================
# subsample_tiles tests
# =============================================================================


def test_subsample_tiles_random_reduces_count():
    features, coords = _make_data(100)
    f_out, c_out = subsample_tiles(
        features, coords, max_tiles=30, strategy="random", slide_sizes=[100], seed=42
    )
    assert f_out.shape == (30, 4)
    assert c_out.shape == (30, 2)


def test_subsample_tiles_random_no_duplicates():
    # Use unique coords to guarantee no coord collisions in this test
    features = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
    coords = np.arange(100 * 2).reshape(100, 2)  # all coords unique by construction
    f_out, c_out = subsample_tiles(
        features, coords, max_tiles=50, strategy="random", slide_sizes=[100], seed=42
    )
    # sampling is without replacement — all selected feature rows should be distinct
    assert len(set(map(tuple, f_out.tolist()))) == 50


def test_subsample_tiles_random_reproducible():
    features, coords = _make_data(200)
    f1, c1 = subsample_tiles(
        features, coords, max_tiles=50, strategy="random", slide_sizes=[200], seed=7
    )
    f2, c2 = subsample_tiles(
        features, coords, max_tiles=50, strategy="random", slide_sizes=[200], seed=7
    )
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(c1, c2)


def test_subsample_tiles_proportional():
    # slide_sizes = [60, 40], max_tiles=20 → 12 from slide 0, 8 from slide 1
    rng = np.random.default_rng(0)
    f0 = rng.random((60, 4)).astype(np.float32)
    f1 = rng.random((40, 4)).astype(np.float32)
    c0 = np.zeros((60, 2), dtype=int)
    c1 = np.ones((40, 2), dtype=int)
    features = np.concatenate([f0, f1], axis=0)
    coords = np.concatenate([c0, c1], axis=0)
    f_out, c_out = subsample_tiles(
        features,
        coords,
        max_tiles=20,
        strategy="proportional",
        slide_sizes=[60, 40],
        seed=42,
    )
    assert f_out.shape[0] == 20
    assert c_out.shape[0] == 20


def test_subsample_tiles_equal():
    # slide_sizes = [80, 40], max_tiles=10 → 5 from each slide
    rng = np.random.default_rng(0)
    f0 = rng.random((80, 4)).astype(np.float32)
    f1 = rng.random((40, 4)).astype(np.float32)
    c0 = np.zeros((80, 2), dtype=int)
    c1 = np.ones((40, 2), dtype=int)
    features = np.concatenate([f0, f1], axis=0)
    coords = np.concatenate([c0, c1], axis=0)
    f_out, c_out = subsample_tiles(
        features, coords, max_tiles=10, strategy="equal", slide_sizes=[80, 40], seed=42
    )
    assert f_out.shape[0] == 10
    assert c_out.shape[0] == 10


def test_subsample_tiles_no_op_when_below_max():
    features, coords = _make_data(50)
    f_out, c_out = subsample_tiles(
        features, coords, max_tiles=100, strategy="random", slide_sizes=[50], seed=0
    )
    np.testing.assert_array_equal(f_out, features)
    np.testing.assert_array_equal(c_out, coords)


def test_subsample_tiles_invalid_strategy():
    features, coords = _make_data(50)
    with pytest.raises(ValueError, match="strategy"):
        subsample_tiles(
            features, coords, max_tiles=10, strategy="bogus", slide_sizes=[50], seed=0
        )


# =============================================================================
# aggregate_sample_features tests
# =============================================================================


from mussel.utils.feature_extract import (
    aggregate_sample_features as _aggregate_sample_features,
)


def test_aggregate_sample_features_invalid_shapes():
    """Per-slide validation raises informative ValueError on bad input."""
    feats, coords = _make_data(10)

    # 1-D features array
    with pytest.raises(ValueError, match="2-D"):
        _aggregate_sample_features(
            features_list=[feats.ravel()],
            coords_list=[coords],
            sample_ids=["s"],
        )

    # coords wrong second dim
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        _aggregate_sample_features(
            features_list=[feats],
            coords_list=[coords[:, :1]],
            sample_ids=["s"],
        )

    # mismatched lengths
    with pytest.raises(ValueError, match="different lengths"):
        _aggregate_sample_features(
            features_list=[feats],
            coords_list=[coords[:5]],
            sample_ids=["s"],
        )


def test_aggregate_sample_features_single_slide(tmp_path):
    """One slide per sample — output equals input."""
    feats_a, coords_a = _make_data(30)

    results = _aggregate_sample_features(
        features_list=[feats_a],
        coords_list=[coords_a],
        sample_ids=["sample1"],
        max_tiles=None,
        subsampling_strategy="random",
        seed=42,
    )

    assert "sample1" in results
    np.testing.assert_array_equal(results["sample1"][0], feats_a)
    np.testing.assert_array_equal(results["sample1"][1], coords_a)


def test_aggregate_sample_features_multi_slide(tmp_path):
    """Two slides per sample — features are concatenated."""
    rng = np.random.default_rng(0)
    feats_a = rng.random((20, 4)).astype(np.float32)
    coords_a = rng.integers(0, 1000, (20, 2))
    feats_b = rng.random((15, 4)).astype(np.float32)
    coords_b = rng.integers(0, 1000, (15, 2))

    results = _aggregate_sample_features(
        features_list=[feats_a, feats_b],
        coords_list=[coords_a, coords_b],
        sample_ids=["sampleX", "sampleX"],
        max_tiles=None,
    )

    assert results["sampleX"][0].shape == (35, 4)
    assert results["sampleX"][1].shape == (35, 2)
    np.testing.assert_array_equal(
        results["sampleX"][0], np.concatenate([feats_a, feats_b], axis=0)
    )
    np.testing.assert_array_equal(
        results["sampleX"][1], np.concatenate([coords_a, coords_b], axis=0)
    )


def test_aggregate_sample_features_save_pt_false(tmp_path):
    """save_pt=False — only H5 is written, no PT file."""
    h5_a = tmp_path / "slide_a.h5"
    _write_fake_h5(h5_a, n_tiles=10, seed=0)

    cfg = AggregateSampleFeaturesConfig(
        patch_features_h5_paths=[str(h5_a)],
        sample_ids=["s1"],
        output_dir=str(tmp_path / "out"),
        save_pt=False,
    )
    mussel.cli.aggregate_sample_features.main(OmegaConf.structured(cfg))

    assert (tmp_path / "out" / "s1.features.h5").exists()
    assert not (tmp_path / "out" / "s1.features.pt").exists()


def test_aggregate_sample_features_two_samples(tmp_path):
    """Three slides, two samples — two entries in result."""
    rng = np.random.default_rng(0)
    slides = [
        (rng.random((10, 4)).astype(np.float32), rng.integers(0, 1000, (10, 2)))
        for _ in range(3)
    ]
    features_list = [f for f, _ in slides]
    coords_list = [c for _, c in slides]

    results = _aggregate_sample_features(
        features_list=features_list,
        coords_list=coords_list,
        sample_ids=["sA", "sA", "sB"],
        max_tiles=None,
    )

    assert results["sA"][0].shape[0] == 20
    assert results["sB"][0].shape[0] == 10


def test_aggregate_sample_features_with_subsampling(tmp_path):
    """Subsampling reduces output to max_tiles, is reproducible, and keeps features/coords aligned."""
    # Use identifiable rows: feature row i has value i in all dims, coord row i
    # is (i, i). After subsampling, each selected feature row must equal its
    # corresponding coord row, proving the two arrays stay in sync.
    n_a, n_b = 80, 60
    feats_a = np.tile(np.arange(n_a, dtype=np.float32)[:, None], (1, 4))
    coords_a = np.tile(np.arange(n_a)[:, None], (1, 2))
    feats_b = np.tile(np.arange(n_a, n_a + n_b, dtype=np.float32)[:, None], (1, 4))
    coords_b = np.tile(np.arange(n_a, n_a + n_b)[:, None], (1, 2))

    def run():
        return _aggregate_sample_features(
            features_list=[feats_a, feats_b],
            coords_list=[coords_a, coords_b],
            sample_ids=["big", "big"],
            max_tiles=50,
            subsampling_strategy="random",
            seed=99,
        )

    r1 = run()
    r2 = run()
    f_out, c_out = r1["big"]

    assert f_out.shape[0] == 50
    assert c_out.shape[0] == 50

    # Reproducible with same seed
    np.testing.assert_array_equal(r1["big"][0], r2["big"][0])
    np.testing.assert_array_equal(r1["big"][1], r2["big"][1])

    # features and coords remain aligned: feature value == coord value for each row
    np.testing.assert_array_equal(f_out[:, 0].astype(np.int64), c_out[:, 0])


# =============================================================================
# CLI tests
# =============================================================================


from omegaconf import OmegaConf

import mussel.cli.aggregate_sample_features
from mussel.cli.aggregate_sample_features import AggregateSampleFeaturesConfig


def test_aggregate_sample_features_cli(tmp_path):
    """End-to-end CLI test using synthetic H5 files."""
    h5_a = tmp_path / "slide_a.h5"
    h5_b = tmp_path / "slide_b.h5"
    h5_c = tmp_path / "slide_c.h5"
    _write_fake_h5(h5_a, n_tiles=25, seed=10)
    _write_fake_h5(h5_b, n_tiles=20, seed=11)
    _write_fake_h5(h5_c, n_tiles=30, seed=12)

    out_dir = str(tmp_path / "samples")

    cfg = AggregateSampleFeaturesConfig(
        patch_features_h5_paths=[str(h5_a), str(h5_b), str(h5_c)],
        sample_ids=["P1", "P1", "P2"],
        output_dir=out_dir,
        max_tiles=30,
        subsampling_strategy="random",
        seed=0,
    )
    mussel.cli.aggregate_sample_features.main(OmegaConf.structured(cfg))

    with h5py.File(tmp_path / "samples" / "P1.features.h5") as f:
        assert f["features"].shape[0] == 30  # 25+20=45 → subsampled to 30
    with h5py.File(tmp_path / "samples" / "P2.features.h5") as f:
        assert f["features"].shape[0] == 30  # 30 ≤ 30, no subsampling
    assert (tmp_path / "samples" / "P1.features.pt").exists()
    assert (tmp_path / "samples" / "P2.features.pt").exists()


def test_cli_mismatched_lengths_raises(tmp_path):
    cfg = AggregateSampleFeaturesConfig(
        patch_features_h5_paths=["a.h5", "b.h5"],
        sample_ids=["s1"],  # wrong length
        output_dir=str(tmp_path),
    )
    with pytest.raises((ValueError, Exception)):
        mussel.cli.aggregate_sample_features.main(OmegaConf.structured(cfg))


# =============================================================================
# Precision upcast tests
# =============================================================================


def test_float16_features_upcast_to_float32(tmp_path):
    """float16 features stored in h5 are upcast to float32 in the output H5."""
    n_tiles, dim = 6, 8
    h5_path = tmp_path / "slide.h5"
    _write_fake_h5(h5_path, n_tiles=n_tiles, dim=dim, seed=0, feature_dtype=np.float16)

    # Read back the exact values written so expected matches what was stored.
    with h5py.File(h5_path) as f:
        expected = f["features"][:].astype(np.float32)

    cfg = AggregateSampleFeaturesConfig(
        patch_features_h5_paths=[str(h5_path)],
        sample_ids=["s1"],
        output_dir=str(tmp_path / "out"),
        save_pt=False,
    )
    mussel.cli.aggregate_sample_features.main(OmegaConf.structured(cfg))

    with h5py.File(tmp_path / "out" / "s1.features.h5") as f:
        out_features = f["features"][:]

    assert (
        out_features.dtype == np.float32
    ), f"Expected float32 output, got {out_features.dtype}"
    np.testing.assert_array_equal(out_features, expected)


def test_bfloat16_features_upcast_to_float32(tmp_path):
    """bfloat16 features (stored as |V2 opaque void in h5) are upcast to float32."""
    n_tiles, dim = 6, 8
    h5_path = tmp_path / "slide.h5"
    _write_fake_h5(
        h5_path, n_tiles=n_tiles, dim=dim, seed=0, feature_dtype=ml_dtypes.bfloat16
    )

    # Read back the exact opaque bytes written and reinterpret as float32.
    with h5py.File(h5_path) as f:
        raw = np.array(f["features"])
    expected = raw.view(ml_dtypes.bfloat16).astype(np.float32)

    cfg = AggregateSampleFeaturesConfig(
        patch_features_h5_paths=[str(h5_path)],
        sample_ids=["s1"],
        output_dir=str(tmp_path / "out"),
        save_pt=False,
    )
    mussel.cli.aggregate_sample_features.main(OmegaConf.structured(cfg))

    with h5py.File(tmp_path / "out" / "s1.features.h5") as f:
        out_features = f["features"][:]

    assert (
        out_features.dtype == np.float32
    ), f"Expected float32 output, got {out_features.dtype}"
    np.testing.assert_array_equal(out_features, expected)
