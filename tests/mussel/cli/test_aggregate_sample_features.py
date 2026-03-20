import h5py
import numpy as np
import pytest

from mussel.utils.feature_extract import subsample_tiles


def _make_data(n_tiles, dim=4):
    rng = np.random.default_rng(0)
    features = rng.random((n_tiles, dim)).astype(np.float32)
    coords = rng.integers(0, 1000, (n_tiles, 2))
    return features, coords


def _write_fake_h5(path, n_tiles, dim=4, seed=0):
    """Write a synthetic feature H5 with 'features' and 'coords' datasets."""
    rng = np.random.default_rng(seed)
    features = rng.random((n_tiles, dim)).astype(np.float32)
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
    f_out, c_out = subsample_tiles(features, coords, max_tiles=30, strategy="random", slide_sizes=[100], seed=42)
    assert f_out.shape == (30, 4)
    assert c_out.shape == (30, 2)


def test_subsample_tiles_random_no_duplicates():
    # Use unique coords to guarantee no coord collisions in this test
    features = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
    coords = np.arange(100 * 2).reshape(100, 2)  # all coords unique by construction
    f_out, c_out = subsample_tiles(features, coords, max_tiles=50, strategy="random", slide_sizes=[100], seed=42)
    # sampling is without replacement — all selected feature rows should be distinct
    assert len(set(map(tuple, f_out.tolist()))) == 50


def test_subsample_tiles_random_reproducible():
    features, coords = _make_data(200)
    f1, c1 = subsample_tiles(features, coords, max_tiles=50, strategy="random", slide_sizes=[200], seed=7)
    f2, c2 = subsample_tiles(features, coords, max_tiles=50, strategy="random", slide_sizes=[200], seed=7)
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
    f_out, c_out = subsample_tiles(features, coords, max_tiles=20, strategy="proportional", slide_sizes=[60, 40], seed=42)
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
    f_out, c_out = subsample_tiles(features, coords, max_tiles=10, strategy="equal", slide_sizes=[80, 40], seed=42)
    assert f_out.shape[0] == 10
    assert c_out.shape[0] == 10


def test_subsample_tiles_no_op_when_below_max():
    features, coords = _make_data(50)
    f_out, c_out = subsample_tiles(features, coords, max_tiles=100, strategy="random", slide_sizes=[50], seed=0)
    np.testing.assert_array_equal(f_out, features)
    np.testing.assert_array_equal(c_out, coords)


def test_subsample_tiles_invalid_strategy():
    features, coords = _make_data(50)
    with pytest.raises(ValueError, match="strategy"):
        subsample_tiles(features, coords, max_tiles=10, strategy="bogus", slide_sizes=[50], seed=0)


# =============================================================================
# aggregate_sample_features tests
# =============================================================================


from mussel.utils.feature_extract import aggregate_sample_features as _aggregate_sample_features


def test_aggregate_sample_features_single_slide(tmp_path):
    """One slide per sample — output equals input."""
    h5_a = tmp_path / "slide_a.h5"
    feats_a, coords_a = _write_fake_h5(h5_a, n_tiles=30)

    _aggregate_sample_features(
        patch_features_h5_paths=[str(h5_a)],
        sample_ids=["sample1"],
        output_dir=str(tmp_path / "out"),
        output_h5_suffix="features.h5",
        max_tiles=None,
        subsampling_strategy="random",
        seed=42,
    )

    out_h5 = tmp_path / "out" / "sample1.features.h5"
    assert out_h5.exists()
    with h5py.File(out_h5) as f:
        np.testing.assert_array_equal(f["features"][:], feats_a)
        np.testing.assert_array_equal(f["coords"][:], coords_a)


def test_aggregate_sample_features_multi_slide(tmp_path):
    """Two slides per sample — features are concatenated."""
    h5_a = tmp_path / "slide_a.h5"
    h5_b = tmp_path / "slide_b.h5"
    _write_fake_h5(h5_a, n_tiles=20, seed=1)
    _write_fake_h5(h5_b, n_tiles=15, seed=2)

    _aggregate_sample_features(
        patch_features_h5_paths=[str(h5_a), str(h5_b)],
        sample_ids=["sampleX", "sampleX"],
        output_dir=str(tmp_path / "out"),
        max_tiles=None,
    )

    out_h5 = tmp_path / "out" / "sampleX.features.h5"
    assert out_h5.exists()
    with h5py.File(out_h5) as f:
        assert f["features"].shape == (35, 4)
        assert f["coords"].shape == (35, 2)


def test_aggregate_sample_features_two_samples(tmp_path):
    """Three slides, two samples — two output files."""
    paths = [tmp_path / f"s{i}.h5" for i in range(3)]
    for i, p in enumerate(paths):
        _write_fake_h5(p, n_tiles=10, seed=i)

    _aggregate_sample_features(
        patch_features_h5_paths=[str(p) for p in paths],
        sample_ids=["sA", "sA", "sB"],
        output_dir=str(tmp_path / "out"),
        max_tiles=None,
    )

    out_a = tmp_path / "out" / "sA.features.h5"
    out_b = tmp_path / "out" / "sB.features.h5"
    assert out_a.exists() and out_b.exists()
    with h5py.File(out_a) as f:
        assert f["features"].shape[0] == 20
    with h5py.File(out_b) as f:
        assert f["features"].shape[0] == 10


def test_aggregate_sample_features_with_subsampling(tmp_path):
    """Subsampling reduces output to max_tiles."""
    h5_a = tmp_path / "s0.h5"
    h5_b = tmp_path / "s1.h5"
    _write_fake_h5(h5_a, n_tiles=80, seed=0)
    _write_fake_h5(h5_b, n_tiles=60, seed=1)

    _aggregate_sample_features(
        patch_features_h5_paths=[str(h5_a), str(h5_b)],
        sample_ids=["big", "big"],
        output_dir=str(tmp_path / "out"),
        max_tiles=50,
        subsampling_strategy="random",
        seed=99,
    )

    out_h5 = tmp_path / "out" / "big.features.h5"
    with h5py.File(out_h5) as f:
        assert f["features"].shape[0] == 50


# =============================================================================
# CLI tests
# =============================================================================


import mussel.cli.aggregate_sample_features
from mussel.cli.aggregate_sample_features import AggregateSampleFeaturesConfig
from omegaconf import OmegaConf


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
        assert f["features"].shape[0] == 30   # 25+20=45 → subsampled to 30
    with h5py.File(tmp_path / "samples" / "P2.features.h5") as f:
        assert f["features"].shape[0] == 30   # 30 ≤ 30, no subsampling


def test_cli_mismatched_lengths_raises(tmp_path):
    cfg = AggregateSampleFeaturesConfig(
        patch_features_h5_paths=["a.h5", "b.h5"],
        sample_ids=["s1"],  # wrong length
        output_dir=str(tmp_path),
    )
    with pytest.raises((ValueError, Exception)):
        mussel.cli.aggregate_sample_features.main(OmegaConf.structured(cfg))
