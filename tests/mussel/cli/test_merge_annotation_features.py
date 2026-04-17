"""Tests for mussel.cli.merge_annotation_features.

Covers the class_mapping branch which was previously buggy:
- annotation polygons were built from the remapped image but labeled with
  raw class IDs, causing non-tumor tiles to merge with background and be
  filtered out, leaving only tumor tiles.
"""

import tempfile
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pytest
import yaml
from PIL import Image

from mussel.cli.merge_annotation_features import MergeAnnotationFeaturesConfig, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_h5(path, n_tiles=10, n_features=4, tile_size=256):
    """Write a minimal HDF5 with coords and features."""
    coords = np.array([[i * tile_size, 0] for i in range(n_tiles)], dtype=np.int32)
    features = np.random.default_rng(0).standard_normal((n_tiles, n_features)).astype(np.float32)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("coords", data=coords)
        ds.attrs["patch_size"] = tile_size
        f.create_dataset("features", data=features)


def _make_bmp(path, width, height, annotation_map):
    """
    Write a BMP where each region is filled with a pixel value.

    annotation_map: dict of {pixel_value: (x_start, x_end)} column slices
    """
    arr = np.zeros((height, width), dtype=np.uint8)
    for pv, (x0, x1) in annotation_map.items():
        arr[:, x0:x1] = pv
    Image.fromarray(arr).save(path)


def _make_class_mapping_yaml(path, mapping):
    with open(path, "w") as f:
        yaml.dump(mapping, f)


# ---------------------------------------------------------------------------
# Tests — without class_mapping (baseline, regression guard)
# ---------------------------------------------------------------------------


def test_no_class_mapping_keeps_nonzero_annotations(tmp_path):
    """Without class_mapping, tiles overlapping non-zero BMP regions are kept."""
    n_tiles, tile_size = 4, 256
    _make_h5(tmp_path / "features.h5", n_tiles=n_tiles, tile_size=tile_size)

    # 4 tiles wide; pixel value = tile column index+1 (all non-zero)
    width = n_tiles * tile_size
    _make_bmp(
        tmp_path / "labels.bmp",
        width=width,
        height=tile_size,
        annotation_map={1: (0, width // 2), 2: (width // 2, width)},
    )

    out_parquet = str(tmp_path / "out.parquet")
    cfg = MergeAnnotationFeaturesConfig(
        features_h5_path=str(tmp_path / "features.h5"),
        annotation_bmp_path=str(tmp_path / "labels.bmp"),
        output_parquet_path=out_parquet,
        slide_id="test_slide",
    )
    main(cfg)

    gdf = gpd.read_parquet(out_parquet)
    assert len(gdf) > 0, "Expected annotated tiles in output"
    assert set(gdf["annotation"].unique()) <= {1, 2}


# ---------------------------------------------------------------------------
# Tests — with class_mapping (the critical bug fix)
# ---------------------------------------------------------------------------


def test_class_mapping_preserves_both_mapped_classes(tmp_path):
    """
    With class_mapping, both mapped classes (0=non-tumor, 1=tumor) must appear
    in the output parquet.  The old code would merge non-tumor with background
    and then filter it out, leaving only tumor tiles.
    """
    n_tiles, tile_size = 6, 256
    _make_h5(tmp_path / "features.h5", n_tiles=n_tiles, tile_size=tile_size)

    width = n_tiles * tile_size
    # Left half → raw class 1 (tumor in mapping), right half → raw class 4 (non-tumor)
    _make_bmp(
        tmp_path / "labels.bmp",
        width=width,
        height=tile_size,
        annotation_map={1: (0, width // 2), 4: (width // 2, width)},
    )

    # 1 → 1 (tumor), 4 → 0 (non-tumor)
    mapping = {1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0}
    _make_class_mapping_yaml(tmp_path / "mapping.yaml", mapping)

    out_parquet = str(tmp_path / "out.parquet")
    cfg = MergeAnnotationFeaturesConfig(
        features_h5_path=str(tmp_path / "features.h5"),
        annotation_bmp_path=str(tmp_path / "labels.bmp"),
        output_parquet_path=out_parquet,
        slide_id="test_slide",
        class_mapping_yaml_path=str(tmp_path / "mapping.yaml"),
    )
    main(cfg)

    gdf = gpd.read_parquet(out_parquet)
    assert len(gdf) > 0, "Expected tiles in output"
    annotation_values = set(gdf["annotation"].unique())
    assert 0 in annotation_values, "Non-tumor tiles (annotation=0) must be present"
    assert 1 in annotation_values, "Tumor tiles (annotation=1) must be present"


def test_class_mapping_annotation_values_are_mapped(tmp_path):
    """annotation column contains mapped values (0/1), not original raw class IDs."""
    n_tiles, tile_size = 4, 256
    _make_h5(tmp_path / "features.h5", n_tiles=n_tiles, tile_size=tile_size)

    width = n_tiles * tile_size
    _make_bmp(
        tmp_path / "labels.bmp",
        width=width,
        height=tile_size,
        annotation_map={2: (0, width // 2), 5: (width // 2, width)},
    )

    mapping = {2: 1, 5: 0}
    _make_class_mapping_yaml(tmp_path / "mapping.yaml", mapping)

    out_parquet = str(tmp_path / "out.parquet")
    cfg = MergeAnnotationFeaturesConfig(
        features_h5_path=str(tmp_path / "features.h5"),
        annotation_bmp_path=str(tmp_path / "labels.bmp"),
        output_parquet_path=out_parquet,
        class_mapping_yaml_path=str(tmp_path / "mapping.yaml"),
    )
    main(cfg)

    gdf = gpd.read_parquet(out_parquet)
    raw_ids = {2, 5}
    for raw_id in raw_ids:
        assert raw_id not in gdf["annotation"].unique(), (
            f"Raw class ID {raw_id} should not appear; annotation should be mapped values"
        )
    assert set(gdf["annotation"].unique()) <= {0, 1}


def test_unannotated_background_excluded(tmp_path):
    """Unannotated background (pixel=0) tiles are always excluded from output."""
    n_tiles, tile_size = 4, 256
    _make_h5(tmp_path / "features.h5", n_tiles=n_tiles, tile_size=tile_size)

    width = n_tiles * tile_size
    # First tile = background (0), rest = annotation class 1
    _make_bmp(
        tmp_path / "labels.bmp",
        width=width,
        height=tile_size,
        annotation_map={1: (tile_size, width)},  # tile_size..width annotated; 0..tile_size = 0
    )

    out_parquet = str(tmp_path / "out.parquet")
    cfg = MergeAnnotationFeaturesConfig(
        features_h5_path=str(tmp_path / "features.h5"),
        annotation_bmp_path=str(tmp_path / "labels.bmp"),
        output_parquet_path=out_parquet,
    )
    main(cfg)

    if Path(out_parquet).exists():
        gdf = gpd.read_parquet(out_parquet)
        assert 0 not in gdf["annotation"].unique(), "Background (annotation=0) should be excluded"
