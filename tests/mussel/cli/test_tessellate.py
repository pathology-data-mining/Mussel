import os
import numpy as np
import h5py
from omegaconf import OmegaConf

import mussel.cli.tessellate
from mussel.cli.tessellate import TessellateConfig, SegConfig

# Dimensions of the test slide (85656 x 19917 at level 0)
_SLIDE_WIDTH  = 85656
_SLIDE_HEIGHT = 19917


def test_tessellate(tmp_path, num_workers):
    slide_path = "tests/testdata/948176.svs"
    patch_h5_path = tmp_path / "test.h5"
    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateConfig(
        slide_path=slide_path,
        output_h5_path=patch_h5_path,
        seg_config=seg_config,
        num_workers=num_workers,
    )
    mussel.cli.tessellate.main(OmegaConf.create(cfg))

    assert os.path.exists(patch_h5_path)

    with h5py.File(patch_h5_path, "r") as f:
        # H5 must have a 'coords' dataset
        assert "coords" in f, "H5 output missing 'coords' dataset"

        coords = f["coords"][:]
        attrs  = dict(f["coords"].attrs)

        # Shape and dtype
        assert coords.ndim == 2 and coords.shape[1] == 2, (
            f"coords should be (N, 2), got {coords.shape}"
        )
        assert coords.dtype == np.int64, f"coords dtype should be int64, got {coords.dtype}"

        # Must have produced some patches
        assert coords.shape[0] > 0, "tessellation produced zero patches"

        # Required metadata attributes
        for attr in ("patch_size", "mpp"):
            assert attr in attrs, f"coords missing attribute '{attr}'"

        # All coordinates must lie within the slide dimensions
        patch_size = int(attrs["patch_size"])
        assert np.all(coords[:, 0] >= 0), "negative x coordinates"
        assert np.all(coords[:, 1] >= 0), "negative y coordinates"
        assert np.all(coords[:, 0] + patch_size <= _SLIDE_WIDTH),  "x + patch_size exceeds slide width"
        assert np.all(coords[:, 1] + patch_size <= _SLIDE_HEIGHT), "y + patch_size exceeds slide height"



def test_seg_config_new_fields_defaults():
    """SegConfig has the new fields with correct defaults."""
    cfg = SegConfig()
    assert cfg.overlap == 0
    assert cfg.min_tissue_proportion == 0.0
    assert cfg.remove_artifacts is False
    assert cfg.remove_penmarks is False
    assert cfg.seg_model == "classic"


def test_seg_config_overlap_set():
    """SegConfig accepts a non-zero overlap."""
    cfg = SegConfig(overlap=64)
    assert cfg.overlap == 64


def test_seg_config_min_tissue_proportion_set():
    """SegConfig accepts min_tissue_proportion between 0 and 1."""
    cfg = SegConfig(min_tissue_proportion=0.5)
    assert cfg.min_tissue_proportion == 0.5


def test_seg_config_seg_model_neural():
    """SegConfig accepts seg_model='neural'."""
    cfg = SegConfig(seg_model="neural")
    assert cfg.seg_model == "neural"
