import os
from omegaconf import OmegaConf

import mussel.cli.tessellate
from mussel.cli.tessellate import TessellateConfig, SegConfig

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
