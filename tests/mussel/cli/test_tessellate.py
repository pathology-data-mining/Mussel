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


