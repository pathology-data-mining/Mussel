import os
from omegaconf import OmegaConf

import mussel.cli.tessellate
from mussel.cli.tessellate import TessellateConfig

def test_tessellate(tmp_path):
    slide_path = "tests/testdata/948176.svs"
    patch_h5_path = tmp_path / "test.h5"
    cfg = TessellateConfig(
        slide_path=slide_path,
        output_h5_path=patch_h5_path,
        segment_threshold=0,
        num_workers=1,
    )
    mussel.cli.tessellate.main(OmegaConf.create(cfg))
    assert os.path.exists(patch_h5_path)


