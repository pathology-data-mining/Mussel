import os
from omegaconf import OmegaConf

import mussel.cli.tessellate
from mussel.cli.tessellate import TessellateConfig, SegConfig, PatchConfig

def test_tessellate(tmp_path):
    slide_path = "tests/testdata/948176.svs"
    patch_h5_path = tmp_path / "test.h5"
    stitch_path = tmp_path / "test.jpg"
    seg_config = SegConfig(segment_threshold=0)
    patch_config = PatchConfig(num_workers=1)
    cfg = TessellateConfig(
        slide_path=slide_path,
        output_h5_path=patch_h5_path,
        stitch_jpeg_path=stitch_path,
        seg_config=seg_config,
        patch_config=patch_config,
    )
    mussel.cli.tessellate.main(OmegaConf.create(cfg))
    assert os.path.exists(patch_h5_path)
    assert os.path.exists(stitch_path)


