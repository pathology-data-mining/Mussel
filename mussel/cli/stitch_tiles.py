from dataclasses import dataclass
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from mussel.utils.wsi import StitchCoords
from mussel.WholeSlideImage import WholeSlideImage


@dataclass
class StitchConfig:
    slide_path: str = MISSING
    h5_path: str = MISSING
    output_jpeg_path: str = MISSING

cs = ConfigStore.instance()
cs.store(name="stitch_config", node=StitchConfig)

@hydra.main(version_base=None, config_path=".", config_name="stitch_config")
def main(cfg: StitchConfig):
    slide = WholeSlideImage(cfg.slide_path)
    heatmap = StitchCoords(
        cfg.h5_path,
        slide,
        downscale=64,
        bg_color=(0, 0, 0),
        alpha=-1,
        draw_grid=False,
    )
    heatmap.save(cfg.output_jpeg_path)


if __name__ == "__main__":
    main()
