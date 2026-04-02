import logging
from dataclasses import dataclass, field
from typing import List, Optional

import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore

from mussel.utils.feature_extract import aggregate_sample_features

logger = logging.getLogger(__name__)


@dataclass
class AggregateSampleFeaturesConfig:
    """
    Configuration for aggregating per-slide patch features into per-sample features.

    Args:
        patch_features_h5_paths (List[str]): Paths to per-slide feature H5 files
            (produced by extract_features).  Must have the same length as sample_ids.
        sample_ids (List[str]): Sample identifier for each slide.  Slides sharing
            the same sample_id are concatenated into a single output file.
        output_dir (str): Directory where output H5 files are written.
        output_h5_suffix (str): Filename suffix for output files (default "features.h5").
            Each sample writes to "{output_dir}/{sample_id}.{output_h5_suffix}".
        max_tiles (Optional[int]): Maximum number of tiles per sample after concatenation.
            When the total tile count exceeds this, tiles are subsampled.
            None (default) keeps all tiles.
        subsampling_strategy (str): Strategy for subsampling when max_tiles is set.
            "random" (default) — uniformly sample max_tiles from the full pool.
            "proportional" — sample from each slide in proportion to its size.
            "equal" — sample an equal number of tiles from each slide.
        seed (int): Random seed for reproducibility (default 42).
    """

    patch_features_h5_paths: List[str] = field(default_factory=list)
    sample_ids: List[str] = field(default_factory=list)
    output_dir: str = ""
    output_h5_suffix: str = "features.h5"
    max_tiles: Optional[int] = None
    subsampling_strategy: str = "random"
    seed: int = 42


desc_doc = """== ${hydra.help.app_name} ==

Concatenate per-slide patch-level feature embeddings into per-sample feature files.

This tool reads HDF5 feature files produced by extract_features (one per slide),
groups them by sample_id, concatenates all tiles on the tile axis, optionally
subsamples to a max_tiles budget, and writes one output H5 per unique sample.

Subsampling strategies (when max_tiles is set):
  - random:        uniformly sample max_tiles from the full tile pool (default)
  - proportional:  each slide contributes tiles proportional to its tile count
  - equal:         each slide contributes an equal number of tiles

Example:
  aggregate_sample_features \\
      'patch_features_h5_paths=[slide1.h5,slide2.h5,slide3.h5]' \\
      'sample_ids=[P001,P001,P002]' \\
      output_dir=/results/samples \\
      max_tiles=10000 \\
      subsampling_strategy=proportional
"""

parameter_doc = f"""== Available Parameters ==
{AggregateSampleFeaturesConfig.__doc__}
"""

cs = ConfigStore.instance()
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=desc_doc, footer=parameter_doc)),
    provider="hydra",
)
cs.store(name="aggregate_sample_features_config", node=AggregateSampleFeaturesConfig)


@hydra.main(
    version_base=None,
    config_path=None,
    config_name="aggregate_sample_features_config",
)
def main(cfg: AggregateSampleFeaturesConfig):
    logger.info("Starting sample feature aggregation")
    logger.info("Slides:  %d", len(cfg.patch_features_h5_paths))
    logger.info("Samples: %d unique", len(set(cfg.sample_ids)))
    logger.info("Output:  %s", cfg.output_dir)
    if cfg.max_tiles:
        logger.info(
            "Subsampling to %d tiles per sample (strategy=%s, seed=%d)",
            cfg.max_tiles,
            cfg.subsampling_strategy,
            cfg.seed,
        )

    aggregate_sample_features(
        patch_features_h5_paths=list(cfg.patch_features_h5_paths),
        sample_ids=list(cfg.sample_ids),
        output_dir=cfg.output_dir,
        output_h5_suffix=cfg.output_h5_suffix,
        max_tiles=cfg.max_tiles,
        subsampling_strategy=cfg.subsampling_strategy,
        seed=cfg.seed,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
