import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import h5py
import hydra
import numpy as np
import torch
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore

from mussel.utils.feature_extract import aggregate_sample_features
from mussel.utils.file import save_hdf5, save_torch_tensor

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
        output_dir (str): Directory where output files are written.
        output_h5_suffix (str): Filename suffix for H5 output files (default "features.h5").
            Each sample writes to "{output_dir}/{sample_id}.{output_h5_suffix}".
        output_pt_suffix (str): Filename suffix for PT output files (default "features.pt").
            Only used when save_pt=True.
        save_pt (bool): Whether to also save a PyTorch .pt tensor alongside the H5
            (default True).
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
    output_pt_suffix: str = "features.pt"
    save_pt: bool = True
    max_tiles: Optional[int] = None
    subsampling_strategy: str = "random"
    seed: int = 42


desc_doc = """== ${hydra.help.app_name} ==

Concatenate per-slide patch-level feature embeddings into per-sample feature files.

This tool reads HDF5 feature files produced by extract_features (one per slide),
groups them by sample_id, concatenates all tiles on the tile axis, optionally
subsamples to a max_tiles budget, and writes one output H5 and one output PT
tensor per unique sample.  To write only the H5 (no PT file), pass save_pt=false.

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

  # H5-only output (no .pt file):
  aggregate_sample_features \\
      'patch_features_h5_paths=[slide1.h5,slide2.h5]' \\
      'sample_ids=[P001,P001]' \\
      output_dir=/results/samples \\
      save_pt=false
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

    patch_features_h5_paths = list(cfg.patch_features_h5_paths)
    sample_ids = list(cfg.sample_ids)

    if len(patch_features_h5_paths) != len(sample_ids):
        raise ValueError(
            f"patch_features_h5_paths ({len(patch_features_h5_paths)}) and "
            f"sample_ids ({len(sample_ids)}) must have the same length."
        )

    # Group indices by sample_id first so we only hold one sample's slides in
    # memory at a time, keeping peak memory proportional to the largest sample
    # rather than the entire input set.
    groups: dict[str, list[int]] = {}
    for idx, sid in enumerate(sample_ids):
        groups.setdefault(sid, []).append(idx)

    os.makedirs(cfg.output_dir, exist_ok=True)
    for sample_id, indices in groups.items():
        features_list = []
        coords_list = []
        for i in indices:
            with h5py.File(patch_features_h5_paths[i], "r") as h5:
                features_list.append(np.array(h5["features"]))
                coords_list.append(h5["coords"][:])

        result = aggregate_sample_features(
            features_list=features_list,
            coords_list=coords_list,
            sample_ids=[sample_id] * len(indices),
            max_tiles=cfg.max_tiles,
            subsampling_strategy=cfg.subsampling_strategy,
            seed=cfg.seed,
        )

        features, coords = result[sample_id]
        out_h5 = os.path.join(cfg.output_dir, f"{sample_id}.{cfg.output_h5_suffix}")
        save_hdf5(out_h5, {"features": features, "coords": coords}, mode="w")
        logger.info(
            "Wrote %s (%d tiles, dim=%d)", out_h5, len(features), features.shape[1]
        )

        out_pt = os.path.join(cfg.output_dir, f"{sample_id}.{cfg.output_pt_suffix}")
        if cfg.save_pt:
            save_torch_tensor(out_pt, torch.from_numpy(features))
            logger.info("Wrote %s", out_pt)
        else:
            out_pt_path = Path(out_pt)
            if out_pt_path.exists():
                out_pt_path.unlink()
                logger.info("Removed stale PT output %s because save_pt=False", out_pt)

    logger.info("Done.")


if __name__ == "__main__":
    main()
