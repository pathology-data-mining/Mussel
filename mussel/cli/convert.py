"""CLI command to convert slides to pyramidal TIFF (``mussel convert``)."""

from dataclasses import dataclass, field
from typing import List, Optional

import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ConvertConfig:
    """
    Configuration for converting slide images to pyramidal TIFF.

    Args:
        input_path: Path to a single slide file OR a directory of slides for
            batch conversion.
        output_dir: Directory where converted TIFF files will be written.
        mpp: Microns-per-pixel of the source image.  Required for single-file
            mode.  Ignored in batch mode (MPP comes from ``mpp_csv``).
        mpp_csv: Path to a CSV with columns ``wsi`` (filename with extension)
            and ``mpp`` (microns-per-pixel).  Required for batch/directory mode.
        downscale_by: Integer downsample factor applied during conversion
            (default 1 = no downscaling).  ``2`` saves a 40x slide as 20x.
        num_workers: Number of parallel worker processes for batch mode.
            ``0`` uses all available CPUs; ``1`` is sequential.
        bigtiff: Write BigTIFF format (required for files larger than ~4 GB).
    """

    input_path: str = MISSING
    output_dir: str = MISSING
    mpp: Optional[float] = None
    mpp_csv: Optional[str] = None
    downscale_by: int = 1
    num_workers: int = 1
    bigtiff: bool = False


_desc = """== ${hydra.help.app_name} ==

Convert slide images of any supported format to pyramidal GeoTIFF.

Supported input formats: SVS, NDPI, SCN, MRXS, TIFF, LIF, VSI, CZI, OME-TIFF,
DICOM, ZVI, NRRD, PNG, JPEG, and more.  Requires pyvips and/or aicsimageio.

Single-file mode:
  convert input_path=slide.lif output_dir=./tiffs mpp=0.25

Batch mode (directory + CSV):
  convert input_path=./wsis output_dir=./tiffs mpp_csv=mpp.csv num_workers=4

CSV format (mpp.csv):
  wsi,mpp
  slide1.lif,0.25
  slide2.vsi,0.50
"""

cs = ConfigStore.instance()
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=_desc, footer=f"\n{ConvertConfig.__doc__}")),
    provider="hydra",
)
cs.store(name="convert_config", node=ConvertConfig)


@hydra.main(version_base=None, config_path=None, config_name="convert_config")
def main(cfg: ConvertConfig) -> None:
    """Convert slides to pyramidal TIFF.

    Examples::

        # Single file
        convert input_path=slide.lif output_dir=./tiffs mpp=0.25

        # Batch (directory + CSV)
        convert input_path=./wsis output_dir=./tiffs mpp_csv=mpp.csv num_workers=4

        # With downscaling (40x → 20x)
        convert input_path=./wsis output_dir=./tiffs mpp_csv=mpp.csv downscale_by=2
    """
    import os

    from mussel.utils.converter import AnyToTiffConverter

    converter = AnyToTiffConverter(job_dir=cfg.output_dir, bigtiff=cfg.bigtiff)

    if os.path.isfile(cfg.input_path):
        if cfg.mpp is None:
            raise ValueError(
                "mpp is required for single-file conversion. "
                "Pass mpp=<value> or use a directory + mpp_csv for batch mode."
            )
        converter.process_file(cfg.input_path, mpp=cfg.mpp)
    elif os.path.isdir(cfg.input_path):
        if cfg.mpp_csv is None:
            raise ValueError(
                "mpp_csv is required for batch/directory conversion. "
                "Pass mpp_csv=<path_to_csv> where CSV has columns: wsi,mpp"
            )
        converter.process_all(
            input_dir=cfg.input_path,
            mpp_csv=cfg.mpp_csv,
            downscale_by=cfg.downscale_by,
            num_workers=cfg.num_workers,
        )
    else:
        raise ValueError(f"input_path does not exist: {cfg.input_path}")


if __name__ == "__main__":
    main()
