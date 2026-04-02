"""AnyToTiffConverter — convert exotic slide formats to pyramidal TIFF.

Converts DICOM, LIF, VSI, OME-TIFF, CZI, ZVI, NRRD, and flat images to
pyramidal GeoTIFF using pyvips (fast streaming path) with an aicsimageio
fallback for formats that pyvips cannot decode.

Optional dependencies (install as needed):
    pip install pyvips
    pip install "aicsimageio[bioformats]"   # requires Java for Bio-Formats
    pip install pylibCZIrw                  # for CZI (Zeiss)
"""

import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

# Extensions handled by Bio-Formats via aicsimageio.
BIOFORMAT_EXTENSIONS = frozenset({
    ".tif", ".tiff", ".ndpi", ".svs", ".lif", ".ims", ".vsi", ".bif", ".btf",
    ".mrxs", ".scn", ".ome.tiff", ".ome.tif", ".h5", ".hdf", ".hdf5", ".he5",
    ".dicom", ".dcm", ".ome.xml", ".zvi", ".pcoraw", ".jp2", ".qptiff",
    ".nrrd", ".ome.btf", ".fg7",
})

PIL_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})
CZI_EXTENSIONS = frozenset({".czi"})

SUPPORTED_EXTENSIONS = BIOFORMAT_EXTENSIONS | PIL_EXTENSIONS | CZI_EXTENSIONS


def _splitext(path: str):
    """Split path into (stem, ext), handling compound extensions like .ome.tiff."""
    basename = os.path.basename(path)
    for compound in (".ome.tiff", ".ome.tif", ".ome.btf"):
        if basename.lower().endswith(compound):
            return basename[: -len(compound)], compound
    return os.path.splitext(basename)


def _process_file_worker(args) -> None:
    """Top-level worker function for multiprocessing (must be picklable)."""
    job_dir, bigtiff, input_file, mpp, zoom = args
    converter = AnyToTiffConverter(job_dir=job_dir, bigtiff=bigtiff)
    converter.process_file(input_file=input_file, mpp=mpp, zoom=zoom)


class AnyToTiffConverter:
    """Convert slides of any supported format to pyramidal GeoTIFF.

    Args:
        job_dir: Output directory for converted TIFF files.
        bigtiff: Write BigTIFF (required for files > 4 GB).
    """

    def __init__(self, job_dir: str, bigtiff: bool = False) -> None:
        self.job_dir = job_dir
        self.bigtiff = bigtiff
        self._detected_mpp: Dict[str, float] = {}
        os.makedirs(job_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_file(self, input_file: str, mpp: float, zoom: float = 1.0) -> None:
        """Convert a single file to pyramidal TIFF.

        Args:
            input_file: Path to the source slide or image.
            mpp: Target microns-per-pixel.  Used as-is unless override is applied.
            zoom: Resize factor (1.0 = original size, 0.5 = half resolution).
        """
        try:
            embedded_mpp = self._detect_embedded_mpp(input_file)
            if embedded_mpp is not None:
                self._detected_mpp[input_file] = embedded_mpp
                if abs(embedded_mpp - mpp) > 1e-3:
                    logger.warning(
                        "MPP mismatch for %s: CSV mpp=%.6f, embedded mpp=%.6f. "
                        "Using CSV value.",
                        os.path.basename(input_file),
                        mpp,
                        embedded_mpp,
                    )

            stem, _ = _splitext(input_file)
            output_mpp = mpp * (1.0 / zoom)
            save_path = os.path.join(self.job_dir, f"{stem}.tiff")

            if self._try_pyvips_convert(input_file, save_path, zoom, output_mpp):
                return

            img = self._read_image(input_file, zoom)
            self._save_tiff(img, stem, output_mpp)
        except Exception:
            logger.exception("Error processing %s", input_file)

    def process_all(
        self,
        input_dir: str,
        mpp_csv: str,
        downscale_by: int = 1,
        num_workers: int = 1,
    ) -> None:
        """Convert all images listed in a CSV to pyramidal TIFF.

        Args:
            input_dir: Directory containing source images.
            mpp_csv: CSV with columns ``wsi`` (filename with extension) and
                ``mpp`` (microns-per-pixel).
            downscale_by: Integer downsample factor (≥1).  ``2`` halves resolution.
            num_workers: Worker processes.  ``0`` uses all available CPUs.
        """
        import pandas as pd
        from tqdm import tqdm

        if downscale_by < 1:
            raise ValueError(f"downscale_by must be ≥ 1, got {downscale_by}.")
        if num_workers < 0:
            raise ValueError(f"num_workers must be ≥ 0, got {num_workers}.")
        if not os.path.isfile(mpp_csv):
            raise ValueError(f"MPP CSV not found: {mpp_csv}.")

        df = pd.read_csv(mpp_csv)
        missing = {"wsi", "mpp"} - set(df.columns)
        if missing:
            raise ValueError(f"MPP CSV is missing columns: {sorted(missing)}")
        if df.empty:
            raise ValueError("MPP CSV is empty.")

        df = df.dropna(subset=["wsi", "mpp"]).copy()
        df["wsi"] = df["wsi"].astype(str)

        tasks, skipped_missing, skipped_unsupported = [], [], []
        for filename in df["wsi"].tolist():
            img_path = os.path.join(input_dir, filename)
            if not os.path.exists(img_path):
                skipped_missing.append(filename)
                continue
            ext = "".join(Path(filename).suffixes).lower()
            if ext not in SUPPORTED_EXTENSIONS:
                skipped_unsupported.append(filename)
                continue
            mpp = float(df.loc[df["wsi"] == filename, "mpp"].values[0])
            tasks.append((self.job_dir, self.bigtiff, img_path, mpp, 1.0 / downscale_by))

        if skipped_missing:
            logger.warning("Skipping %d files not found in input_dir.", len(skipped_missing))
        if skipped_unsupported:
            logger.warning(
                "Skipping %d files with unsupported extension.", len(skipped_unsupported)
            )
        if not tasks:
            raise ValueError("No valid conversion tasks found from CSV entries.")

        if num_workers == 0:
            num_workers = mp.cpu_count()

        if num_workers <= 1:
            for _, _, img_path, mpp, zoom in tqdm(tasks, desc="Converting slides"):
                self.process_file(img_path, mpp, zoom=zoom)
        else:
            with mp.Pool(processes=num_workers) as pool:
                for _ in tqdm(
                    pool.imap_unordered(_process_file_worker, tasks),
                    total=len(tasks),
                    desc="Converting slides",
                ):
                    pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_embedded_mpp(self, file_path: str) -> Optional[float]:
        """Try to read embedded MPP from metadata without full decode."""
        if file_path in self._detected_mpp:
            return self._detected_mpp[file_path]
        mpp = self._detect_mpp_pyvips(file_path)
        if mpp is None:
            mpp = self._detect_mpp_aicsimageio(file_path)
        return mpp

    def _detect_mpp_pyvips(self, file_path: str) -> Optional[float]:
        try:
            import pyvips
        except ImportError:
            return None
        try:
            img = pyvips.Image.new_from_file(file_path, access="sequential")
            if img.get_typeof("xres") == 0:
                return None
            xres = float(img.get("xres"))
            return (1000.0 / xres) if xres > 0 else None
        except Exception:
            return None

    def _detect_mpp_aicsimageio(self, file_path: str) -> Optional[float]:
        ext = "".join(Path(file_path).suffixes).lower()
        if ext not in BIOFORMAT_EXTENSIONS:
            return None
        try:
            from aicsimageio import AICSImage
        except ImportError:
            return None
        try:
            px = AICSImage(file_path).physical_pixel_sizes
            return float(px.X) if px and px.X is not None else None
        except Exception:
            return None

    def _try_pyvips_convert(
        self, input_file: str, save_path: str, zoom: float, mpp: float
    ) -> bool:
        """Attempt fast streaming conversion via pyvips; returns True on success."""
        if input_file.lower().endswith(".czi"):
            return False
        try:
            import pyvips
        except ImportError:
            return False
        try:
            img = pyvips.Image.new_from_file(input_file, access="sequential")
            if zoom != 1.0:
                img = img.resize(zoom)
            self._save_pyvips_tiff(img, save_path, mpp, pyvips)
            return True
        except Exception:
            return False

    def _read_image(self, file_path: str, zoom: float = 1.0) -> np.ndarray:
        """Decode image to a numpy RGB array, applying zoom."""
        if file_path.lower().endswith(".czi"):
            try:
                import pylibCZIrw.czi as pyczi
            except ImportError as e:
                raise ImportError(
                    "pylibCZIrw is required for CZI files: pip install pylibCZIrw"
                ) from e
            with pyczi.open_czi(file_path) as czidoc:
                return czidoc.read(zoom=zoom)

        ext = "".join(Path(file_path).suffixes).lower()
        if ext in BIOFORMAT_EXTENSIONS:
            try:
                from aicsimageio import AICSImage
            except ImportError as e:
                raise ImportError(
                    "aicsimageio is required for this format: "
                    "pip install 'aicsimageio[bioformats]'"
                ) from e
            img_obj = AICSImage(file_path)
            czyx = img_obj.get_image_data("CZYX", T=0)
            if czyx.ndim != 4:
                raise ValueError(f"Unexpected image shape: {czyx.shape}")
            first_z = czyx[:, 0, :, :]
            data = first_z[0] if first_z.shape[0] == 1 else np.transpose(first_z[:3], (1, 2, 0))
            if zoom != 1.0:
                pil_img = Image.fromarray(data)
                new_w = int(pil_img.width * zoom)
                new_h = int(pil_img.height * zoom)
                data = np.array(pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS))
            px = img_obj.physical_pixel_sizes
            if px and px.X is not None:
                self._detected_mpp[file_path] = float(px.X)
            return data

        with Image.open(file_path) as img:
            new_w = int(img.width * zoom)
            new_h = int(img.height * zoom)
            return np.array(img.resize((new_w, new_h), Image.Resampling.LANCZOS))

    def _save_tiff(self, img: np.ndarray, stem: str, mpp: float) -> None:
        try:
            import pyvips
        except ImportError as e:
            raise ImportError(
                "pyvips is required for saving pyramidal TIFFs: pip install pyvips"
            ) from e
        save_path = os.path.join(self.job_dir, f"{stem}.tiff")
        pyvips_img = pyvips.Image.new_from_array(img)
        self._save_pyvips_tiff(pyvips_img, save_path, mpp, pyvips)

    def _save_pyvips_tiff(self, img, save_path: str, mpp: float, pyvips_module) -> None:
        img.tiffsave(
            save_path,
            bigtiff=self.bigtiff,
            pyramid=True,
            tile=True,
            tile_width=256,
            tile_height=256,
            compression="jpeg",
            resunit=pyvips_module.enums.ForeignTiffResunit.CM,
            xres=1.0 / (mpp * 1e-4),
            yres=1.0 / (mpp * 1e-4),
        )
        logger.info("Saved pyramidal TIFF → %s", save_path)
