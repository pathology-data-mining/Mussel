"""Multi-backend WSI opening with automatic backend selection.

Provides :func:`open_slide` as a drop-in replacement for
``tiffslide.open_slide`` / ``openslide.open_slide`` that transparently
selects the best available backend for each slide path.

Backend priority (first available wins):

1. **CuCIM** (``cucim.clara.CuImage``) — NVIDIA GPU-accelerated reader for
   SVS/TIFF/NDPI.  Fastest on GPU nodes; requires ``pip install cucim``.
   Activated when ``MUSSEL_WSI_BACKEND=cucim`` or when cucim is installed
   and the file is a supported format.

2. **OME-Zarr** (``zarr`` + ``ome-zarr``) — Cloud-native chunked format for
   ``.zarr`` directories/stores.  Requires ``pip install zarr ome-zarr``.
   Always selected for ``.zarr`` paths.

3. **tiffslide** (default) — Wraps OpenSlide; works on all standard formats.

Environment variables:

* ``MUSSEL_WSI_BACKEND`` — Force a specific backend: ``tiffslide``,
  ``cucim``, or ``zarr``.

Only tiffslide is a hard dependency.  CuCIM and OME-Zarr are used only when
installed and explicitly requested or auto-detected.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Formats that CuCIM can handle natively.
_CUCIM_EXTENSIONS = frozenset({
    ".svs", ".tif", ".tiff", ".ndpi", ".scn", ".mrxs", ".bif", ".qptiff",
})

# OME-Zarr is always used for .zarr paths.
_ZARR_EXTENSIONS = frozenset({".zarr"})


def open_slide(path: str, backend: Optional[str] = None) -> Any:
    """Open a whole-slide image with the best available backend.

    Returns an object that is API-compatible with ``tiffslide.TiffSlide``
    (``read_region``, ``level_dimensions``, ``level_downsamples``,
    ``get_best_level_for_downsample``, ``properties``).

    Args:
        path: Path to slide file or ``.zarr`` directory.
        backend: Force a specific backend — ``"tiffslide"``, ``"cucim"``,
            or ``"zarr"``.  If ``None``, auto-detected.

    Returns:
        An open slide object.
    """
    if backend is None:
        backend = os.environ.get("MUSSEL_WSI_BACKEND", "auto").lower()

    ext = "".join(Path(path).suffixes).lower()

    if backend == "zarr" or (backend == "auto" and ext in _ZARR_EXTENSIONS):
        return _open_zarr(path)

    if backend == "cucim":
        return _open_cucim(path)

    if backend == "auto" and ext in _CUCIM_EXTENSIONS:
        cucim_slide = _try_cucim(path)
        if cucim_slide is not None:
            return cucim_slide

    return _open_tiffslide(path)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def _open_tiffslide(path: str) -> Any:
    import tiffslide as openslide

    return openslide.open_slide(path)


def _open_cucim(path: str) -> "CuCIMSlide":
    try:
        from cucim import CuImage
    except ImportError as e:
        raise ImportError(
            "cucim is required for the cucim backend: pip install cucim"
        ) from e
    return CuCIMSlide(CuImage(path))


def _try_cucim(path: str) -> Optional["CuCIMSlide"]:
    try:
        from cucim import CuImage

        return CuCIMSlide(CuImage(path))
    except Exception:
        return None


def _open_zarr(path: str) -> "OmeZarrSlide":
    try:
        import zarr
    except ImportError as e:
        raise ImportError(
            "zarr is required for OME-Zarr support: pip install zarr ome-zarr"
        ) from e
    return OmeZarrSlide(zarr.open(path, mode="r"))


# ---------------------------------------------------------------------------
# CuCIM wrapper — presents tiffslide-compatible API
# ---------------------------------------------------------------------------


class CuCIMSlide:
    """Thin wrapper around ``cucim.CuImage`` with tiffslide-compatible API."""

    def __init__(self, cuimage: Any) -> None:
        self._img = cuimage
        res = cuimage.resolutions
        self.level_dimensions: list = [
            (int(w), int(h)) for w, h in res["level_dimensions"]
        ]
        self.level_downsamples: list = [
            float(d) for d in res["level_downsamples"]
        ]
        # Build a properties-like dict from metadata
        self.properties: dict = dict(cuimage.metadata) if cuimage.metadata else {}

    def read_region(self, location, level: int, size) -> Any:
        """Return a PIL Image for the requested region.

        Args:
            location: (x, y) top-left corner in level-0 coordinates.
            level: Pyramid level.
            size: (width, height) of the region at the requested level.
        """
        from PIL import Image as PILImage

        x, y = int(location[0]), int(location[1])
        w, h = int(size[0]), int(size[1])
        region = self._img.read_region(location=(x, y), size=(w, h), level=level)
        # CuCIM returns a numpy array (H, W, C) or similar; convert to PIL.
        import numpy as np

        arr = np.asarray(region)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return PILImage.fromarray(arr, mode="RGBA")
        return PILImage.fromarray(arr, mode="RGB")

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Return the largest level with downsample ≤ the requested value."""
        best = 0
        for i, ds in enumerate(self.level_downsamples):
            if ds <= downsample:
                best = i
        return best

    def close(self) -> None:
        try:
            self._img.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# OME-Zarr wrapper — presents tiffslide-compatible API
# ---------------------------------------------------------------------------


class OmeZarrSlide:
    """Thin wrapper around an OME-Zarr store with tiffslide-compatible API.

    Supports OME-Zarr v0.4 multiscale arrays stored as a single series.
    The first ``multiscales`` entry is used.
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._arrays, self._downsamples = self._parse_multiscales(store)
        self.level_dimensions: list = [
            (int(arr.shape[-1]), int(arr.shape[-2])) for arr in self._arrays
        ]
        self.level_downsamples: list = self._downsamples
        self.properties: dict = dict(store.attrs) if hasattr(store, "attrs") else {}

    @staticmethod
    def _parse_multiscales(store):
        """Extract sorted pyramid arrays and their downsamples."""
        import numpy as np

        attrs = dict(store.attrs)
        multiscales = attrs.get("multiscales", [{}])[0]
        datasets = multiscales.get("datasets", [])

        arrays, downsamples = [], []
        base_shape = None
        for ds in datasets:
            path = ds.get("path", "0")
            arr = store[path]
            if base_shape is None:
                base_shape = arr.shape
            h0, w0 = base_shape[-2], base_shape[-1]
            h, w = arr.shape[-2], arr.shape[-1]
            downsamples.append(float(h0) / float(h) if h > 0 else 1.0)
            arrays.append(arr)

        if not arrays:
            # Fallback: treat numeric keys as level indices
            for key in sorted(store.keys(), key=lambda k: int(k) if k.isdigit() else 0):
                arr = store[key]
                if arr.ndim >= 2:
                    if base_shape is None:
                        base_shape = arr.shape
                    h0, w0 = base_shape[-2], base_shape[-1]
                    h = arr.shape[-2]
                    downsamples.append(float(h0) / float(h) if h > 0 else 1.0)
                    arrays.append(arr)

        return arrays, downsamples

    def read_region(self, location, level: int, size) -> Any:
        """Return a PIL Image for the requested region."""
        from PIL import Image as PILImage

        import numpy as np

        x, y = int(location[0]), int(location[1])
        w, h = int(size[0]), int(size[1])

        # Adjust coordinates to this level.
        ds = self.level_downsamples[level]
        xl = int(round(x / ds))
        yl = int(round(y / ds))

        arr = self._arrays[level]
        # Shape: (... C, H, W) or (... H, W, C) — handle both.
        ndim = arr.ndim
        if ndim == 4:
            # (T/Z, C, H, W) or (C, Z, H, W) — assume (C, Z, H, W) or (1, C, H, W)
            region = np.asarray(arr[0, :, yl : yl + h, xl : xl + w])
            region = np.transpose(region, (1, 2, 0))  # (H, W, C)
        elif ndim == 3:
            # Could be (C, H, W) or (H, W, C)
            if arr.shape[0] <= 4:
                region = np.asarray(arr[:, yl : yl + h, xl : xl + w])
                region = np.transpose(region, (1, 2, 0))
            else:
                region = np.asarray(arr[yl : yl + h, xl : xl + w, :])
        else:
            region = np.asarray(arr[yl : yl + h, xl : xl + w])

        region = region[:, :, :3].astype(np.uint8) if region.ndim == 3 else region.astype(np.uint8)
        return PILImage.fromarray(region, mode="RGB")

    def get_best_level_for_downsample(self, downsample: float) -> int:
        best = 0
        for i, ds in enumerate(self.level_downsamples):
            if ds <= downsample:
                best = i
        return best

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
