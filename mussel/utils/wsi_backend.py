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

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tiffslide
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# Formats that CuCIM can handle natively.
_CUCIM_EXTENSIONS = frozenset(
    {
        ".svs",
        ".tif",
        ".tiff",
        ".ndpi",
        ".scn",
        ".mrxs",
        ".bif",
        ".qptiff",
    }
)

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
    return tiffslide.open_slide(path)


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
# Shared base for tiffslide-compatible wrappers
# ---------------------------------------------------------------------------


class _SlideBase:
    """Mixin providing shared tiffslide-compatible API for backend wrappers."""

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Return the largest level whose downsample factor is ≤ ``downsample``."""
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


# ---------------------------------------------------------------------------
# CuCIM wrapper — presents tiffslide-compatible API
# ---------------------------------------------------------------------------

_MICROMETER_UNITS = frozenset({"micrometer", "micrometers", "micron", "microns", "um", "µm", "μm"})


def _flatten_metadata(metadata: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested metadata to ``vendor.key`` names (``{"aperio": {"MPP": x}}`` -> ``aperio.MPP``)."""
    flat: Dict[str, Any] = {}
    for key, value in metadata.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_metadata(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def _cucim_properties(metadata: Any) -> Dict[str, Any]:
    """tiffslide-style properties from ``CuImage.metadata``.

    cuCIM nests vendor metadata by vendor (``{"aperio": {"MPP": ...}, "cucim":
    {"spacing": [...], "spacing_units": [...], "dims": "YXC"}}``), so the
    ``aperio.MPP`` / ``tiffslide.mpp-x`` lookups in
    :func:`mussel.utils.segment.get_slide_mpp` missed it and fell back to the
    default MPP. The metadata is flattened to ``vendor.key`` names, and cuCIM's
    pixel spacing, when in micrometres, is exposed as ``tiffslide.mpp-x`` /
    ``tiffslide.mpp-y``.
    """
    if not metadata:
        return {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            return {}
    if not isinstance(metadata, dict):
        metadata = dict(metadata)
    props = _flatten_metadata(metadata)

    cucim_meta = metadata.get("cucim")
    if isinstance(cucim_meta, dict):
        spacing = cucim_meta.get("spacing") or []
        units = cucim_meta.get("spacing_units") or []
        dims = str(cucim_meta.get("dims") or "YXC").upper()
        for axis, key in (("X", "tiffslide.mpp-x"), ("Y", "tiffslide.mpp-y")):
            i = dims.find(axis)
            if key in props or i < 0 or i >= len(spacing) or i >= len(units):
                continue
            if str(units[i]).strip().lower() not in _MICROMETER_UNITS:
                continue
            try:
                value = float(spacing[i])
            except (TypeError, ValueError):
                continue
            if value > 0:
                props[key] = value
    return props



class CuCIMSlide(_SlideBase):
    """Thin wrapper around ``cucim.CuImage`` with tiffslide-compatible API."""

    def __init__(self, cuimage: Any) -> None:
        self._img = cuimage
        res = cuimage.resolutions
        self.level_dimensions: list = [
            (int(w), int(h)) for w, h in res["level_dimensions"]
        ]
        self.level_downsamples: list = [float(d) for d in res["level_downsamples"]]
        # Build a tiffslide-like flat properties dict from cuCIM's nested metadata
        self.properties: dict = _cucim_properties(cuimage.metadata)

    def read_region(self, location, level: int, size) -> Any:
        """Return a PIL Image for the requested region.

        Args:
            location: (x, y) top-left corner in level-0 coordinates.
            level: Pyramid level.
            size: (width, height) of the region at the requested level.
        """
        x, y = int(location[0]), int(location[1])
        w, h = int(size[0]), int(size[1])
        region = self._img.read_region(location=(x, y), size=(w, h), level=level)
        # CuCIM returns a numpy array (H, W, C) or similar; convert to PIL.
        arr = np.asarray(region)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return PILImage.fromarray(arr, mode="RGBA")
        return PILImage.fromarray(arr, mode="RGB")

    def close(self) -> None:
        try:
            self._img.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OME-Zarr wrapper — presents tiffslide-compatible API
# ---------------------------------------------------------------------------


class OmeZarrSlide(_SlideBase):
    """Thin wrapper around an OME-Zarr store with tiffslide-compatible API.

    Supports OME-Zarr multiscale arrays stored as a single series.
    The first ``multiscales`` entry is used.

    The y / x / channel axes come from the ``multiscales[0].axes`` metadata
    (NGFF v0.3 names or v0.4 ``{"name": ...}`` dicts) when present, so
    ``(T, C, Z, Y, X)`` and channel-last arrays are read correctly. Without axes
    metadata the layout is guessed from the shape: 2-D ``(Y, X)``; 3-D
    ``(C, Y, X)`` when the first dimension is <= 4, else ``(Y, X, C)``; 4-D
    ``(T, C, Y, X)``; 5-D ``(T, C, Z, Y, X)``. Other axes (T, Z) are read at index 0.
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._axes_names = self._axes_from_metadata(store)
        self._arrays, self._downsamples = self._parse_multiscales(store, self._axes_names)
        self._axes: List[Tuple[int, int, Optional[int]]] = [
            self._resolve_axes(arr, self._axes_names) for arr in self._arrays
        ]
        self.level_dimensions: list = [
            (int(arr.shape[xi]), int(arr.shape[yi]))
            for arr, (yi, xi, _) in zip(self._arrays, self._axes)
        ]
        self.level_downsamples: list = self._downsamples
        self.properties: dict = dict(store.attrs) if hasattr(store, "attrs") else {}

    @staticmethod
    def _axes_from_metadata(store) -> Optional[List[str]]:
        """Lower-cased axis names from ``multiscales[0].axes``, or None if absent."""
        try:
            attrs = dict(store.attrs)
            axes = attrs.get("multiscales", [{}])[0].get("axes")
        except Exception:
            return None
        if not axes:
            return None
        names = []
        for ax in axes:
            name = ax.get("name") if isinstance(ax, dict) else ax
            if not isinstance(name, str):
                return None
            names.append(name.lower())
        return names

    @staticmethod
    def _resolve_axes(arr, axes_names: Optional[List[str]]) -> Tuple[int, int, Optional[int]]:
        """(y, x, channel-or-None) axis indices of ``arr``."""
        ndim = len(arr.shape)
        if axes_names is not None and len(axes_names) == ndim and "y" in axes_names and "x" in axes_names:
            c = axes_names.index("c") if "c" in axes_names else None
            return axes_names.index("y"), axes_names.index("x"), c
        # Heuristic fallback for stores without (usable) axes metadata.
        if ndim == 2:
            return 0, 1, None
        if ndim == 3:
            if arr.shape[0] <= 4:
                return 1, 2, 0  # (C, Y, X)
            return 0, 1, 2  # (Y, X, C)
        if ndim == 4:
            return 2, 3, 1  # (T, C, Y, X)
        # NGFF order (T, C, Z, Y, X) for 5-D; generally (..., C, ..., Y, X).
        return ndim - 2, ndim - 1, ndim - 4 if ndim >= 5 else None

    @classmethod
    def _parse_multiscales(cls, store, axes_names: Optional[List[str]] = None):
        """Extract sorted pyramid arrays and their downsamples."""
        attrs = dict(store.attrs)
        multiscales = attrs.get("multiscales", [{}])[0]
        datasets = multiscales.get("datasets", [])

        arrays, downsamples = [], []
        base_h = None
        for ds in datasets:
            path = ds.get("path", "0")
            arr = store[path]
            h = arr.shape[cls._resolve_axes(arr, axes_names)[0]]
            if base_h is None:
                base_h = h
            downsamples.append(float(base_h) / float(h) if h > 0 else 1.0)
            arrays.append(arr)

        if not arrays:
            # Fallback: treat numeric keys as level indices
            for key in sorted(store.keys(), key=lambda k: int(k) if k.isdigit() else 0):
                arr = store[key]
                if arr.ndim >= 2:
                    h = arr.shape[cls._resolve_axes(arr, axes_names)[0]]
                    if base_h is None:
                        base_h = h
                    downsamples.append(float(base_h) / float(h) if h > 0 else 1.0)
                    arrays.append(arr)

        return arrays, downsamples

    def read_region(self, location, level: int, size) -> Any:
        """Return an RGB PIL Image for the requested region.

        Like tiffslide, the image is always ``size``: area outside the slide is
        padded with zeros.
        """
        x, y = int(location[0]), int(location[1])
        w, h = int(size[0]), int(size[1])

        # Adjust coordinates to this level.
        ds = self.level_downsamples[level]
        xl = int(round(x / ds))
        yl = int(round(y / ds))

        arr = self._arrays[level]
        yi, xi, ci = self._axes[level]
        height, width = int(arr.shape[yi]), int(arr.shape[xi])
        y0, y1 = max(yl, 0), min(yl + h, height)
        x0, x1 = max(xl, 0), min(xl + w, width)

        out = np.zeros((h, w, 3), dtype=np.uint8)
        if y1 <= y0 or x1 <= x0:
            return PILImage.fromarray(out)

        index: list = []
        for axis in range(len(arr.shape)):
            if axis == yi:
                index.append(slice(y0, y1))
            elif axis == xi:
                index.append(slice(x0, x1))
            elif axis == ci:
                index.append(slice(None))
            else:
                index.append(0)  # first timepoint / z-plane
        region = np.asarray(arr[tuple(index)])
        # Remaining axes keep their relative order; move them to (Y, X[, C]).
        kept = [a for a in (yi, xi, ci) if a is not None]
        order = sorted(kept)
        region = np.transpose(region, [order.index(a) for a in kept])
        if region.ndim == 2:
            region = region[:, :, None]
        n_channels = region.shape[2]
        if n_channels >= 3:
            region = region[:, :, :3]
        else:
            region = np.repeat(region[:, :, :1], 3, axis=2)

        out[y0 - yl : y1 - yl, x0 - xl : x1 - xl] = region.astype(np.uint8)
        return PILImage.fromarray(out)  # (H, W, 3) uint8 -> RGB
