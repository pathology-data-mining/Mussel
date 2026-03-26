"""Generate before/after pen mark removal figures for 1007867.svs.

Produces two images in docs/:
  penmark_overview.png     — full-slide overview with mask overlay
  penmark_crop.png         — three crops showing pen marks before/after removal

Usage:
    uv run python docs/generate_penmark_figures.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_SLIDE_PATH = Path.home() / ".cache" / "mussel-test-slides" / "1007867.svs"
_OUT_DIR = Path(__file__).parent
_DEVICE = "cuda"

# Pyramid level used for GrandQC input (~2 µm/px at 4× downsample)
_GRANDQC_LEVEL = 1
# Pyramid level used for the display overview (~8 µm/px at 16× downsample)
_DISPLAY_LEVEL = 2


def _tissue_mask_otsu(img_rgb: np.ndarray) -> np.ndarray:
    """Simple Otsu tissue mask to avoid running GrandQC on background."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(np.uint8)


def _get_mask(img: np.ndarray, tissue_mask: np.ndarray, img_mpp: float, device: str) -> np.ndarray:
    from mussel.utils import GrandQCArtifactRemover

    remover = GrandQCArtifactRemover(
        remove_penmarks_only=True,
        device=device,
        batch_size=8,
    )
    return remover(img, tissue_mask, img_mpp)


def _overlay(img_rgb: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.45) -> np.ndarray:
    out = img_rgb.copy().astype(np.float32)
    tint = np.zeros_like(out)
    tint[mask == 1] = color
    out = out * (1 - alpha) + tint * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _red_overlay(img_rgb: np.ndarray, removed: np.ndarray) -> np.ndarray:
    out = img_rgb.copy()
    out[removed == 1] = (
        out[removed == 1] * 0.4 + np.array([220, 30, 30]) * 0.6
    ).astype(np.uint8)
    return out


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def _find_crop_centers(removed: np.ndarray, tissue: np.ndarray,
                        crop_h: int, crop_w: int, n: int = 3) -> list[tuple[int, int]]:
    """Find n crop centres where removed AND tissue pixels overlap most densely."""
    h, w = removed.shape
    overlap = (removed * tissue).astype(np.float32)
    # Blur to find dense regions
    kernel_h = min(crop_h // 2, h // 4)
    kernel_w = min(crop_w // 2, w // 4)
    if kernel_h % 2 == 0:
        kernel_h += 1
    if kernel_w % 2 == 0:
        kernel_w += 1
    density = cv2.GaussianBlur(overlap, (kernel_w, kernel_h), 0)

    centres = []
    margin_y, margin_x = crop_h // 2, crop_w // 2
    # Mask out edges
    edge_mask = np.zeros_like(density)
    edge_mask[margin_y:h - margin_y, margin_x:w - margin_x] = 1
    density = density * edge_mask

    for _ in range(n):
        if density.max() == 0:
            break
        idx = int(np.argmax(density))
        cy, cx = divmod(idx, w)
        centres.append((cy, cx))
        # Suppress nearby peaks (radius = crop size)
        y0 = max(0, cy - crop_h)
        y1 = min(h, cy + crop_h)
        x0 = max(0, cx - crop_w)
        x1 = min(w, cx + crop_w)
        density[y0:y1, x0:x1] = 0

    return centres


def main() -> None:
    import tiffslide

    from mussel.utils.segment import get_slide_mpp

    print(f"Opening {_SLIDE_PATH} ...")
    with tiffslide.TiffSlide(str(_SLIDE_PATH)) as wsi:
        slide_mpp = get_slide_mpp(wsi, str(_SLIDE_PATH))

        grandqc_dims = wsi.level_dimensions[_GRANDQC_LEVEL]
        grandqc_ds = wsi.level_downsamples[_GRANDQC_LEVEL]
        grandqc_mpp = slide_mpp * grandqc_ds
        print(f"Reading level {_GRANDQC_LEVEL}: {grandqc_dims}, mpp={grandqc_mpp:.2f}")
        img_grandqc = np.array(
            wsi.read_region((0, 0), _GRANDQC_LEVEL, grandqc_dims)
        )[:, :, :3]

        disp_dims = wsi.level_dimensions[_DISPLAY_LEVEL]
        disp_ds = wsi.level_downsamples[_DISPLAY_LEVEL]
        print(f"Reading level {_DISPLAY_LEVEL}: {disp_dims} for display")
        img_disp = np.array(
            wsi.read_region((0, 0), _DISPLAY_LEVEL, disp_dims)
        )[:, :, :3]

    # Use Otsu tissue mask as input — avoids removing background (class 7)
    # and focuses GrandQC on pen marks within actual tissue.
    print("Computing Otsu tissue mask ...")
    tissue_mask = _tissue_mask_otsu(img_grandqc)
    tissue_pct = 100.0 * tissue_mask.sum() / tissue_mask.size
    print(f"Tissue area: {tissue_pct:.1f}% of slide")

    print("Running GrandQC (remove_penmarks_only=True) ...")
    mask_after = _get_mask(img_grandqc, tissue_mask, grandqc_mpp, _DEVICE)
    removed = ((tissue_mask - mask_after) > 0).astype(np.uint8)

    pct_removed = 100.0 * removed.sum() / max(tissue_mask.sum(), 1)
    print(f"Pen-marked tissue removed: {removed.sum():,} / {tissue_mask.sum():,} ({pct_removed:.1f}%)")

    # Scale masks to display level
    disp_h, disp_w = img_disp.shape[:2]
    tissue_disp = cv2.resize(tissue_mask, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    after_disp  = cv2.resize(mask_after,  (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    removed_disp = cv2.resize(removed,    (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

    # -----------------------------------------------------------------------
    # Figure 1: full-slide overview
    # -----------------------------------------------------------------------
    panel_before = _overlay(img_disp, tissue_disp, (0, 180, 0))
    panel_after  = _overlay(img_disp, after_disp,  (0, 180, 0))
    panel_diff   = _red_overlay(img_disp, removed_disp)

    panel_before = _label(panel_before, "Tissue mask (before)")
    panel_after  = _label(panel_after,  f"After pen mark removal ({pct_removed:.1f}% removed)")
    panel_diff   = _label(panel_diff,   "Removed regions (red)")

    # Scale down to reasonable output size
    max_w = 4500
    overview = np.hstack([panel_before, panel_after, panel_diff])
    if overview.shape[1] > max_w:
        scale = max_w / overview.shape[1]
        overview = cv2.resize(overview, (max_w, int(overview.shape[0] * scale)))

    out_path = _OUT_DIR / "penmark_overview.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(overview, cv2.COLOR_RGB2BGR))
    print(f"Saved {out_path}  ({overview.shape[1]}×{overview.shape[0]})")

    # -----------------------------------------------------------------------
    # Figure 2: three crops at GrandQC level centred on pen-mark regions
    # -----------------------------------------------------------------------
    crop_h, crop_w = 512, 768
    centres = _find_crop_centers(removed, tissue_mask, crop_h, crop_w, n=3)
    print(f"Crop centres (y, x): {centres}")

    if not centres:
        print("No crop centres found — skipping crop figure")
        return

    rows_out = []
    for cy, cx in centres:
        x0 = max(0, cx - crop_w // 2)
        y0 = max(0, cy - crop_h // 2)
        x1, y1 = x0 + crop_w, y0 + crop_h

        crop_img     = img_grandqc[y0:y1, x0:x1]
        crop_tissue  = tissue_mask[y0:y1, x0:x1]
        crop_after   = mask_after[y0:y1, x0:x1]
        crop_rem     = removed[y0:y1, x0:x1]

        p_before = _overlay(crop_img, crop_tissue, (0, 180, 0))
        p_after  = _overlay(crop_img, crop_after,  (0, 180, 0))
        p_diff   = _red_overlay(crop_img, crop_rem)

        rows_out.append(np.hstack([p_before, p_after, p_diff]))

    crop_fig = np.vstack(rows_out)
    out_path2 = _OUT_DIR / "penmark_crop.png"
    cv2.imwrite(str(out_path2), cv2.cvtColor(crop_fig, cv2.COLOR_RGB2BGR))
    print(f"Saved {out_path2}  ({crop_fig.shape[1]}×{crop_fig.shape[0]})")



if __name__ == "__main__":
    main()
