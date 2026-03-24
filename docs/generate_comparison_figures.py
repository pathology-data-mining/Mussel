"""Generate tissue-mask and patch-grid comparison figures.

Produces four images saved to docs/:
  comparison_tissue_mask.png  — side-by-side tissue mask overlays
  comparison_patch_grid.png   — side-by-side patch grid overlays
  comparison_overlap.png      — side-by-side with overlap=64
  comparison_mtp.png          — side-by-side with min_tissue_proportion=0.5

Run from the repo root:
    uv run python docs/generate_comparison_figures.py
"""

from __future__ import annotations
import os, sys, tempfile, textwrap, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from PIL import Image
import tiffslide

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from mussel.utils.segment import segment_tissue, contours_to_polygon
from mussel.utils.segment import scale_geometry

WSI_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "testdata", "948176.svs")
WSI_PATH = os.path.abspath(WSI_PATH)
DOCS_DIR = os.path.dirname(__file__)
REF_DIR  = "/gpfs/cdsi_ess/home/limr/ess/repos/wsi-reference-pipeline"
REF_PY   = os.path.join(REF_DIR, "venv", "bin", "python")

THUMB_LEVEL = 3   # ~32× downsample; fast to read
ALPHA_MASK  = 0.45
ALPHA_GRID  = 0.55
MUSSEL_COLOR = "#1f77b4"  # blue
REF_COLOR    = "#ff7f0e"  # orange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_thumbnail(wsi, level: int) -> np.ndarray:
    w, h = wsi.level_dimensions[level]
    img = np.array(wsi.read_region((0, 0), level, (w, h)))
    return img[..., :3]   # drop alpha if present


def _run_ref(tmpdir, overlap=0, mtp=0.0):
    """Run reference pipeline, return coords array."""
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {REF_DIR!r})
        from trident import load_wsi
        from trident.segmentation_models import segmentation_model_factory
        import h5py, numpy as np, json

        wsi = load_wsi({WSI_PATH!r})
        seg = segmentation_model_factory('otsu')
        wsi.segment_tissue(seg, target_mag=10, job_dir={tmpdir!r}, device='cpu', verbose=False)
        h5 = wsi.extract_tissue_coords(
            target_mag=20, patch_size=256, save_coords={tmpdir!r},
            overlap={overlap}, min_tissue_proportion={mtp},
        )
        with h5py.File(h5) as f:
            coords = f['coords'][:]
        print(json.dumps(coords.tolist()))
    """)
    res = subprocess.run([REF_PY, "-c", script], capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        raise RuntimeError(res.stderr)
    import json
    return np.array(json.loads(res.stdout.strip().splitlines()[-1]), dtype=np.int64)


def _run_mussel(overlap=0, mtp=0.0):
    """Run Mussel segment_tissue, return (polygon, coords)."""
    td = tempfile.mkdtemp()
    out = os.path.join(td, "mussel.h5")
    result = segment_tissue(
        slide_path=WSI_PATH,
        patch_size=256,
        mpp=0.5,
        use_otsu=False,
        tissue_area_threshold=1,
        output_h5_path=out,
        overlap=overlap,
        min_tissue_proportion=mtp,
    )
    polygon, grid, coords, attrs = result
    return polygon, np.array(coords, dtype=np.int64)


def _draw_mask(ax, thumb: np.ndarray, polygon, downsample: float, color: str, title: str, n_patches: int):
    ax.imshow(thumb)
    ax.set_title(f"{title}\n({n_patches:,} patches)", fontsize=10)
    ax.axis("off")
    if polygon is None:
        return
    from shapely.geometry import MultiPolygon
    scale = 1.0 / downsample
    scaled = scale_geometry(polygon, scale)
    geoms = list(scaled.geoms) if isinstance(scaled, MultiPolygon) else [scaled]
    for geom in geoms:
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, alpha=ALPHA_MASK, fc=color, ec=color, lw=1.5)
        for interior in geom.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, alpha=1.0, fc="white", ec="white", lw=0)


def _draw_grid(ax, thumb: np.ndarray, coords: np.ndarray, patch_size_px: int,
               downsample: float, color: str, title: str):
    ax.imshow(thumb)
    ax.set_title(f"{title}\n({len(coords):,} patches)", fontsize=10)
    ax.axis("off")
    s = patch_size_px / downsample
    for x, y in coords:
        rect = Rectangle(
            (x / downsample, y / downsample), s, s,
            linewidth=0.3, edgecolor=color, facecolor=color, alpha=0.25,
        )
        ax.add_patch(rect)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Opening WSI …")
    wsi = tiffslide.open_slide(WSI_PATH)
    thumb = _get_thumbnail(wsi, THUMB_LEVEL)
    downsample = wsi.level_downsamples[THUMB_LEVEL]
    wsi.close()
    print(f"Thumbnail: {thumb.shape[1]}×{thumb.shape[0]} px  (downsample={downsample:.1f}×)")

    # --- Run both pipelines for all conditions ---
    print("Running Mussel (baseline) …")
    m_poly_base, m_coords_base = _run_mussel(overlap=0, mtp=0.0)

    print("Running reference pipeline (baseline) …")
    td = tempfile.mkdtemp()
    r_coords_base = _run_ref(td, overlap=0, mtp=0.0)

    print("Running Mussel (overlap=64) …")
    _, m_coords_ov = _run_mussel(overlap=64, mtp=0.0)

    print("Running reference (overlap=64) …")
    td2 = tempfile.mkdtemp()
    r_coords_ov = _run_ref(td2, overlap=64, mtp=0.0)

    print("Running Mussel (mtp=0.5) …")
    _, m_coords_mtp = _run_mussel(overlap=0, mtp=0.5)

    print("Running reference (mtp=0.5) …")
    td3 = tempfile.mkdtemp()
    r_coords_mtp = _run_ref(td3, overlap=0, mtp=0.5)

    # -----------------------------------------------------------------------
    # Figure 1: Tissue mask comparison (baseline)
    # -----------------------------------------------------------------------
    print("Drawing Figure 1: tissue mask …")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="white")
    fig.suptitle("Tissue Mask — Mussel (HSV) vs Reference Pipeline (Otsu)\n"
                 "Slide 948176.svs · 256 px patches · 0.5 MPP · no overlap", fontsize=12)

    _draw_mask(axes[0], thumb, m_poly_base, downsample,
               MUSSEL_COLOR, "Mussel (HSV segmentation)", len(m_coords_base))

    # Reference: no polygon available — draw coord heat-map instead
    axes[1].imshow(thumb)
    axes[1].set_title(f"Reference pipeline (Otsu segmentation)\n({len(r_coords_base):,} patches)", fontsize=10)
    axes[1].axis("off")
    s = 256 / downsample
    for x, y in r_coords_base:
        rect = Rectangle((x / downsample, y / downsample), s, s,
                          lw=0, facecolor=REF_COLOR, alpha=0.35)
        axes[1].add_patch(rect)

    for ax, color, label in zip(axes, [MUSSEL_COLOR, REF_COLOR], ["Mussel", "Reference"]):
        ax.legend(handles=[mpatches.Patch(color=color, label=label)], loc="lower right", fontsize=9)

    plt.tight_layout()
    out1 = os.path.join(DOCS_DIR, "comparison_tissue_mask.png")
    fig.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out1}")

    # -----------------------------------------------------------------------
    # Figure 2: Patch grid (baseline)
    # -----------------------------------------------------------------------
    print("Drawing Figure 2: patch grid …")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="white")
    fig.suptitle("Patch Grid — Mussel vs Reference Pipeline\n"
                 "Slide 948176.svs · 256 px patches · 0.5 MPP · no overlap", fontsize=12)

    _draw_grid(axes[0], thumb, m_coords_base, 255, downsample,
               MUSSEL_COLOR, f"Mussel (HSV) — {len(m_coords_base):,} patches")
    _draw_grid(axes[1], thumb, r_coords_base, 256, downsample,
               REF_COLOR,    f"Reference (Otsu) — {len(r_coords_base):,} patches")

    plt.tight_layout()
    out2 = os.path.join(DOCS_DIR, "comparison_patch_grid.png")
    fig.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out2}")

    # -----------------------------------------------------------------------
    # Figure 3: Overlap=64
    # -----------------------------------------------------------------------
    print("Drawing Figure 3: overlap=64 …")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="white")
    fig.suptitle("Patch Grid with Overlap=64 px — Mussel vs Reference\n"
                 "Slide 948176.svs · 256 px patches · step=192 px", fontsize=12)

    _draw_grid(axes[0], thumb, m_coords_ov, 255, downsample,
               MUSSEL_COLOR, f"Mussel — {len(m_coords_ov):,} patches")
    _draw_grid(axes[1], thumb, r_coords_ov, 256, downsample,
               REF_COLOR,    f"Reference — {len(r_coords_ov):,} patches")

    plt.tight_layout()
    out3 = os.path.join(DOCS_DIR, "comparison_overlap.png")
    fig.savefig(out3, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out3}")

    # -----------------------------------------------------------------------
    # Figure 4: min_tissue_proportion=0.5
    # -----------------------------------------------------------------------
    print("Drawing Figure 4: min_tissue_proportion=0.5 …")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="white")
    fig.suptitle("Patch Grid with min_tissue_proportion=0.5 — Mussel vs Reference\n"
                 "Only patches with ≥50% tissue area retained", fontsize=12)

    _draw_grid(axes[0], thumb, m_coords_mtp, 255, downsample,
               MUSSEL_COLOR, f"Mussel — {len(m_coords_mtp):,} patches")
    _draw_grid(axes[1], thumb, r_coords_mtp, 256, downsample,
               REF_COLOR,    f"Reference — {len(r_coords_mtp):,} patches")

    plt.tight_layout()
    out4 = os.path.join(DOCS_DIR, "comparison_mtp.png")
    fig.savefig(out4, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out4}")

    # -----------------------------------------------------------------------
    # Figure 5: Combined summary (2×2)
    # -----------------------------------------------------------------------
    print("Drawing Figure 5: combined 2×2 summary …")
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), facecolor="white")
    fig.suptitle("Mussel vs Reference Pipeline — Tessellation Comparison\n"
                 "Slide 948176.svs (85,656×19,917 px, 0.5026 MPP)", fontsize=13)

    panels = [
        (axes[0,0], thumb, m_coords_base, 255, MUSSEL_COLOR,
         f"Mussel · HSV seg · overlap=0\n{len(m_coords_base):,} patches"),
        (axes[0,1], thumb, r_coords_base, 256, REF_COLOR,
         f"Reference · Otsu seg · overlap=0\n{len(r_coords_base):,} patches"),
        (axes[1,0], thumb, m_coords_ov,   255, MUSSEL_COLOR,
         f"Mussel · HSV seg · overlap=64 px\n{len(m_coords_ov):,} patches"),
        (axes[1,1], thumb, r_coords_ov,   256, REF_COLOR,
         f"Reference · Otsu seg · overlap=64 px\n{len(r_coords_ov):,} patches"),
    ]
    for ax, th, coords, ps, color, title in panels:
        _draw_grid(ax, th, coords, ps, downsample, color, title)

    plt.tight_layout()
    out5 = os.path.join(DOCS_DIR, "comparison_summary.png")
    fig.savefig(out5, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out5}")

    print("\nDone. Images saved to docs/")


if __name__ == "__main__":
    main()
