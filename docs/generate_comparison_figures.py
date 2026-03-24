"""Generate tissue-mask and patch-grid comparison figures.

Produces figures saved to docs/:
  comparison_overview.png      — full-slide overview at level 2 (16×)
  comparison_tissue_mask.png   — side-by-side tissue mask overlays (level 2)
  comparison_patch_grid.png    — side-by-side patch grids, 3 zoomed crops (level 1, 4×)
  comparison_overlap.png       — overlap=64 zoomed crop
  comparison_mtp.png           — min_tissue_proportion=0.5 zoomed crop
  comparison_summary.png       — combined summary

Run from the repo root:
    uv run python docs/generate_comparison_figures.py
"""

from __future__ import annotations
import os, sys, tempfile, textwrap, subprocess, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
import tiffslide

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from mussel.utils.segment import segment_tissue, scale_geometry

WSI_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "testdata", "948176.svs")
WSI_PATH = os.path.abspath(WSI_PATH)
DOCS_DIR = os.path.dirname(__file__)
REF_DIR  = "/gpfs/cdsi_ess/home/limr/ess/repos/wsi-reference-pipeline"
REF_PY   = os.path.join(REF_DIR, "venv", "bin", "python")

MUSSEL_COLOR = "#1f77b4"   # blue
REF_COLOR    = "#d62728"   # red
ALPHA_GRID   = 0.30
ALPHA_MASK   = 0.40

# Three representative horizontal crops (level-0 coords) that show dense tissue
# Each is ~12 000 × 14 000 px at level 0 → 3 000 × 3 500 at level 1 (4×)
CROPS = [
    dict(label="Left",   x0=1000,  x1=13000, y0=2000, y1=18000),
    dict(label="Centre", x0=37000, x1=49000, y0=2000, y1=18000),
    dict(label="Right",  x0=68000, x1=80000, y0=2000, y1=18000),
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _run_ref(overlap=0, mtp=0.0):
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {REF_DIR!r})
        from trident import load_wsi
        from trident.segmentation_models import segmentation_model_factory
        import tempfile, h5py, json, numpy as np
        wsi = load_wsi({WSI_PATH!r})
        seg = segmentation_model_factory('otsu')
        td = tempfile.mkdtemp()
        wsi.segment_tissue(seg, target_mag=10, job_dir=td, device='cpu', verbose=False)
        h5 = wsi.extract_tissue_coords(
            target_mag=20, patch_size=256, save_coords=td,
            overlap={overlap}, min_tissue_proportion={mtp},
        )
        with h5py.File(h5) as f:
            coords = f['coords'][:]
        print(json.dumps(coords.tolist()))
    """)
    res = subprocess.run([REF_PY, "-c", script], capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-2000:])
    return np.array(json.loads(res.stdout.strip().splitlines()[-1]), dtype=np.int64)


def _run_mussel(overlap=0, mtp=0.0):
    td = tempfile.mkdtemp()
    out = os.path.join(td, "mussel.h5")
    poly, grid, coords, attrs = segment_tissue(
        slide_path=WSI_PATH, patch_size=256, mpp=0.5,
        use_otsu=False, tissue_area_threshold=1,
        output_h5_path=out, overlap=overlap, min_tissue_proportion=mtp,
    )
    return poly, np.array(coords, dtype=np.int64)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _read_crop(wsi, level, x0, y0, w, h):
    """Read a region [x0,y0,w,h] (level-0 coords) at `level`."""
    ds = wsi.level_downsamples[level]
    region = wsi.read_region((x0, y0), level, (round(w / ds), round(h / ds)))
    return np.array(region)[..., :3]


def _patches_in_crop(coords, patch_size, x0, y0, x1, y1):
    """Return coords whose top-left corner falls inside the crop."""
    mask = (
        (coords[:, 0] >= x0) & (coords[:, 0] + patch_size <= x1) &
        (coords[:, 1] >= y0) & (coords[:, 1] + patch_size <= y1)
    )
    return coords[mask]


def _draw_patches(ax, img, coords, patch_size_l0, x0_l0, y0_l0, ds, color, alpha=ALPHA_GRID):
    ax.imshow(img)
    ax.axis("off")
    ps = patch_size_l0 / ds
    for cx, cy in coords:
        rx = (cx - x0_l0) / ds
        ry = (cy - y0_l0) / ds
        rect = Rectangle((rx, ry), ps, ps, lw=0.8,
                          edgecolor=color, facecolor=color, alpha=alpha)
        ax.add_patch(rect)


def _draw_mask_on_crop(ax, img, polygon, x0_l0, y0_l0, ds, color):
    from shapely.affinity import translate, scale as shp_scale
    from shapely.geometry import MultiPolygon
    ax.imshow(img)
    ax.axis("off")
    if polygon is None:
        return
    shifted = translate(polygon, xoff=-x0_l0, yoff=-y0_l0)
    shrunk  = shp_scale(shifted, xfact=1/ds, yfact=1/ds, origin=(0, 0))
    geoms = list(shrunk.geoms) if isinstance(shrunk, MultiPolygon) else [shrunk]
    for geom in geoms:
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, alpha=ALPHA_MASK, fc=color, ec=color, lw=1.2)
        for interior in geom.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, alpha=1.0, fc="white", ec="white", lw=0)


import tempfile


def main():
    print("Opening WSI …")
    wsi = tiffslide.open_slide(WSI_PATH)
    level1_ds  = wsi.level_downsamples[1]   # 4.0
    level2_ds  = wsi.level_downsamples[2]   # 16.0
    slide_w, slide_h = wsi.level_dimensions[0]
    print(f"Level 1 downsample={level1_ds}×  Level 2 downsample={level2_ds}×")

    # Full-slide overview thumbnail (level 2, 16×)
    ov_w, ov_h = wsi.level_dimensions[2]
    thumb_l2 = np.array(wsi.read_region((0, 0), 2, (ov_w, ov_h)))[..., :3]

    # Read the 3 crop images at level 1 (patches = 256/4 = 64px, clearly visible)
    crop_imgs = []
    for c in CROPS:
        w = c["x1"] - c["x0"]
        h = c["y1"] - c["y0"]
        crop_imgs.append(_read_crop(wsi, 1, c["x0"], c["y0"], w, h))
    wsi.close()

    # --- Run pipelines ---
    print("Running Mussel (baseline) …")
    m_poly_base, m_coords_base = _run_mussel(overlap=0, mtp=0.0)

    print("Running reference pipeline (baseline) …")
    r_coords_base = _run_ref(overlap=0, mtp=0.0)

    print("Running Mussel (overlap=64) …")
    _, m_coords_ov = _run_mussel(overlap=64, mtp=0.0)

    print("Running reference (overlap=64) …")
    r_coords_ov = _run_ref(overlap=64, mtp=0.0)

    print("Running Mussel (mtp=0.5) …")
    _, m_coords_mtp = _run_mussel(overlap=0, mtp=0.5)

    print("Running reference (mtp=0.5) …")
    r_coords_mtp = _run_ref(overlap=0, mtp=0.5)

    # -----------------------------------------------------------------------
    # Figure 1: Overview (level 2) — both grids overlaid on full slide
    # -----------------------------------------------------------------------
    print("Figure 1: overview …")
    fig, axes = plt.subplots(1, 2, figsize=(20, 5), facecolor="white")
    fig.suptitle(
        "Full-slide overview  |  948176.svs (85,656×19,917 px, 0.5 MPP)\n"
        "Level 2 (16× downsample) — each patch ≈ 16 px", fontsize=11,
    )
    for ax, coords, ps_l0, color, label in [
        (axes[0], m_coords_base, 255, MUSSEL_COLOR, f"Mussel (HSV) — {len(m_coords_base):,} patches"),
        (axes[1], r_coords_base, 256, REF_COLOR,    f"Reference (Otsu) — {len(r_coords_base):,} patches"),
    ]:
        ax.imshow(thumb_l2)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
        ps = ps_l0 / level2_ds
        for cx, cy in coords:
            rect = Rectangle(
                (cx / level2_ds, cy / level2_ds), ps, ps,
                lw=0.3, edgecolor=color, facecolor=color, alpha=0.35,
            )
            ax.add_patch(rect)
    plt.tight_layout()
    p = os.path.join(DOCS_DIR, "comparison_overview.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    # -----------------------------------------------------------------------
    # Figure 2: Tissue mask (level 1 crops, 3 regions)
    # -----------------------------------------------------------------------
    print("Figure 2: tissue mask (zoomed) …")
    fig, axes = plt.subplots(2, 3, figsize=(20, 9), facecolor="white")
    fig.suptitle(
        "Tissue Mask — Mussel (HSV, blue) vs Reference (Otsu, red)\n"
        "Three representative crops at 4× downsample (each patch = 64 px)", fontsize=11,
    )
    for col, (c, img) in enumerate(zip(CROPS, crop_imgs)):
        x0, y0 = c["x0"], c["y0"]
        _draw_mask_on_crop(axes[0, col], img, m_poly_base, x0, y0, level1_ds, MUSSEL_COLOR)
        axes[0, col].set_title(f"{c['label']} — Mussel ({len(m_coords_base):,} total)", fontsize=9)

        # Reference: shade covered area using patch rectangles
        r_crop = _patches_in_crop(r_coords_base, 256, x0, y0, c["x1"], c["y1"])
        axes[1, col].imshow(img)
        axes[1, col].axis("off")
        axes[1, col].set_title(f"{c['label']} — Reference ({len(r_coords_base):,} total)", fontsize=9)
        for cx, cy in r_crop:
            axes[1, col].add_patch(Rectangle(
                ((cx - x0) / level1_ds, (cy - y0) / level1_ds),
                256 / level1_ds, 256 / level1_ds,
                lw=0, facecolor=REF_COLOR, alpha=0.40,
            ))

    axes[0, 0].set_ylabel("Mussel", fontsize=10)
    axes[1, 0].set_ylabel("Reference", fontsize=10)
    plt.tight_layout()
    p = os.path.join(DOCS_DIR, "comparison_tissue_mask.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    # -----------------------------------------------------------------------
    # Figure 3: Patch grid side-by-side (zoomed, 3 crops)
    # -----------------------------------------------------------------------
    print("Figure 3: patch grid (zoomed) …")
    fig, axes = plt.subplots(2, 3, figsize=(20, 9), facecolor="white")
    fig.suptitle(
        "Patch Grid — Mussel (blue) vs Reference (red) | Baseline: 256 px patches, no overlap\n"
        "Three representative crops at 4× downsample (patch = 64 px)", fontsize=11,
    )
    for col, (c, img) in enumerate(zip(CROPS, crop_imgs)):
        x0, y0, x1, y1 = c["x0"], c["y0"], c["x1"], c["y1"]
        m_crop = _patches_in_crop(m_coords_base, 255, x0, y0, x1, y1)
        r_crop = _patches_in_crop(r_coords_base, 256, x0, y0, x1, y1)

        _draw_patches(axes[0, col], img, m_crop, 255, x0, y0, level1_ds, MUSSEL_COLOR)
        axes[0, col].set_title(f"{c['label']} — Mussel: {len(m_crop)} patches", fontsize=9)

        _draw_patches(axes[1, col], img, r_crop, 256, x0, y0, level1_ds, REF_COLOR)
        axes[1, col].set_title(f"{c['label']} — Reference: {len(r_crop)} patches", fontsize=9)

    axes[0, 0].set_ylabel("Mussel", fontsize=10)
    axes[1, 0].set_ylabel("Reference", fontsize=10)
    plt.tight_layout()
    p = os.path.join(DOCS_DIR, "comparison_patch_grid.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    # -----------------------------------------------------------------------
    # Figure 4: Overlap=64 (centre crop only, side-by-side)
    # -----------------------------------------------------------------------
    print("Figure 4: overlap=64 …")
    c, img = CROPS[1], crop_imgs[1]
    x0, y0, x1, y1 = c["x0"], c["y0"], c["x1"], c["y1"]
    m_crop = _patches_in_crop(m_coords_ov, 255, x0, y0, x1, y1)
    r_crop = _patches_in_crop(r_coords_ov, 256, x0, y0, x1, y1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")
    fig.suptitle(
        "Patch Grid with overlap=64 px (step=192 px) — Centre crop\n"
        "Mussel: {:,} total | Reference: {:,} total".format(len(m_coords_ov), len(r_coords_ov)),
        fontsize=11,
    )
    _draw_patches(axes[0], img, m_crop, 255, x0, y0, level1_ds, MUSSEL_COLOR)
    axes[0].set_title(f"Mussel (HSV) — {len(m_crop)} patches in crop", fontsize=10)
    _draw_patches(axes[1], img, r_crop, 256, x0, y0, level1_ds, REF_COLOR)
    axes[1].set_title(f"Reference (Otsu) — {len(r_crop)} patches in crop", fontsize=10)

    plt.tight_layout()
    p = os.path.join(DOCS_DIR, "comparison_overlap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    # -----------------------------------------------------------------------
    # Figure 5: min_tissue_proportion=0.5 (centre crop)
    # -----------------------------------------------------------------------
    print("Figure 5: min_tissue_proportion=0.5 …")
    m_crop_mtp = _patches_in_crop(m_coords_mtp, 255, x0, y0, x1, y1)
    r_crop_mtp = _patches_in_crop(r_coords_mtp, 256, x0, y0, x1, y1)
    m_crop_no  = _patches_in_crop(m_coords_base, 255, x0, y0, x1, y1)
    r_crop_no  = _patches_in_crop(r_coords_base, 256, x0, y0, x1, y1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")
    fig.suptitle(
        "Effect of min_tissue_proportion=0.5 — Centre crop\n"
        "Patches requiring <50% tissue area are removed", fontsize=11,
    )
    _draw_patches(axes[0, 0], img, m_crop_no,  255, x0, y0, level1_ds, MUSSEL_COLOR)
    axes[0, 0].set_title(f"Mussel — no filter ({len(m_crop_no)} patches)", fontsize=9)
    _draw_patches(axes[0, 1], img, r_crop_no,  256, x0, y0, level1_ds, REF_COLOR)
    axes[0, 1].set_title(f"Reference — no filter ({len(r_crop_no)} patches)", fontsize=9)
    _draw_patches(axes[1, 0], img, m_crop_mtp, 255, x0, y0, level1_ds, MUSSEL_COLOR)
    axes[1, 0].set_title(f"Mussel — mtp=0.5 ({len(m_crop_mtp)} patches)", fontsize=9)
    _draw_patches(axes[1, 1], img, r_crop_mtp, 256, x0, y0, level1_ds, REF_COLOR)
    axes[1, 1].set_title(f"Reference — mtp=0.5 ({len(r_crop_mtp)} patches)", fontsize=9)

    plt.tight_layout()
    p = os.path.join(DOCS_DIR, "comparison_mtp.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    # -----------------------------------------------------------------------
    # Figure 6: Side-by-side diff — which patches appear in one but not other
    # -----------------------------------------------------------------------
    print("Figure 6: differential (unique patches) …")
    c, img = CROPS[1], crop_imgs[1]
    x0, y0, x1, y1 = c["x0"], c["y0"], c["x1"], c["y1"]
    m_crop = _patches_in_crop(m_coords_base, 255, x0, y0, x1, y1)
    r_crop = _patches_in_crop(r_coords_base, 256, x0, y0, x1, y1)

    # Round ref coords to nearest native step to roughly align grids
    # (256 ref ↔ 255 mussel — diff by 1px, so snap to 255 grid for comparison)
    def _snap(coords, step):
        return set(map(tuple, (coords // step * step).tolist()))

    m_set = _snap(m_crop, 255)
    r_set = _snap(r_crop, 255)
    mussel_only  = np.array([c for c in m_crop if tuple((np.array(c)//255*255).tolist()) not in r_set], dtype=np.int64)
    ref_only     = np.array([c for c in r_crop if tuple((np.array(c)//255*255).tolist()) not in m_set], dtype=np.int64)
    shared_m     = np.array([c for c in m_crop if tuple((np.array(c)//255*255).tolist()) in r_set],     dtype=np.int64)

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        f"Centre crop — Shared patches (green), Mussel-only (blue), Reference-only (red)\n"
        f"Shared: {len(shared_m)} · Mussel-only: {len(mussel_only)} · Ref-only: {len(ref_only)}",
        fontsize=10,
    )
    ps_m = 255 / level1_ds
    ps_r = 256 / level1_ds
    for coords_arr, ps, color, a in [
        (shared_m,    ps_m, "#2ca02c", 0.30),   # green
        (mussel_only, ps_m, MUSSEL_COLOR, 0.50), # blue
        (ref_only,    ps_r, REF_COLOR,    0.50), # red
    ]:
        for cx, cy in coords_arr:
            ax.add_patch(Rectangle(
                ((cx - x0) / level1_ds, (cy - y0) / level1_ds), ps, ps,
                lw=0.8, edgecolor=color, facecolor=color, alpha=a,
            ))
    legend_handles = [
        mpatches.Patch(color="#2ca02c", label=f"Shared ({len(shared_m)})"),
        mpatches.Patch(color=MUSSEL_COLOR, label=f"Mussel-only ({len(mussel_only)})"),
        mpatches.Patch(color=REF_COLOR,    label=f"Reference-only ({len(ref_only)})"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    plt.tight_layout()
    p = os.path.join(DOCS_DIR, "comparison_diff.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    print("\nAll figures saved to docs/")


if __name__ == "__main__":
    main()
