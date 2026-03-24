"""Generate tissue-mask and patch-grid comparison figures.

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
import tiffslide

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from mussel.utils.segment import segment_tissue

WSI_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "testdata", "948176.svs"))
DOCS_DIR = os.path.dirname(__file__)
REF_DIR  = "/gpfs/cdsi_ess/home/limr/ess/repos/wsi-reference-pipeline"
REF_PY   = os.path.join(REF_DIR, "venv", "bin", "python")

MUSSEL_COLOR = "#1f77b4"
REF_COLOR    = "#d62728"
SHARED_COLOR = "#2ca02c"

# Three representative crops in level-0 coords covering left/centre/right tissue
CROPS = [
    dict(label="Left",   x0=2000,  x1=18000, y0=2000, y1=18000),
    dict(label="Centre", x0=37000, x1=53000, y0=2000, y1=18000),
    dict(label="Right",  x0=64000, x1=80000, y0=2000, y1=18000),
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _run_ref(overlap=0, mtp=0.0):
    script = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {REF_DIR!r})
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


def _read_crop(wsi, level, x0, y0, x1, y1):
    ds = wsi.level_downsamples[level]
    w = round((x1 - x0) / ds)
    h = round((y1 - y0) / ds)
    region = wsi.read_region((x0, y0), level, (w, h))
    return np.array(region)[..., :3]


def _in_crop(coords, ps, x0, y0, x1, y1):
    """Coords whose patch overlaps the crop (top-left inside crop)."""
    m = (coords[:, 0] >= x0) & (coords[:, 0] < x1) & \
        (coords[:, 1] >= y0) & (coords[:, 1] < y1)
    return coords[m]


def _ax_patches(ax, img, coords, patch_size_l0, x0_l0, y0_l0, ds, color,
                alpha_face=0.30, alpha_edge=0.8, lw=0.8, title=""):
    """Draw slide image with patch rectangles overlaid; locks axis limits."""
    h_img, w_img = img.shape[:2]
    ax.imshow(img, extent=[0, w_img, h_img, 0])  # explicit extent → stable limits
    ax.set_xlim(0, w_img)
    ax.set_ylim(h_img, 0)
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    ps = patch_size_l0 / ds
    for cx, cy in coords:
        rx = (cx - x0_l0) / ds
        ry = (cy - y0_l0) / ds
        ax.add_patch(Rectangle(
            (rx, ry), ps, ps, lw=lw,
            edgecolor=color, facecolor=color, alpha=alpha_face,
        ))
        ax.add_patch(Rectangle(
            (rx, ry), ps, ps, lw=lw,
            edgecolor=color, facecolor="none", alpha=alpha_edge,
        ))


def _ax_mask(ax, img, polygon, x0_l0, y0_l0, x1_l0, y1_l0, ds, color, title=""):
    """Draw tissue polygon cropped to image bounds; locks axis limits."""
    from shapely.geometry import box, MultiPolygon
    from shapely.affinity import translate, scale as shp_scale

    h_img, w_img = img.shape[:2]
    ax.imshow(img, extent=[0, w_img, h_img, 0])
    ax.set_xlim(0, w_img)
    ax.set_ylim(h_img, 0)
    ax.axis("off")
    ax.set_title(title, fontsize=9)

    if polygon is None:
        return

    # Clip polygon to crop, then transform to image pixel space
    crop_box = box(x0_l0, y0_l0, x1_l0, y1_l0)
    clipped = polygon.intersection(crop_box)
    if clipped.is_empty:
        return

    shifted = translate(clipped, xoff=-x0_l0, yoff=-y0_l0)
    scaled  = shp_scale(shifted, xfact=1/ds, yfact=1/ds, origin=(0, 0, 0))

    geoms = list(scaled.geoms) if isinstance(scaled, MultiPolygon) else [scaled]
    for geom in geoms:
        if geom.is_empty:
            continue
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, alpha=0.45, fc=color, ec=color, lw=1.5)
        for interior in geom.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, alpha=1.0, fc="white", ec="none")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Opening WSI …")
    wsi = tiffslide.open_slide(WSI_PATH)
    ds1 = wsi.level_downsamples[1]   # 4×
    ds2 = wsi.level_downsamples[2]   # 16×
    ov_w, ov_h = wsi.level_dimensions[2]
    thumb_l2 = np.array(wsi.read_region((0, 0), 2, (ov_w, ov_h)))[..., :3]
    crop_imgs = [_read_crop(wsi, 1, c["x0"], c["y0"], c["x1"], c["y1"]) for c in CROPS]
    wsi.close()
    print(f"Levels: 1={ds1:.0f}× 2={ds2:.0f}×  |  crops: {[i.shape for i in crop_imgs]}")

    print("Running pipelines …")
    m_poly_base, m_coords_base = _run_mussel(0, 0.0)
    r_coords_base               = _run_ref(0, 0.0)
    _,            m_coords_ov   = _run_mussel(64, 0.0)
    r_coords_ov                 = _run_ref(64, 0.0)
    _,            m_coords_mtp  = _run_mussel(0, 0.5)
    r_coords_mtp                = _run_ref(0, 0.5)
    print(f"Baseline: Mussel={len(m_coords_base):,}  Ref={len(r_coords_base):,}")

    # -----------------------------------------------------------------------
    # Figure 1: Full-slide overview (level 2)
    # -----------------------------------------------------------------------
    print("Figure 1: overview …")
    fig, axes = plt.subplots(1, 2, figsize=(20, 5), facecolor="white")
    fig.suptitle(
        "Full-slide overview — 948176.svs (85,656×19,917 px, 0.5 MPP)\n"
        "Level 2 (16× downsample)", fontsize=11,
    )
    for ax, coords, ps, color, label in [
        (axes[0], m_coords_base, 255, MUSSEL_COLOR, f"Mussel (HSV) — {len(m_coords_base):,} patches"),
        (axes[1], r_coords_base, 256, REF_COLOR,    f"Reference (Otsu) — {len(r_coords_base):,} patches"),
    ]:
        _ax_patches(ax, thumb_l2, coords, ps, 0, 0, ds2, color, title=label, lw=0.2)
    plt.tight_layout()
    _save(fig, "comparison_overview.png")

    # -----------------------------------------------------------------------
    # Figure 2: Tissue mask — 3 crops × 2 pipelines
    # -----------------------------------------------------------------------
    print("Figure 2: tissue mask …")
    fig, axes = plt.subplots(2, 3, figsize=(21, 9), facecolor="white")
    fig.suptitle(
        "Tissue Mask — Mussel (HSV, blue) vs Reference (Otsu, red)\n"
        "Three crops at 4× downsample — each patch ≈ 64 px", fontsize=11,
    )
    for col, (c, img) in enumerate(zip(CROPS, crop_imgs)):
        x0, y0, x1, y1 = c["x0"], c["y0"], c["x1"], c["y1"]
        r_crop = _in_crop(r_coords_base, 256, x0, y0, x1, y1)
        n_m = len(_in_crop(m_coords_base, 255, x0, y0, x1, y1))
        n_r = len(r_crop)

        _ax_mask(axes[0, col], img, m_poly_base, x0, y0, x1, y1, ds1, MUSSEL_COLOR,
                 title=f"{c['label']} — Mussel ({n_m} patches in crop)")
        _ax_patches(axes[1, col], img, r_crop, 256, x0, y0, ds1, REF_COLOR,
                    alpha_face=0.40, title=f"{c['label']} — Reference ({n_r} patches in crop)")

    axes[0, 0].set_ylabel("Mussel", fontsize=10)
    axes[1, 0].set_ylabel("Reference", fontsize=10)
    plt.tight_layout()
    _save(fig, "comparison_tissue_mask.png")

    # -----------------------------------------------------------------------
    # Figure 3: Patch grid — 3 crops × 2 pipelines
    # -----------------------------------------------------------------------
    print("Figure 3: patch grid …")
    fig, axes = plt.subplots(2, 3, figsize=(21, 9), facecolor="white")
    fig.suptitle(
        "Patch Grid — Mussel (blue) vs Reference (red) — Baseline: 256 px, no overlap\n"
        "Three crops at 4× downsample — each patch ≈ 64 px", fontsize=11,
    )
    for col, (c, img) in enumerate(zip(CROPS, crop_imgs)):
        x0, y0, x1, y1 = c["x0"], c["y0"], c["x1"], c["y1"]
        m_crop = _in_crop(m_coords_base, 255, x0, y0, x1, y1)
        r_crop = _in_crop(r_coords_base, 256, x0, y0, x1, y1)
        _ax_patches(axes[0, col], img, m_crop, 255, x0, y0, ds1, MUSSEL_COLOR,
                    title=f"{c['label']} — Mussel: {len(m_crop)} patches")
        _ax_patches(axes[1, col], img, r_crop, 256, x0, y0, ds1, REF_COLOR,
                    title=f"{c['label']} — Reference: {len(r_crop)} patches")
    axes[0, 0].set_ylabel("Mussel", fontsize=10)
    axes[1, 0].set_ylabel("Reference", fontsize=10)
    plt.tight_layout()
    _save(fig, "comparison_patch_grid.png")

    # -----------------------------------------------------------------------
    # Figure 4: Overlap=64 — centre crop
    # -----------------------------------------------------------------------
    print("Figure 4: overlap …")
    c, img = CROPS[1], crop_imgs[1]
    x0, y0, x1, y1 = c["x0"], c["y0"], c["x1"], c["y1"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")
    fig.suptitle(
        f"overlap=64 px (step=192 px) — Centre crop\n"
        f"Mussel: {len(m_coords_ov):,} total  |  Reference: {len(r_coords_ov):,} total", fontsize=11,
    )
    _ax_patches(axes[0], img, _in_crop(m_coords_ov, 255, x0, y0, x1, y1),
                255, x0, y0, ds1, MUSSEL_COLOR,
                title=f"Mussel — {len(_in_crop(m_coords_ov,255,x0,y0,x1,y1))} patches in crop")
    _ax_patches(axes[1], img, _in_crop(r_coords_ov, 256, x0, y0, x1, y1),
                256, x0, y0, ds1, REF_COLOR,
                title=f"Reference — {len(_in_crop(r_coords_ov,256,x0,y0,x1,y1))} patches in crop")
    plt.tight_layout()
    _save(fig, "comparison_overlap.png")

    # -----------------------------------------------------------------------
    # Figure 5: min_tissue_proportion=0.5 — centre crop before/after
    # -----------------------------------------------------------------------
    print("Figure 5: mtp …")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")
    fig.suptitle(
        "Effect of min_tissue_proportion=0.5 — Centre crop\n"
        "Patches with <50% tissue area removed (bottom row)", fontsize=11,
    )
    pairs = [
        (axes[0,0], m_coords_base, 255, MUSSEL_COLOR, "Mussel — no filter"),
        (axes[0,1], r_coords_base, 256, REF_COLOR,    "Reference — no filter"),
        (axes[1,0], m_coords_mtp,  255, MUSSEL_COLOR, "Mussel — mtp=0.5"),
        (axes[1,1], r_coords_mtp,  256, REF_COLOR,    "Reference — mtp=0.5"),
    ]
    for ax, coords, ps, color, lbl in pairs:
        crop = _in_crop(coords, ps, x0, y0, x1, y1)
        _ax_patches(ax, img, crop, ps, x0, y0, ds1, color,
                    title=f"{lbl}: {len(crop)} patches in crop")
    plt.tight_layout()
    _save(fig, "comparison_mtp.png")

    # -----------------------------------------------------------------------
    # Figure 6: Differential — shared / Mussel-only / Ref-only
    # -----------------------------------------------------------------------
    print("Figure 6: differential …")
    m_crop = _in_crop(m_coords_base, 255, x0, y0, x1, y1)
    r_crop = _in_crop(r_coords_base, 256, x0, y0, x1, y1)

    # Snap both to a common 255-px grid for alignment comparison
    def snap(arr, step=255):
        return set(map(tuple, (arr // step * step).tolist()))

    m_set, r_set = snap(m_crop), snap(r_crop)
    shared    = np.array([c for c in m_crop if tuple((np.array(c)//255*255).tolist()) in r_set])
    mussel_only = np.array([c for c in m_crop if tuple((np.array(c)//255*255).tolist()) not in r_set])
    ref_only    = np.array([c for c in r_crop if tuple((np.array(c)//255*255).tolist()) not in m_set])

    fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")
    h_img, w_img = img.shape[:2]
    ax.imshow(img, extent=[0, w_img, h_img, 0])
    ax.set_xlim(0, w_img); ax.set_ylim(h_img, 0); ax.axis("off")
    ax.set_title(
        f"Centre crop — Shared (green): {len(shared)}  "
        f"Mussel-only (blue): {len(mussel_only)}  "
        f"Reference-only (red): {len(ref_only)}",
        fontsize=10,
    )
    for coords_arr, ps, color, af, ae in [
        (shared,      255, SHARED_COLOR,  0.25, 0.7),
        (mussel_only, 255, MUSSEL_COLOR,  0.45, 0.9),
        (ref_only,    256, REF_COLOR,     0.45, 0.9),
    ]:
        for cx, cy in coords_arr:
            ps_px = ps / ds1
            rx, ry = (cx - x0) / ds1, (cy - y0) / ds1
            ax.add_patch(Rectangle((rx, ry), ps_px, ps_px,
                                   lw=0.6, edgecolor=color, facecolor=color, alpha=af))
    ax.legend(handles=[
        mpatches.Patch(color=SHARED_COLOR,  label=f"Shared ({len(shared)})"),
        mpatches.Patch(color=MUSSEL_COLOR,  label=f"Mussel-only ({len(mussel_only)})"),
        mpatches.Patch(color=REF_COLOR,     label=f"Reference-only ({len(ref_only)})"),
    ], loc="lower right", fontsize=9)
    plt.tight_layout()
    _save(fig, "comparison_diff.png")

    print("\nDone.")


def _save(fig, name):
    p = os.path.join(DOCS_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
