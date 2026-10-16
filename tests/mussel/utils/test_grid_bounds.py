"""Tile grid: anchored at the slide origin, and covering the whole tissue bounding box."""
from pathlib import Path

import h5py
import numpy as np
import pytest
from shapely.affinity import translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from mussel.utils.segment import grid_bounds, partition, segment_tissue

_SLIDE = str(Path(__file__).parent.parent.parent / "testdata" / "948176.svs")


def _origins(grid):
    return {(int(g.bounds[0]), int(g.bounds[1])) for g in grid}


# An irregular "tissue" outline away from the slide origin.
_TISSUE = Polygon([(1037, 2213), (5121, 2003), (6890, 4417), (4400, 7311), (1500, 6100)])


@pytest.mark.parametrize("step, patch", [(224, 224), (509, 509), (200, 224)])
def test_origins_lie_on_the_slide_origin_lattice(step, patch):
    origins = _origins(partition(_TISSUE, step, patch))
    assert origins and all(x % step == 0 and y % step == 0 for x, y in origins)


def test_a_pixel_of_tissue_moves_only_edge_tiles():
    """Shifting the outline by a few pixels (another GPU's mask) keeps the interior tiles."""
    a = _origins(partition(_TISSUE, 224, 224))
    b = _origins(partition(translate(_TISSUE, xoff=-4), 224, 224))
    assert len(a & b) > 0.9 * len(a)
    # Every tile that differs is on the tissue's edge: it isn't fully inside the outline.
    for x, y in a ^ b:
        assert not _TISSUE.contains(box(x, y, x + 224, y + 224))


@pytest.mark.parametrize("step, patch", [(224, 224), (200, 224)])
def test_the_whole_bounding_box_is_covered(step, patch):
    """No strip at the right or bottom edge is left untiled (the last row and column)."""
    grid = partition(_TISSUE, step, patch)
    assert _TISSUE.difference(unary_union(grid)).area == 0


def test_rectangle_edges():
    tissue = box(0, 0, 1000, 450)
    grid = grid_bounds(tissue, 224, 224)
    xs = sorted({int(g.bounds[0]) for g in grid})
    ys = sorted({int(g.bounds[1]) for g in grid})
    assert xs == [0, 224, 448, 672, 896]  # 896 covers 896-1120, i.e. tissue up to x=1000
    assert ys == [0, 224, 448]


def test_tiles_end_inside_the_slide():
    grid = grid_bounds(box(0, 0, 1000, 450), 224, 224, slide_size=(1000, 460))
    assert max(int(g.bounds[2]) for g in grid) <= 1000 and max(int(g.bounds[3]) for g in grid) <= 460
    assert sorted({int(g.bounds[0]) for g in grid}) == [0, 224, 448, 672]  # 896 + 224 > 1000


def test_origins_never_negative():
    grid = grid_bounds(box(10, 10, 300, 300), 200, 224)  # with overlap, a tile at -200 would overlap x=10
    assert min(int(g.bounds[0]) for g in grid) == 0 and min(int(g.bounds[1]) for g in grid) == 0


def test_real_slide_tiles_on_the_lattice_and_cover_the_tissue(tmp_path):
    out = tmp_path / "s.patch.h5"
    polygon, grid, coords, attrs = segment_tissue(
        slide_path=_SLIDE, output_h5_path=str(out), seg_model="classic", segment_threshold=15,
        median_blur_ksize=11, morphology_ex_kernel=2, tissue_area_threshold=1, hole_area_threshold=1,
        max_num_holes=2, mpp=0.5, patch_size=224)
    step, patch = int(attrs["step_size"]), int(attrs["patch_size"])
    assert all(int(x) % step == 0 and int(y) % step == 0 for x, y in coords)
    width, height = (int(v) for v in attrs["level_dim"])
    assert all(x + patch <= width and y + patch <= height for x, y in coords)
    # Tissue is left untiled only where a tile would cross the slide's border.
    inner = box(0, 0, (width - patch) // step * step + patch, (height - patch) // step * step + patch)
    assert polygon.intersection(inner).difference(unary_union(grid)).area < 1e-6 * polygon.area
    with h5py.File(out) as f:
        assert f["coords"].attrs["grid_anchor"] == "slide_origin"
    assert np.array_equal(np.asarray(coords, dtype=np.int64), h5py.File(out)["coords"][:])
