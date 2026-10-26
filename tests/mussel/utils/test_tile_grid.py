"""Tile grids built from an existing tessellation (mussel.utils.tile_grid)."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from omegaconf import OmegaConf

from mussel.cli.derive_tile_grid import DeriveTileGridConfig
from mussel.cli.derive_tile_grid import main as derive_main
from mussel.utils.segment import segment_tissue
from mussel.utils.tile_grid import GRID_METHOD, derive_grid, derive_patch_h5

_SLIDE = str(Path(__file__).parent.parent.parent / "testdata" / "948176.svs")


def _write_source(path, coords, *, tile_px=224, native=448, mpp=0.5, slide_mpp=0.25, level=0):
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("coords", data=np.asarray(coords, dtype=np.int64).reshape(-1, 2))
        ds.attrs.update(dict(patch_size=native, patch_level=level, patch_size_to_resize_to_for_desired_mpp=tile_px,
                             mpp=mpp, native_mpp=slide_mpp, mpp_is_fallback=False, name="src", seg_model="neural"))


def _read(path):
    with h5py.File(path, "r") as f:
        return f["coords"][:], dict(f["coords"].attrs)


def test_cells_are_those_containing_a_source_tile_centre():
    # 224 px tiles at 0.5 µm/px on a 0.25 µm/px slide are 448 px at level 0; 512 px cells are 1024.
    coords = np.array([[0, 0], [448, 0], [896, 0], [0, 1344], [3000, 3000]])
    grid, native = derive_grid(coords, 448, 512, 0.5, 0.25)
    assert native == 1024
    # centres: (224,224) (672,224) (1120,224) (224,1568) (3224,3224)
    assert grid.tolist() == [[0, 0], [0, 1024], [1024, 0], [3072, 3072]]


def test_matches_an_independent_cell_assignment():
    rng = np.random.default_rng(0)
    coords = rng.integers(0, 50, size=(500, 2)) * 425  # a 425 px level-0 grid, as on a 0.2635 µm/px slide
    grid, native = derive_grid(coords, 425, 512, 0.5, 0.2635)
    assert native == round(512 * 0.5 / 0.2635)
    expected = sorted({(int((x + 212.5) // native) * native, int((y + 212.5) // native) * native) for x, y in coords})
    assert [tuple(c) for c in grid.tolist()] == expected


def test_no_source_tiles_gives_an_empty_grid():
    grid, native = derive_grid(np.zeros((0, 2)), 448, 512, 0.5, 0.25)
    assert grid.shape == (0, 2) and native == 1024


def test_derived_h5_reads_like_a_tessellation(tmp_path):
    _write_source(tmp_path / "s.patch.h5", [[0, 0], [448, 448]])
    summary = derive_patch_h5(tmp_path / "s.patch.h5", tmp_path / "s.512px.patch.h5", 512, slide_id="S1")
    coords, attrs = _read(tmp_path / "s.512px.patch.h5")
    assert summary == {"source_tiles": 2, "tiles": 1, "patch_size": 1024}
    assert coords.tolist() == [[0, 0]]
    assert attrs["patch_size"] == 1024 and attrs["step_size"] == 1024
    assert attrs["patch_size_to_resize_to_for_desired_mpp"] == 512
    assert attrs["mpp"] == 0.5 and attrs["patch_level"] == 0 and attrs["native_mpp"] == 0.25
    assert attrs["name"] == "S1" and attrs["seg_model"] == "neural"  # source attrs carried over
    assert attrs["derived_grid_method"] == GRID_METHOD and attrs["derived_source_tiles"] == 2


@pytest.mark.parametrize("kwargs, message", [
    (dict(level=1), "patch_level"),
    (dict(tile_px=1024), "must not be smaller"),
    (dict(slide_mpp=0.0), "native_mpp"),
])
def test_unusable_sources_are_refused(tmp_path, kwargs, message):
    _write_source(tmp_path / "s.patch.h5", [[0, 0]], **kwargs)
    with pytest.raises((ValueError, ZeroDivisionError), match=message):
        derive_patch_h5(tmp_path / "s.patch.h5", tmp_path / "out.h5", 512)


def test_real_slide_every_source_tile_lies_in_a_derived_cell(tmp_path):
    segment_tissue(slide_path=_SLIDE, patch_size=224, output_h5_path=str(tmp_path / "t.patch.h5"),
                   seg_model="classic", segment_threshold=15, median_blur_ksize=11, morphology_ex_kernel=2,
                   tissue_area_threshold=1, hole_area_threshold=1, max_num_holes=2, mpp=0.5)
    derive_patch_h5(tmp_path / "t.patch.h5", tmp_path / "t.512px.patch.h5", 512)
    src, sattrs = _read(tmp_path / "t.patch.h5")
    grid, attrs = _read(tmp_path / "t.512px.patch.h5")
    n = attrs["patch_size"]
    cells = {tuple(c) for c in grid.tolist()}
    centres = {(int((x + sattrs["patch_size"] / 2) // n) * n, int((y + sattrs["patch_size"] / 2) // n) * n)
               for x, y in src.tolist()}
    assert cells == centres
    assert 0 < len(grid) < len(src)


def test_cli_single_and_batch(tmp_path):
    for name in ("a", "b"):
        _write_source(tmp_path / f"{name}.patch.h5", [[0, 0], [2048, 0]])
    derive_main(OmegaConf.create(DeriveTileGridConfig(
        source_h5_path=str(tmp_path / "a.patch.h5"), output_h5_path=str(tmp_path / "a.512px.patch.h5"),
        slide_id="a", patch_size=512)))
    assert _read(tmp_path / "a.512px.patch.h5")[0].tolist() == [[0, 0], [2048, 0]]
    derive_main(OmegaConf.create(DeriveTileGridConfig(
        source_h5_paths=[str(tmp_path / "b.patch.h5"), str(tmp_path / "missing.patch.h5")],
        output_h5_paths=[str(tmp_path / "b.512px.patch.h5"), str(tmp_path / "missing.512px.patch.h5")],
        patch_size=512, skip_missing=True)))
    assert (tmp_path / "b.512px.patch.h5").exists() and not (tmp_path / "missing.512px.patch.h5").exists()


def test_titan_subsample_is_the_same_in_every_process():
    # Python's hash() of a string differs between processes (PYTHONHASHSEED); the
    # max_slide_patches subset must not.
    import subprocess
    import sys

    code = ("import json; from mussel.utils.feature_extract import slide_subsample_indices as f; "
            "print(json.dumps(f(1000, 10, '948176.patch_features').tolist()))")
    runs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**__import__("os").environ, "PYTHONHASHSEED": seed}).stdout
            for seed in ("1", "2", "3")}
    assert len(runs) == 1
    indices = json.loads(runs.pop())
    assert len(indices) == 10 and indices == sorted(indices) and len(set(indices)) == 10
