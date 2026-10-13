"""Tiling one segmentation at several patch sizes (segment_tissue_multi).

Each size's output must be identical to tessellating that size on its own:
only contour filtering (thresholds are in patches), the grid and the tile
filters depend on the size; the tissue mask is shared.
"""
from pathlib import Path

import h5py
import numpy as np
import pytest
from omegaconf import OmegaConf

import mussel.cli.tessellate
from mussel.cli.tessellate import BiopsySegConfig, TessellateConfig
from mussel.utils.segment import segment_tissue, segment_tissue_multi

_SLIDE = str(Path(__file__).parent.parent.parent / "testdata" / "948176.svs")
_CLASSIC = dict(seg_model="classic", segment_threshold=15, median_blur_ksize=11,
                morphology_ex_kernel=2, tissue_area_threshold=1, hole_area_threshold=1,
                max_num_holes=2, mpp=0.5)


def _read(path):
    with h5py.File(path, "r") as f:
        return f["coords"][:], dict(f["coords"].attrs)


def test_each_size_matches_single_size_tessellation(tmp_path):
    sizes = [224, 512]
    multi = segment_tissue_multi(
        slide_path=_SLIDE,
        patch_sizes=sizes,
        output_h5_paths=[str(tmp_path / f"multi_{s}.h5") for s in sizes],
        **_CLASSIC,
    )
    for size in sizes:
        single = segment_tissue(
            slide_path=_SLIDE, patch_size=size,
            output_h5_path=str(tmp_path / f"single_{size}.h5"), **_CLASSIC,
        )
        assert multi[size] is not None and single is not None
        np.testing.assert_array_equal(np.asarray(multi[size][2]), np.asarray(single[2]))
        m_coords, m_attrs = _read(tmp_path / f"multi_{size}.h5")
        s_coords, s_attrs = _read(tmp_path / f"single_{size}.h5")
        np.testing.assert_array_equal(m_coords, s_coords)
        assert m_attrs.keys() == s_attrs.keys()
        for key in s_attrs:
            np.testing.assert_array_equal(m_attrs[key], s_attrs[key], err_msg=key)
        assert m_attrs["patch_size_to_resize_to_for_desired_mpp"] == size
    # Larger tiles: fewer of them, at a coarser level-0 spacing.
    assert len(multi[512][2]) < len(multi[224][2])


def test_segmentation_runs_once(tmp_path, monkeypatch):
    import mussel.utils.segment as seg

    calls = []
    real = seg.cv2.threshold
    monkeypatch.setattr(seg.cv2, "threshold", lambda *a, **k: calls.append(1) or real(*a, **k))
    segment_tissue_multi(slide_path=_SLIDE, patch_sizes=[224, 256, 512], **_CLASSIC)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(patch_sizes=[224, 224]), "distinct"),
        (dict(patch_sizes=[]), "at least one"),
        (dict(patch_sizes=[224, 512], output_h5_paths=["a.h5"]), "must match"),
        (dict(patch_sizes=[224, 512], step_size=200), "single patch size"),
    ],
)
def test_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        segment_tissue_multi(slide_path=_SLIDE, **{**_CLASSIC, **kwargs})


def _cfg(**kwargs):
    seg = BiopsySegConfig(seg_model="classic", patch_sizes=[224, 512])
    return OmegaConf.create(TessellateConfig(seg_config=seg, **kwargs))


def test_cli_single_slide_writes_one_h5_per_size(tmp_path):
    mussel.cli.tessellate.main(_cfg(
        slide_path=_SLIDE,
        output_h5_path=str(tmp_path / "tiles_{patch_size}" / "948176.patch.h5"),
    ))
    for size in (224, 512):
        coords, attrs = _read(tmp_path / f"tiles_{size}" / "948176.patch.h5")
        assert len(coords) > 0
        assert attrs["patch_size_to_resize_to_for_desired_mpp"] == size


def test_cli_batch_output_dir_names_files_by_size(tmp_path):
    mussel.cli.tessellate.main(_cfg(
        slide_paths=[_SLIDE], slide_ids=["S"], output_dir=str(tmp_path),
    ))
    assert (tmp_path / "S.224px.patch.h5").exists()
    assert (tmp_path / "S.512px.patch.h5").exists()


def test_cli_requires_size_placeholder(tmp_path):
    with pytest.raises(ValueError, match=r"\{patch_size\}"):
        mussel.cli.tessellate.main(_cfg(
            slide_path=_SLIDE, output_h5_path=str(tmp_path / "out.patch.h5"),
        ))


# Tile sets at different resolutions (patch_mpps), e.g. H-optimus-1 224 px at 0.5
# µm/px and CHIEF 256 px at 1.0, plus the same size at two resolutions.
_SPECS = [(224, 0.5), (256, 1.0), (224, 1.0)]


def test_each_resolution_matches_single_tessellation(tmp_path):
    multi = segment_tissue_multi(
        slide_path=_SLIDE,
        patch_sizes=[s for s, _ in _SPECS],
        patch_mpps=[m for _, m in _SPECS],
        output_h5_paths=[str(tmp_path / f"multi_{s}_{m}.h5") for s, m in _SPECS],
        **_CLASSIC,
    )
    assert set(multi) == set(_SPECS)  # keyed by (size, mpp)
    for size, mpp in _SPECS:
        single = segment_tissue(
            slide_path=_SLIDE, patch_size=size,
            output_h5_path=str(tmp_path / f"single_{size}_{mpp}.h5"), **{**_CLASSIC, "mpp": mpp},
        )
        m_coords, m_attrs = _read(tmp_path / f"multi_{size}_{mpp}.h5")
        s_coords, s_attrs = _read(tmp_path / f"single_{size}_{mpp}.h5")
        np.testing.assert_array_equal(m_coords, s_coords)
        for key in s_attrs:
            np.testing.assert_array_equal(m_attrs[key], s_attrs[key], err_msg=key)
        assert m_attrs["mpp"] == mpp and m_attrs["patch_size_to_resize_to_for_desired_mpp"] == size
    # 224 px at 1.0 µm/px covers twice the width of 224 px at 0.5: fewer, larger level-0 tiles.
    assert _read(tmp_path / "multi_224_1.0.h5")[1]["patch_size"] == pytest.approx(
        2 * _read(tmp_path / "multi_224_0.5.h5")[1]["patch_size"], abs=1)
    assert len(multi[(224, 1.0)][2]) < len(multi[(224, 0.5)][2])


def test_resolutions_share_one_segmentation(monkeypatch):
    import mussel.utils.segment as seg

    calls = []
    real = seg.cv2.threshold
    monkeypatch.setattr(seg.cv2, "threshold", lambda *a, **k: calls.append(1) or real(*a, **k))
    segment_tissue_multi(slide_path=_SLIDE, patch_sizes=[224, 256], patch_mpps=[0.5, 1.0], **_CLASSIC)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(patch_sizes=[224, 224], patch_mpps=[0.5, 0.5]), "distinct"),
        (dict(patch_sizes=[224, 256], patch_mpps=[0.5]), "must match"),
    ],
)
def test_invalid_resolutions(kwargs, match):
    with pytest.raises(ValueError, match=match):
        segment_tissue_multi(slide_path=_SLIDE, **{**_CLASSIC, **kwargs})


def _cfg_mpps(**kwargs):
    seg = BiopsySegConfig(seg_model="classic", patch_sizes=[224, 256, 224], patch_mpps=[0.5, 1.0, 1.0])
    return OmegaConf.create(TessellateConfig(seg_config=seg, **kwargs))


def test_cli_writes_one_h5_per_size_and_resolution(tmp_path):
    mussel.cli.tessellate.main(_cfg_mpps(
        slide_path=_SLIDE,
        output_h5_path=str(tmp_path / "S.{patch_size}px_{mpp}mpp.patch.h5"),
    ))
    for name, (size, mpp) in (("224px_0.5mpp", (224, 0.5)), ("256px_1mpp", (256, 1.0)), ("224px_1mpp", (224, 1.0))):
        coords, attrs = _read(tmp_path / f"S.{name}.patch.h5")
        assert len(coords) > 0 and attrs["mpp"] == mpp
        assert attrs["patch_size_to_resize_to_for_desired_mpp"] == size


def test_cli_batch_output_dir_names_files_by_size_and_resolution(tmp_path):
    mussel.cli.tessellate.main(_cfg_mpps(slide_paths=[_SLIDE], slide_ids=["S"], output_dir=str(tmp_path)))
    for name in ("224px_0.5mpp", "256px_1mpp", "224px_1mpp"):
        assert (tmp_path / f"S.{name}.patch.h5").exists()


def test_cli_requires_paths_that_tell_resolutions_apart(tmp_path):
    with pytest.raises(ValueError, match=r"\{mpp\}"):
        mussel.cli.tessellate.main(_cfg_mpps(
            slide_path=_SLIDE, output_h5_path=str(tmp_path / "S.{patch_size}px.patch.h5"),
        ))


def test_cli_rejects_mpps_without_sizes(tmp_path):
    seg = BiopsySegConfig(seg_model="classic", patch_mpps=[0.5])
    with pytest.raises(ValueError, match="patch_mpps needs seg_config.patch_sizes"):
        mussel.cli.tessellate.main(OmegaConf.create(TessellateConfig(
            seg_config=seg, slide_path=_SLIDE, output_h5_path=str(tmp_path / "S.patch.h5"))))
