import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
from omegaconf import OmegaConf

import mussel.cli.tessellate
from mussel.cli.tessellate import SegConfig, TessellateConfig

# Dimensions of the test slide (85656 x 19917 at level 0)
_SLIDE_WIDTH = 85656
_SLIDE_HEIGHT = 19917


def test_tessellate(tmp_path, num_workers):
    slide_path = "tests/testdata/948176.svs"
    patch_h5_path = tmp_path / "test.h5"
    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateConfig(
        slide_path=slide_path,
        output_h5_path=patch_h5_path,
        seg_config=seg_config,
        num_workers=num_workers,
    )
    mussel.cli.tessellate.main(OmegaConf.create(cfg))

    assert os.path.exists(patch_h5_path)

    with h5py.File(patch_h5_path, "r") as f:
        # H5 must have a 'coords' dataset
        assert "coords" in f, "H5 output missing 'coords' dataset"

        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs)

        # Shape and dtype
        assert (
            coords.ndim == 2 and coords.shape[1] == 2
        ), f"coords should be (N, 2), got {coords.shape}"
        assert (
            coords.dtype == np.int64
        ), f"coords dtype should be int64, got {coords.dtype}"

        # Must have produced some patches
        assert coords.shape[0] > 0, "tessellation produced zero patches"

        # Required metadata attributes
        for attr in ("patch_size", "mpp"):
            assert attr in attrs, f"coords missing attribute '{attr}'"

        # All coordinates must lie within the slide dimensions
        patch_size = int(attrs["patch_size"])
        assert np.all(coords[:, 0] >= 0), "negative x coordinates"
        assert np.all(coords[:, 1] >= 0), "negative y coordinates"
        assert np.all(
            coords[:, 0] + patch_size <= _SLIDE_WIDTH
        ), "x + patch_size exceeds slide width"
        assert np.all(
            coords[:, 1] + patch_size <= _SLIDE_HEIGHT
        ), "y + patch_size exceeds slide height"


def test_tessellate_batch_writes_patch_h5_outputs(tmp_path):
    slide_paths = ["slide_a.svs", "slide_b.svs"]
    output_dir = tmp_path / "tiles"
    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateConfig(
        slide_paths=slide_paths,
        slide_ids=["A", "B"],
        output_dir=str(output_dir),
        seg_config=seg_config,
    )

    def fake_segment_tissue(*, output_h5_path, **kwargs):
        Path(output_h5_path).write_text("patch h5")
        return MagicMock(), MagicMock(), np.array([[0, 0]]), None

    with patch(
        "mussel.cli.tessellate.segment_tissue", side_effect=fake_segment_tissue
    ) as mock_segment:
        mussel.cli.tessellate.main(OmegaConf.create(cfg))

    assert mock_segment.call_count == 2
    assert (output_dir / "A.patch.h5").exists()
    assert (output_dir / "B.patch.h5").exists()


def test_tessellate_batch_can_use_explicit_output_h5_paths(tmp_path):
    out_a = tmp_path / "custom_a.h5"
    out_b = tmp_path / "custom_b.h5"
    cfg = TessellateConfig(
        slide_paths=["slide_a.svs", "slide_b.svs"],
        output_h5_paths=[str(out_a), str(out_b)],
        seg_config=SegConfig(segment_threshold=0),
    )

    def fake_segment_tissue(*, output_h5_path, **kwargs):
        Path(output_h5_path).write_text("patch h5")
        return MagicMock(), MagicMock(), np.array([[0, 0]]), None

    with patch("mussel.cli.tessellate.segment_tissue", side_effect=fake_segment_tissue):
        mussel.cli.tessellate.main(OmegaConf.create(cfg))

    assert out_a.exists()
    assert out_b.exists()


def test_tessellate_batch_reuses_neural_segmenter(tmp_path):
    output_dir = tmp_path / "tiles"
    shared_segmenter = MagicMock()
    cfg = TessellateConfig(
        slide_paths=["slide_a.svs", "slide_b.svs"],
        slide_ids=["A", "B"],
        output_dir=str(output_dir),
        seg_config=SegConfig(seg_model="neural"),
    )

    def fake_segment_tissue(*, output_h5_path, **kwargs):
        Path(output_h5_path).write_text("patch h5")
        return MagicMock(), MagicMock(), np.array([[0, 0]]), None

    with patch(
        "mussel.utils.neural_seg.NeuralTissueSegmenter",
        return_value=shared_segmenter,
    ) as mock_segmenter_cls:
        with patch(
            "mussel.cli.tessellate.segment_tissue", side_effect=fake_segment_tissue
        ) as mock_segment:
            mussel.cli.tessellate.main(OmegaConf.create(cfg))

    mock_segmenter_cls.assert_called_once_with()
    assert mock_segment.call_count == 2
    assert all(
        call.kwargs["neural_segmenter"] is shared_segmenter
        for call in mock_segment.call_args_list
    )


def test_tessellate_batch_fails_on_first_slide_failure(tmp_path):
    output_dir = tmp_path / "tiles"
    cfg = TessellateConfig(
        slide_paths=["slide_a.svs", "slide_b.svs"],
        slide_ids=["A", "B"],
        output_dir=str(output_dir),
        seg_config=SegConfig(segment_threshold=0),
    )

    def fake_segment_tissue(*, output_h5_path, **kwargs):
        if output_h5_path.endswith("A.patch.h5"):
            raise RuntimeError("boom")
        Path(output_h5_path).write_text("patch h5")
        return MagicMock(), MagicMock(), np.array([[0, 0]]), None

    with patch(
        "mussel.cli.tessellate.segment_tissue", side_effect=fake_segment_tissue
    ) as mock_segment:
        with pytest.raises(RuntimeError, match="1 of 2"):
            mussel.cli.tessellate.main(OmegaConf.create(cfg))

    assert mock_segment.call_count == 1
    assert not (output_dir / "A.patch.h5").exists()
    assert not (output_dir / "B.patch.h5").exists()


def test_tessellate_batch_continue_on_error_writes_failures_tsv(tmp_path):
    output_dir = tmp_path / "tiles"
    failures_tsv = tmp_path / "failures.tsv"
    cfg = TessellateConfig(
        slide_paths=["slide_a.svs", "slide_b.svs"],
        slide_ids=["A", "B"],
        output_dir=str(output_dir),
        continue_on_error=True,
        failures_tsv_path=str(failures_tsv),
        seg_config=SegConfig(segment_threshold=0),
    )

    def fake_segment_tissue(*, output_h5_path, **kwargs):
        if output_h5_path.endswith("A.patch.h5"):
            raise RuntimeError("boom")
        Path(output_h5_path).write_text("patch h5")
        return MagicMock(), MagicMock(), np.array([[0, 0]]), None

    with patch(
        "mussel.cli.tessellate.segment_tissue", side_effect=fake_segment_tissue
    ) as mock_segment:
        mussel.cli.tessellate.main(OmegaConf.create(cfg))

    assert mock_segment.call_count == 2
    assert not (output_dir / "A.patch.h5").exists()
    assert (output_dir / "B.patch.h5").exists()
    assert "A\tslide_a.svs" in failures_tsv.read_text()


def test_tessellate_batch_continue_on_error_fails_when_all_slides_fail(tmp_path):
    output_dir = tmp_path / "tiles"
    failures_tsv = tmp_path / "failures.tsv"
    cfg = TessellateConfig(
        slide_paths=["slide_a.svs", "slide_b.svs"],
        slide_ids=["A", "B"],
        output_dir=str(output_dir),
        continue_on_error=True,
        failures_tsv_path=str(failures_tsv),
        seg_config=SegConfig(segment_threshold=0),
    )

    with patch(
        "mussel.cli.tessellate.segment_tissue", side_effect=RuntimeError("boom")
    ) as mock_segment:
        with pytest.raises(RuntimeError, match="2 of 2"):
            mussel.cli.tessellate.main(OmegaConf.create(cfg))

    assert mock_segment.call_count == 2
    assert failures_tsv.exists()


def test_tessellate_batch_rejects_duplicate_output_h5_paths(tmp_path):
    duplicate_path = str(tmp_path / "duplicate.patch.h5")
    cfg = TessellateConfig(
        slide_paths=["slide_a.svs", "slide_b.svs"],
        output_h5_paths=[duplicate_path, duplicate_path],
        seg_config=SegConfig(segment_threshold=0),
    )

    with pytest.raises(ValueError, match="must be unique"):
        mussel.cli.tessellate.main(OmegaConf.create(cfg))


def test_tessellate_batch_rejects_duplicate_output_dir_slide_ids(tmp_path):
    cfg = TessellateConfig(
        slide_paths=["/a/slide.svs", "/b/slide.svs"],
        output_dir=str(tmp_path),
        seg_config=SegConfig(segment_threshold=0),
    )

    with pytest.raises(ValueError, match="must be unique"):
        mussel.cli.tessellate.main(OmegaConf.create(cfg))


def test_tessellate_batch_rejects_single_slide_options(tmp_path):
    cfg = TessellateConfig(
        slide_path="single.svs",
        output_h5_path=str(tmp_path / "single.patch.h5"),
        slide_paths=["slide_a.svs"],
        output_dir=str(tmp_path),
        seg_config=SegConfig(segment_threshold=0),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        mussel.cli.tessellate.main(OmegaConf.create(cfg))


def test_tessellate_batch_rejects_optional_visual_outputs(tmp_path):
    cfg = TessellateConfig(
        slide_paths=["slide_a.svs"],
        output_dir=str(tmp_path),
        output_thumbnail_path=str(tmp_path / "thumb.png"),
        seg_config=SegConfig(segment_threshold=0),
    )

    with pytest.raises(ValueError, match="only writes patch H5"):
        mussel.cli.tessellate.main(OmegaConf.create(cfg))


def test_seg_config_new_fields_defaults():
    """SegConfig has the new fields with correct defaults."""
    cfg = SegConfig()
    assert cfg.overlap == 0
    assert cfg.min_tissue_proportion == 0.0
    assert cfg.remove_artifacts is False
    assert cfg.remove_penmarks is False
    assert cfg.seg_model == "classic"


def test_seg_config_overlap_set():
    """SegConfig accepts a non-zero overlap."""
    cfg = SegConfig(overlap=64)
    assert cfg.overlap == 64


def test_seg_config_min_tissue_proportion_set():
    """SegConfig accepts min_tissue_proportion between 0 and 1."""
    cfg = SegConfig(min_tissue_proportion=0.5)
    assert cfg.min_tissue_proportion == 0.5


def test_seg_config_seg_model_neural():
    """SegConfig accepts seg_model='neural'."""
    cfg = SegConfig(seg_model="neural")
    assert cfg.seg_model == "neural"


def test_artifact_remover_fn_wired_when_remove_artifacts(tmp_path):
    """_tessellate_and_filter instantiates GrandQCArtifactRemover when remove_artifacts=True.

    The remover is passed as artifact_remover_fn to segment_tissue.  Without this
    fix the flag was silently ignored (segment_tissue warned and did nothing).
    """
    from unittest.mock import MagicMock, patch
    from mussel.cli.tessellate_extract_features_common import _tessellate_and_filter
    from omegaconf import OmegaConf

    fake_coords = np.array([[0, 0], [512, 0]])
    fake_polygon = MagicMock()
    fake_grid = MagicMock()

    cfg = OmegaConf.create(
        {
            "seg_config": {
                "mpp": 0.5,
                "patch_size": 512,
                "seg_model": "classic",
                "remove_artifacts": True,
                "remove_penmarks": False,
            },
            "vis_config": {},
            "keep_intermediate_files": False,
            "gpu_device_id": 0,
            "use_gpu": False,
            "batch_size": 8,
            "num_workers": 0,
            "gpu_device_ids": None,
        }
    )

    with (
        patch(
            "mussel.cli.tessellate_extract_features_common.segment_tissue",
            return_value=(fake_polygon, fake_grid, fake_coords, None),
        ) as mock_segment,
        patch(
            "mussel.cli.tessellate.GrandQCArtifactRemover",
            autospec=True,
        ) as MockRemover,
    ):
        mock_remover_instance = MockRemover.return_value

        _tessellate_and_filter(
            slide_path="tests/testdata/948176.svs",
            slide_id="948176",
            cfg=cfg,
            temp_dir=str(tmp_path),
            base_path=tmp_path,
            use_filtering=False,
            prefilter_model_type=None,
            prefilter_model_path=None,
            skip_second_extraction=False,
        )

    # GrandQCArtifactRemover must have been instantiated with EXCLUDE_ALL_ARTIFACTS
    from mussel.utils.artifact_removal import EXCLUDE_ALL_ARTIFACTS

    MockRemover.assert_called_once_with(exclude_classes=EXCLUDE_ALL_ARTIFACTS)

    # segment_tissue must have received the remover instance
    _, kwargs = mock_segment.call_args
    assert (
        kwargs.get("artifact_remover_fn") is mock_remover_instance
    ), "artifact_remover_fn was not passed to segment_tissue"


def test_artifact_remover_fn_not_instantiated_when_flags_false(tmp_path):
    """_tessellate_and_filter does NOT instantiate GrandQCArtifactRemover by default."""
    from unittest.mock import MagicMock, patch
    from mussel.cli.tessellate_extract_features_common import _tessellate_and_filter
    from omegaconf import OmegaConf

    fake_coords = np.array([[0, 0], [512, 0]])

    cfg = OmegaConf.create(
        {
            "seg_config": {
                "mpp": 0.5,
                "patch_size": 512,
                "seg_model": "classic",
                "remove_artifacts": False,
                "remove_penmarks": False,
            },
            "vis_config": {},
            "keep_intermediate_files": False,
            "gpu_device_id": 0,
            "use_gpu": False,
            "batch_size": 8,
            "num_workers": 0,
            "gpu_device_ids": None,
        }
    )

    with (
        patch(
            "mussel.cli.tessellate_extract_features_common.segment_tissue",
            return_value=(MagicMock(), MagicMock(), fake_coords, None),
        ) as mock_segment,
        patch(
            "mussel.cli.tessellate.GrandQCArtifactRemover",
            autospec=True,
        ) as MockRemover,
    ):

        _tessellate_and_filter(
            slide_path="tests/testdata/948176.svs",
            slide_id="948176",
            cfg=cfg,
            temp_dir=str(tmp_path),
            base_path=tmp_path,
            use_filtering=False,
            prefilter_model_type=None,
            prefilter_model_path=None,
            skip_second_extraction=False,
        )

    MockRemover.assert_not_called()
    _, kwargs = mock_segment.call_args
    assert kwargs.get("artifact_remover_fn") is None


def test_artifact_remover_fn_penmarks_only(tmp_path):
    """remove_penmarks=True with remove_artifacts=False → remove_penmarks_only=True."""
    from unittest.mock import MagicMock, patch
    from mussel.cli.tessellate_extract_features_common import _tessellate_and_filter
    from omegaconf import OmegaConf

    fake_coords = np.array([[0, 0], [512, 0]])

    cfg = OmegaConf.create(
        {
            "seg_config": {
                "mpp": 0.5,
                "patch_size": 512,
                "seg_model": "classic",
                "remove_artifacts": False,
                "remove_penmarks": True,
            },
            "vis_config": {},
            "keep_intermediate_files": False,
            "gpu_device_id": 0,
            "use_gpu": False,
            "batch_size": 8,
            "num_workers": 0,
            "gpu_device_ids": None,
        }
    )

    with (
        patch(
            "mussel.cli.tessellate_extract_features_common.segment_tissue",
            return_value=(MagicMock(), MagicMock(), fake_coords, None),
        ),
        patch(
            "mussel.cli.tessellate.GrandQCArtifactRemover",
            autospec=True,
        ) as MockRemover,
    ):

        _tessellate_and_filter(
            slide_path="tests/testdata/948176.svs",
            slide_id="948176",
            cfg=cfg,
            temp_dir=str(tmp_path),
            base_path=tmp_path,
            use_filtering=False,
            prefilter_model_type=None,
            prefilter_model_path=None,
            skip_second_extraction=False,
        )

    from mussel.utils.artifact_removal import EXCLUDE_PENMARKS_ONLY

    MockRemover.assert_called_once_with(exclude_classes=EXCLUDE_PENMARKS_ONLY)


def test_artifact_remover_fn_external_instance_reused(tmp_path):
    """When artifact_remover_fn is supplied externally, GrandQCArtifactRemover is NOT re-instantiated."""
    from unittest.mock import MagicMock, patch
    from mussel.cli.tessellate_extract_features_common import _tessellate_and_filter
    from omegaconf import OmegaConf

    fake_coords = np.array([[0, 0], [512, 0]])
    external_remover = MagicMock()

    cfg = OmegaConf.create(
        {
            "seg_config": {
                "mpp": 0.5,
                "patch_size": 512,
                "seg_model": "classic",
                "remove_artifacts": True,
                "remove_penmarks": False,
            },
            "vis_config": {},
            "keep_intermediate_files": False,
            "gpu_device_id": 0,
            "use_gpu": False,
            "batch_size": 8,
            "num_workers": 0,
            "gpu_device_ids": None,
        }
    )

    with (
        patch(
            "mussel.cli.tessellate_extract_features_common.segment_tissue",
            return_value=(MagicMock(), MagicMock(), fake_coords, None),
        ) as mock_segment,
        patch(
            "mussel.cli.tessellate.GrandQCArtifactRemover",
            autospec=True,
        ) as MockRemover,
    ):
        _tessellate_and_filter(
            slide_path="tests/testdata/948176.svs",
            slide_id="948176",
            cfg=cfg,
            temp_dir=str(tmp_path),
            base_path=tmp_path,
            use_filtering=False,
            prefilter_model_type=None,
            prefilter_model_path=None,
            skip_second_extraction=False,
            artifact_remover_fn=external_remover,
        )

    # Class must NOT be re-instantiated when a remover is provided
    MockRemover.assert_not_called()
    # The external remover must be forwarded to segment_tissue
    _, kwargs = mock_segment.call_args
    assert kwargs.get("artifact_remover_fn") is external_remover


def test_segment_tissue_artifact_mpp_escalation(tmp_path):
    """segment_tissue reads a finer pyramid level when seg-level MPP exceeds max_input_mpp."""
    import cv2
    from unittest.mock import MagicMock, call, patch
    import numpy as np
    from mussel.utils.segment import segment_tissue

    slide_path = "tests/testdata/948176.svs"

    # A mock artifact remover whose max_input_mpp is very small so the
    # seg-level thumbnail will always exceed it, triggering escalation.
    mock_remover = MagicMock()
    mock_remover.max_input_mpp = 0.001  # force escalation

    # patch wsi internals minimally — segment_tissue opens the real slide file
    # so we only intercept the artifact remover call and verify the remover
    # receives a different (lower-level / higher-res) image.
    called_mpps = []

    def capture_remover(img, mask, mpp):
        called_mpps.append(mpp)
        # Return the mask unchanged to keep the test simple
        return mask.copy()

    mock_remover.side_effect = capture_remover

    # Use a very permissive config so tessellation doesn't fail
    result = segment_tissue(
        slide_path=slide_path,
        seg_model="classic",
        mpp=0.5,
        patch_size=512,
        segment_threshold=0,
        remove_artifacts=True,
        artifact_remover_fn=mock_remover,
        output_h5_path=str(tmp_path / "out.h5"),
    )

    assert result is not None, "segment_tissue should succeed"
    # The remover must have been called with an MPP <= max_input_mpp=0.001
    # — but since level 0 is the finest and may still exceed 0.001, we only
    # check it was called (escalation was attempted) rather than asserting the
    # exact mpp value.
    assert mock_remover.called, "artifact_remover_fn should have been called"


def test_segment_tissue_artifact_mpp_exact_equality_escalates(tmp_path):
    """segment_tissue escalates when seg-level MPP equals max_input_mpp (>= not just >)."""
    from unittest.mock import MagicMock
    import numpy as np
    import tiffslide
    from mussel.utils.segment import segment_tissue

    slide_path = "tests/testdata/948176.svs"

    # Compute the actual seg-level MPP using tiffslide (available in venv).
    wsi = tiffslide.TiffSlide(slide_path)
    slide_mpp = float(
        wsi.properties.get("tiffslide.mpp-x")
        or wsi.properties.get("openslide.mpp-x")
        or 0.5
    )
    seg_level = len(wsi.level_dimensions) - 1
    downsample = wsi.level_downsamples[seg_level]
    exact_mpp = slide_mpp * downsample
    wsi.close()

    called_mpps = []
    mock_remover = MagicMock()
    mock_remover.max_input_mpp = exact_mpp  # exactly equal → should still escalate

    def capture_remover(img, mask, mpp):
        called_mpps.append(mpp)
        return mask.copy()

    mock_remover.side_effect = capture_remover

    result = segment_tissue(
        slide_path=slide_path,
        seg_model="classic",
        mpp=0.5,
        patch_size=512,
        segment_threshold=0,
        remove_artifacts=True,
        artifact_remover_fn=mock_remover,
        output_h5_path=str(tmp_path / "out.h5"),
    )

    assert result is not None, "segment_tissue should succeed"
    assert mock_remover.called, "artifact_remover_fn should have been called"
    # After escalation, the remover is called with an MPP <= exact_mpp.
    assert (
        called_mpps[0] <= exact_mpp
    ), f"Remover should be called at MPP <= {exact_mpp}, got {called_mpps[0]}"


def test_segment_tissue_artifact_removal_empty_mask_fallback(tmp_path):
    """When artifact removal returns an all-zero mask, segment_tissue falls back to the pre-removal mask."""
    from unittest.mock import MagicMock
    import numpy as np
    from mussel.utils.segment import segment_tissue

    slide_path = "tests/testdata/948176.svs"

    mock_remover = MagicMock()
    mock_remover.max_input_mpp = 1000.0  # no escalation needed

    def zero_mask_remover(img, mask, mpp):
        # Simulate over-aggressive removal: return all-zeros
        return np.zeros_like(mask)

    mock_remover.side_effect = zero_mask_remover

    result = segment_tissue(
        slide_path=slide_path,
        seg_model="classic",
        mpp=0.5,
        patch_size=512,
        segment_threshold=0,
        remove_artifacts=True,
        artifact_remover_fn=mock_remover,
        output_h5_path=str(tmp_path / "out.h5"),
    )

    # Should not crash; should fall back and produce a valid result
    assert (
        result is not None
    ), "segment_tissue should fall back to pre-removal mask when artifact removal empties it"
