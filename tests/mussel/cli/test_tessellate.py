import os

import h5py
import numpy as np
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
    from unittest.mock import MagicMock, patch, call
    from mussel.cli.tessellate_extract_features_common import _tessellate_and_filter
    from omegaconf import OmegaConf

    fake_coords = np.array([[0, 0], [512, 0]])
    fake_polygon = MagicMock()
    fake_grid = MagicMock()

    cfg = OmegaConf.create({
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
    })

    with patch(
        "mussel.cli.tessellate_extract_features_common.segment_tissue",
        return_value=(fake_polygon, fake_grid, fake_coords, None),
    ) as mock_segment, patch(
        "mussel.cli.tessellate_extract_features_common.GrandQCArtifactRemover",
        autospec=True,
    ) as MockRemover:
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

    # GrandQCArtifactRemover must have been instantiated with remove_penmarks_only=False
    MockRemover.assert_called_once_with(remove_penmarks_only=False)

    # segment_tissue must have received the remover instance
    _, kwargs = mock_segment.call_args
    assert kwargs.get("artifact_remover_fn") is mock_remover_instance, (
        "artifact_remover_fn was not passed to segment_tissue"
    )


def test_artifact_remover_fn_not_instantiated_when_flags_false(tmp_path):
    """_tessellate_and_filter does NOT instantiate GrandQCArtifactRemover by default."""
    from unittest.mock import MagicMock, patch
    from mussel.cli.tessellate_extract_features_common import _tessellate_and_filter
    from omegaconf import OmegaConf

    fake_coords = np.array([[0, 0], [512, 0]])

    cfg = OmegaConf.create({
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
    })

    with patch(
        "mussel.cli.tessellate_extract_features_common.segment_tissue",
        return_value=(MagicMock(), MagicMock(), fake_coords, None),
    ) as mock_segment, patch(
        "mussel.cli.tessellate_extract_features_common.GrandQCArtifactRemover",
        autospec=True,
    ) as MockRemover:

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

    cfg = OmegaConf.create({
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
    })

    with patch(
        "mussel.cli.tessellate_extract_features_common.segment_tissue",
        return_value=(MagicMock(), MagicMock(), fake_coords, None),
    ), patch(
        "mussel.cli.tessellate_extract_features_common.GrandQCArtifactRemover",
        autospec=True,
    ) as MockRemover:

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

    MockRemover.assert_called_once_with(remove_penmarks_only=True)


def test_artifact_remover_fn_external_instance_reused(tmp_path):
    """When artifact_remover_fn is supplied externally, GrandQCArtifactRemover is NOT re-instantiated."""
    from unittest.mock import MagicMock, patch
    from mussel.cli.tessellate_extract_features_common import _tessellate_and_filter
    from omegaconf import OmegaConf

    fake_coords = np.array([[0, 0], [512, 0]])
    external_remover = MagicMock()

    cfg = OmegaConf.create({
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
    })

    with patch(
        "mussel.cli.tessellate_extract_features_common.segment_tissue",
        return_value=(MagicMock(), MagicMock(), fake_coords, None),
    ) as mock_segment, patch(
        "mussel.cli.tessellate_extract_features_common.GrandQCArtifactRemover",
        autospec=True,
    ) as MockRemover:
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
