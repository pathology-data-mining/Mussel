import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mussel.cli.tessellate import SegConfig
from mussel.cli.tessellate_extract_features import (
    TessellateExtractFeaturesConfig, main)
from mussel.models import ModelType

# The batch integration tests check orchestration (two slides, filtering, output
# layout), not throughput: a seeded random 300 tiles per slide keeps them in
# seconds of GPU time. The whole test slide (segment_threshold=0) took 15 min
# for the filtering test on a P40, CPU-bound on tile decoding.
_MAX_TILES = 300

# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

_FAKE_COORDS = np.array([[0, 0], [256, 0], [0, 256]], dtype=np.int64)
_FAKE_PATCH_FEATURES = np.random.rand(3, 2048).astype(np.float32)
_FAKE_SLIDE_FEATURES = np.random.rand(1, 768).astype(np.float32)


def _make_segment_tissue_mock():
    def _fake(slide_path, slide_id, output_h5_path, **kwargs):
        os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
        with h5py.File(output_h5_path, "w") as f:
            f.create_dataset("coords", data=_FAKE_COORDS)
        return (MagicMock(), MagicMock(), _FAKE_COORDS.tolist(), None)

    return _fake


def _make_extract_patch_features_batch_mock():
    def _fake(patch_h5_paths, slide_paths, output_h5_paths, **kwargs):
        for path in output_h5_paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with h5py.File(path, "w") as f:
                f.create_dataset("features", data=_FAKE_PATCH_FEATURES)
                f.create_dataset("coords", data=_FAKE_COORDS)

    return _fake


def _make_aggregate_slide_features_batch_mock():
    def _fake(
        patch_features_h5_paths, output_h5_paths=None, output_pt_paths=None, **kwargs
    ):
        if output_h5_paths:
            for path in output_h5_paths:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with h5py.File(path, "w") as f:
                    f.create_dataset("features", data=_FAKE_SLIDE_FEATURES)
                    f.create_dataset("coords", data=_FAKE_COORDS)
        if output_pt_paths:
            from mussel.utils import save_torch_tensor

            for path in output_pt_paths:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                save_torch_tensor(path, torch.from_numpy(_FAKE_SLIDE_FEATURES))

    return _fake


@pytest.mark.slow
@pytest.mark.integration
def test_tessellate_extract_features_batch_basic(
    tmp_path, test_data_path, use_gpu, num_workers
):
    """Test basic batch processing of multiple slides using the unified command."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    # Use the same slide twice to test batch processing
    slide_paths = [slide_path, slide_path]
    output_dir = str(tmp_path / "batch_output")

    seg_config = SegConfig(segment_threshold=0, max_tiles=_MAX_TILES)
    cfg = TessellateExtractFeaturesConfig(
        slide_paths=slide_paths,
        slide_ids=["slide1", "slide2"],
        output_dir=output_dir,
        classifier_pkl=None,  # No filtering
        prefilter_model_type=ModelType.RESNET50,
        seg_config=seg_config,
        num_workers=num_workers,
        batch_size=32,
        use_gpu=use_gpu,
        keep_intermediate_files=False,
    )

    main(OmegaConf.create(cfg))

    # Check that output files were created for both slides
    assert os.path.exists(os.path.join(output_dir, "h5", "slide1.features.h5"))
    assert os.path.exists(os.path.join(output_dir, "pt", "slide1.features.pt"))
    assert os.path.exists(os.path.join(output_dir, "h5", "slide2.features.h5"))
    assert os.path.exists(os.path.join(output_dir, "pt", "slide2.features.pt"))

    # Verify output H5 content for both slides
    for slide_id in ("slide1", "slide2"):
        h5_path = os.path.join(output_dir, "h5", f"{slide_id}.features.h5")
        pt_path = os.path.join(output_dir, "pt", f"{slide_id}.features.pt")
        with h5py.File(h5_path, "r") as f:
            assert "features" in f, f"{slide_id}: missing 'features' in H5"
            assert "coords" in f, f"{slide_id}: missing 'coords' in H5"
            feats = f["features"][:]
            coords = f["coords"][:]
        # Shape sanity
        assert (
            feats.ndim == 2 and feats.shape[0] > 0
        ), f"{slide_id}: features shape unexpected: {feats.shape}"
        assert (
            feats.shape[1] == 1024
        ), f"{slide_id}: RESNET50 feature dim should be 1024, got {feats.shape[1]}"
        assert coords.shape == (
            feats.shape[0],
            2,
        ), f"{slide_id}: coords shape mismatch with features"
        # dtype and numerical health
        assert (
            feats.dtype == np.float32
        ), f"{slide_id}: features dtype should be float32, got {feats.dtype}"
        assert np.all(np.isfinite(feats)), f"{slide_id}: features contain NaN or Inf"
        # .pt file must be loadable and match H5
        pt_feats = torch.load(pt_path, map_location="cpu", weights_only=True)
        assert pt_feats.shape == torch.Size(
            feats.shape
        ), f"{slide_id}: pt shape {pt_feats.shape} != h5 shape {feats.shape}"


def _run_batch(output_dir, test_data_path, use_gpu, num_workers, classifier_pkl=None, threshold=0.5):
    """Two copies of the test slide, RESNET50, the same random tiles; returns {slide_id: (features, coords)}."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    cfg = TessellateExtractFeaturesConfig(
        slide_paths=[slide_path, slide_path],
        slide_ids=["slide1", "slide2"],
        output_dir=output_dir,
        classifier_pkl=classifier_pkl,
        classifier_threshold=threshold,
        prefilter_model_type=ModelType.RESNET50,
        model_type=None,  # Will use same as prefilter
        seg_config=SegConfig(segment_threshold=0, max_tiles=_MAX_TILES),
        num_workers=num_workers,
        batch_size=32,
        use_gpu=use_gpu,
        keep_intermediate_files=False,
    )
    main(OmegaConf.create(cfg))
    out = {}
    for slide_id in ("slide1", "slide2"):
        h5_path = os.path.join(output_dir, "h5", f"{slide_id}.features.h5")
        assert os.path.exists(h5_path), f"missing output for {slide_id}"
        with h5py.File(h5_path, "r") as f:
            out[slide_id] = f["features"][:], f["coords"][:]
    return out


@pytest.mark.slow
@pytest.mark.integration
def test_tessellate_extract_features_batch_with_filtering(
    tmp_path, test_data_path, use_gpu, num_workers, cut_classifier
):
    """Batch filtering keeps exactly the tiles the classifier passes (see ``cut_classifier``)."""
    unfiltered = _run_batch(str(tmp_path / "all"), test_data_path, use_gpu, num_workers)
    classifier_pkl, keep = cut_classifier(unfiltered["slide1"][0])

    filtered = _run_batch(str(tmp_path / "filtered"), test_data_path, use_gpu, num_workers,
                          classifier_pkl=classifier_pkl, threshold=0.5)
    for slide_id, (all_feats, all_coords) in unfiltered.items():
        kept = keep(all_feats)
        assert 0.25 * len(kept) <= kept.sum() <= 0.75 * len(kept)
        out_feats, out_coords = filtered[slide_id]
        np.testing.assert_array_equal(out_coords, all_coords[kept], err_msg=f"{slide_id}: wrong tiles kept")
        assert out_feats.shape == (kept.sum(), 1024), f"{slide_id}: RESNET50 feature dim should be 1024"
        assert np.all(np.isfinite(out_feats)), f"{slide_id}: features contain NaN or Inf after filtering"


def test_tessellate_extract_features_batch_with_model_aggregation(
    tmp_path, test_data_path, use_gpu, num_workers
):
    """Test batch processing with slide-level model aggregation (mocked encoder).

    Uses _make_* mocks so the test runs quickly without loading real models.
    Verifies that the aggregation path produces slide-level feature files.
    """
    slide_path = os.path.join(test_data_path, "948176.svs")
    slide_paths = [slide_path, slide_path]
    output_dir = str(tmp_path / "batch_output")

    seg_config = SegConfig(segment_threshold=0, max_tiles=_MAX_TILES)
    cfg = TessellateExtractFeaturesConfig(
        slide_paths=slide_paths,
        slide_ids=["slide1", "slide2"],
        output_dir=output_dir,
        classifier_pkl=None,
        prefilter_model_type=ModelType.RESNET50,
        aggregation_method="model",
        slide_model_type=ModelType.GIGAPATH_SLIDE,
        seg_config=seg_config,
        num_workers=num_workers,
        batch_size=32,
        slide_batch_size=2,
        use_gpu=use_gpu,
        keep_intermediate_files=True,
    )

    with (
        patch(
            "mussel.cli.tessellate_extract_features_common.segment_tissue",
            side_effect=_make_segment_tissue_mock(),
        ),
        patch(
            "mussel.cli.tessellate_extract_features.extract_patch_features_batch",
            side_effect=_make_extract_patch_features_batch_mock(),
        ),
        patch(
            "mussel.cli.tessellate_extract_features.aggregate_slide_features_batch",
            side_effect=_make_aggregate_slide_features_batch_mock(),
        ),
    ):
        main(OmegaConf.create(cfg))

    for slide_id in ("slide1", "slide2"):
        assert os.path.exists(os.path.join(output_dir, "h5", f"{slide_id}.features.h5"))
        # Slide-level features should be a single vector
        with h5py.File(
            os.path.join(output_dir, "h5", f"{slide_id}.features.h5"), "r"
        ) as f:
            assert (
                f["features"].shape[0] == 1
            ), f"{slide_id}: expected 1 slide-level feature row"


def test_auto_slide_id_generation(tmp_path, test_data_path, use_gpu, num_workers):
    """Test that slide IDs are auto-generated from filenames.

    Uses mocks for tissue segmentation and feature extraction so the test
    runs quickly without real model inference.
    """
    import numpy as np

    slide_path = os.path.join(test_data_path, "948176.svs")
    slide_paths = [slide_path]
    output_dir = str(tmp_path / "batch_output")

    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateExtractFeaturesConfig(
        slide_paths=slide_paths,
        slide_ids=None,  # Should auto-generate from filenames
        output_dir=output_dir,
        classifier_pkl=None,
        prefilter_model_type=ModelType.RESNET50,
        seg_config=seg_config,
        num_workers=num_workers,
        batch_size=32,
        use_gpu=use_gpu,
        keep_intermediate_files=False,
    )

    fake_coords = np.array([[0, 0], [256, 0], [0, 256]], dtype=np.int64)
    fake_features = np.random.rand(3, 2048).astype(np.float32)

    def _fake_segment_tissue(slide_path, slide_id, output_h5_path, **kwargs):
        """Write a minimal tessellate H5 so the pipeline can proceed."""
        os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
        with h5py.File(output_h5_path, "w") as f:
            f.create_dataset("coords", data=fake_coords)
        return (MagicMock(), MagicMock(), fake_coords.tolist(), None)

    def _fake_extract_patch_features_batch(
        patch_h5_paths, slide_paths, output_h5_paths, **kwargs
    ):
        """Write minimal feature H5 files so the pipeline can save PT outputs."""
        for output_h5_path in output_h5_paths:
            os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
            with h5py.File(output_h5_path, "w") as f:
                f.create_dataset("features", data=fake_features)
                f.create_dataset("coords", data=fake_coords)

    with (
        patch(
            "mussel.cli.tessellate_extract_features_common.segment_tissue",
            side_effect=_fake_segment_tissue,
        ),
        patch(
            "mussel.cli.tessellate_extract_features.extract_patch_features_batch",
            side_effect=_fake_extract_patch_features_batch,
        ),
    ):
        main(OmegaConf.create(cfg))

    # Should use filename stem as ID
    assert os.path.exists(os.path.join(output_dir, "h5", "948176.features.h5"))
    assert os.path.exists(os.path.join(output_dir, "pt", "948176.features.pt"))


def test_tile_level_batching_single_model_load(
    tmp_path, test_data_path, use_gpu, num_workers
):
    """Test that tile-level batching produces outputs for all slides.

    Uses mocks so the test runs quickly without real model inference or SVS I/O.
    Model-loading-once behaviour is covered by test_feature_extract.py.
    """
    slide_path = os.path.join(test_data_path, "948176.svs")
    slide_paths = [slide_path, slide_path]
    output_dir = str(tmp_path / "batch_output")

    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateExtractFeaturesConfig(
        slide_paths=slide_paths,
        slide_ids=["slide1", "slide2"],
        output_dir=output_dir,
        classifier_pkl=None,
        prefilter_model_type=ModelType.RESNET50,
        aggregation_method="mean",
        seg_config=seg_config,
        num_workers=num_workers,
        batch_size=32,
        use_gpu=use_gpu,
        keep_intermediate_files=False,
    )

    with (
        patch(
            "mussel.cli.tessellate_extract_features_common.segment_tissue",
            side_effect=_make_segment_tissue_mock(),
        ),
        patch(
            "mussel.cli.tessellate_extract_features.extract_patch_features_batch",
            side_effect=_make_extract_patch_features_batch_mock(),
        ),
        patch(
            "mussel.cli.tessellate_extract_features.aggregate_slide_features_batch",
            side_effect=_make_aggregate_slide_features_batch_mock(),
        ),
    ):
        main(OmegaConf.create(cfg))

    assert os.path.exists(os.path.join(output_dir, "h5", "slide1.features.h5"))
    assert os.path.exists(os.path.join(output_dir, "h5", "slide2.features.h5"))


def test_tile_level_batching_with_slide_aggregation(
    tmp_path, test_data_path, use_gpu, num_workers
):
    """Test tile-level batching combined with slide-level aggregation batching.

    Uses mocks so the test runs quickly without real model inference or SVS I/O.
    """
    slide_path = os.path.join(test_data_path, "948176.svs")
    slide_paths = [slide_path, slide_path]
    output_dir = str(tmp_path / "batch_output")

    seg_config = SegConfig(segment_threshold=0)
    cfg = TessellateExtractFeaturesConfig(
        slide_paths=slide_paths,
        slide_ids=["slide1", "slide2"],
        output_dir=output_dir,
        classifier_pkl=None,
        prefilter_model_type=ModelType.RESNET50,
        aggregation_method="model",
        slide_model_type=ModelType.GIGAPATH_SLIDE,
        seg_config=seg_config,
        num_workers=num_workers,
        batch_size=32,
        slide_batch_size=2,
        use_gpu=use_gpu,
        keep_intermediate_files=True,
    )

    with (
        patch(
            "mussel.cli.tessellate_extract_features_common.segment_tissue",
            side_effect=_make_segment_tissue_mock(),
        ),
        patch(
            "mussel.cli.tessellate_extract_features.extract_patch_features_batch",
            side_effect=_make_extract_patch_features_batch_mock(),
        ),
        patch(
            "mussel.cli.tessellate_extract_features.aggregate_slide_features_batch",
            side_effect=_make_aggregate_slide_features_batch_mock(),
        ),
    ):
        main(OmegaConf.create(cfg))

    # Verify intermediate patch features were created
    assert os.path.exists(os.path.join(output_dir, "tile_h5", "slide1.patch.h5"))
    assert os.path.exists(os.path.join(output_dir, "tile_h5", "slide2.patch.h5"))

    # Verify final aggregated features were created
    assert os.path.exists(os.path.join(output_dir, "h5", "slide1.features.h5"))
    assert os.path.exists(os.path.join(output_dir, "h5", "slide2.features.h5"))

    # Verify patch features have multiple patches
    with h5py.File(os.path.join(output_dir, "tile_h5", "slide1.patch.h5"), "r") as f:
        assert f["features"].shape[0] > 1

    # Verify aggregated features are slide-level (single vector)
    with h5py.File(os.path.join(output_dir, "h5", "slide1.features.h5"), "r") as f:
        assert f["features"].shape[0] == 1


def test_wsi_dir_discovery(test_data_path):
    """Test that wsi_dir collects slides from a directory."""
    from mussel.utils.file import WSI_EXTENSIONS, collect_wsi_paths

    paths = collect_wsi_paths(str(test_data_path), search_nested=False)
    assert len(paths) > 0
    assert all(Path(p).suffix.lower() in WSI_EXTENSIONS for p in paths)


def test_wsi_dir_nested(tmp_path):
    """Test that search_nested finds files in subdirectories."""
    from mussel.utils.file import collect_wsi_paths

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "slide.svs").touch()
    (tmp_path / "top.svs").touch()
    # Non-nested: only top-level
    flat = collect_wsi_paths(str(tmp_path), search_nested=False)
    assert len(flat) == 1
    assert flat[0].endswith("top.svs")
    # Nested: both
    nested = collect_wsi_paths(str(tmp_path), search_nested=True)
    assert len(nested) == 2


def test_wsi_dir_conflict_raises(tmp_path):
    """Test that specifying both wsi_dir and slide_paths (or slide_path) raises an error."""
    from mussel.cli.tessellate_extract_features import \
        TessellateExtractFeaturesConfig

    dummy_wsi = tmp_path / "dummy.svs"
    dummy_wsi.touch()

    # wsi_dir + slide_paths should raise
    cfg = OmegaConf.structured(TessellateExtractFeaturesConfig)
    cfg.wsi_dir = str(tmp_path)
    cfg.slide_paths = ["some_slide.svs"]
    cfg.output_dir = str(tmp_path / "out")
    cfg.model_type = "CLIP"
    with pytest.raises(ValueError, match="Cannot specify both wsi_dir and slide_paths"):
        main(cfg)

    # wsi_dir + slide_path should also raise
    cfg2 = OmegaConf.structured(TessellateExtractFeaturesConfig)
    cfg2.wsi_dir = str(tmp_path)
    cfg2.slide_path = "some_slide.svs"
    cfg2.output_dir = str(tmp_path / "out")
    cfg2.model_type = "CLIP"
    with pytest.raises(ValueError, match="Cannot specify both wsi_dir and slide_path"):
        main(cfg2)


def _two_slides_named_alike(tmp_path, test_data_path):
    """Two slide paths with the same file name (948176.svs) in different folders."""
    paths = []
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        link = tmp_path / folder / "948176.svs"
        link.symlink_to(os.path.join(test_data_path, "948176.svs"))
        paths.append(str(link))
    return paths


def test_batch_slides_with_the_same_file_name_keep_their_own_tiles(tmp_path, test_data_path):
    """Each slide's features are extracted at its own tiles, not another slide's.

    The slides share one temp directory for intermediate files; named by file
    stem, b's tiles overwrote a's, and a was extracted at b's coordinates.
    """
    slide_paths = _two_slides_named_alike(tmp_path, test_data_path)
    own_coords = {slide_paths[0]: np.array([[0, 0]]), slide_paths[1]: np.array([[256, 256], [512, 512]])}

    def segment(slide_path, slide_id, output_h5_path, **kwargs):
        os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
        with h5py.File(output_h5_path, "w") as f:
            f.create_dataset("coords", data=own_coords[slide_path])
        return (MagicMock(), MagicMock(), own_coords[slide_path].tolist(), None)

    extracted = {}

    def extract(patch_h5_paths, slide_paths, output_h5_paths, **kwargs):
        for patch_h5, slide in zip(patch_h5_paths, slide_paths):
            with h5py.File(patch_h5, "r") as f:
                extracted[slide] = f["coords"][:]
        _make_extract_patch_features_batch_mock()(patch_h5_paths, slide_paths, output_h5_paths)

    cfg = TessellateExtractFeaturesConfig(
        slide_paths=slide_paths, slide_ids=["A", "B"], output_dir=str(tmp_path / "out"),
        classifier_pkl=None, prefilter_model_type=ModelType.RESNET50,
        seg_config=SegConfig(segment_threshold=0), num_workers=0, use_gpu=False,
    )
    with patch("mussel.cli.tessellate_extract_features_common.segment_tissue", side_effect=segment), \
            patch("mussel.cli.tessellate_extract_features.extract_patch_features_batch", side_effect=extract):
        main(OmegaConf.create(cfg))
    for slide in slide_paths:
        np.testing.assert_array_equal(extracted[slide], own_coords[slide], err_msg=f"{slide} got another slide's tiles")


def test_batch_rejects_duplicate_slide_ids(tmp_path, test_data_path):
    """Slide IDs default to the file stem; two slides named alike need explicit slide_ids."""
    cfg = TessellateExtractFeaturesConfig(
        slide_paths=_two_slides_named_alike(tmp_path, test_data_path), output_dir=str(tmp_path / "out"),
        classifier_pkl=None, prefilter_model_type=ModelType.RESNET50,
        seg_config=SegConfig(segment_threshold=0), num_workers=0, use_gpu=False,
    )
    # Mocked, so the old code (which accepted the duplicates) fails fast instead
    # of running the pipeline.
    with patch("mussel.cli.tessellate_extract_features_common.segment_tissue",
               side_effect=_make_segment_tissue_mock()), \
            patch("mussel.cli.tessellate_extract_features.extract_patch_features_batch",
                  side_effect=_make_extract_patch_features_batch_mock()):
        with pytest.raises(ValueError, match="slide IDs must be unique"):
            main(OmegaConf.create(cfg))
