import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mussel.cli.filter_tessellate import FilterTessellateConfig, main
from mussel.cli.tessellate import NeuralSegConfig, SegConfig, StainSegConfig
from mussel.models import ModelType

_FILTER_TESSELLATE_REQUIRED = dict(
    slide_path="test.svs",
    output_h5_path="test.h5",
    output_pt_path="test.pt",
    classifier_pkl="test.pkl",
)


@pytest.mark.parametrize(
    "model_type,expected_patch_size",
    [
        (ModelType.CONCH1_5, 512),
        (ModelType.VIRCHOW, 224),
        (ModelType.CLIP, 224),
        (ModelType.GOOGLEPATH, 224),
    ],
)
def test_filter_tessellate_default_patch_size_for_model(
    model_type, expected_patch_size
):
    """Test that patch size is automatically set based on model type in filter_tessellate."""
    cfg = FilterTessellateConfig(
        **_FILTER_TESSELLATE_REQUIRED,
        model_type=model_type,
        seg_config=SegConfig(),
    )
    assert cfg.seg_config.patch_size == expected_patch_size


def test_filter_tessellate_explicit_patch_size_preserved():
    """Test that explicitly set patch size is not overridden in filter_tessellate."""
    cfg = FilterTessellateConfig(
        **_FILTER_TESSELLATE_REQUIRED,
        model_type=ModelType.CONCH1_5,
        seg_config=SegConfig(patch_size=384),
    )
    assert cfg.seg_config.patch_size == 384


def test_filter_tessellate_exposes_neural_runtime_config():
    cfg = FilterTessellateConfig(
        **_FILTER_TESSELLATE_REQUIRED,
        seg_config=StainSegConfig(),
        neural_config=NeuralSegConfig(batch_size=32, device="cpu"),
    )

    assert cfg.neural_config.batch_size == 32
    assert cfg.neural_config.device == "cpu"


def test_filter_tessellate_forwards_reusable_neural_segmenter(tmp_path):
    cfg = OmegaConf.create(
        FilterTessellateConfig(
            **_FILTER_TESSELLATE_REQUIRED,
            seg_config=StainSegConfig(),
            neural_config=NeuralSegConfig(batch_size=32, device="cpu"),
            use_gpu=False,
        )
    )
    shared_segmenter = MagicMock()

    with (
        patch(
            "mussel.cli.filter_tessellate._build_neural_segmenter",
            return_value=shared_segmenter,
        ) as build_segmenter,
        patch(
            "mussel.cli.filter_tessellate._build_artifact_remover", return_value=None
        ),
        patch(
            "mussel.cli.filter_tessellate.segment_tissue", return_value=None
        ) as segment,
    ):
        from mussel.cli.filter_tessellate import _main

        _main(cfg, str(tmp_path), tmp_path)

    build_segmenter.assert_called_once()
    assert build_segmenter.call_args.kwargs["neural_config"]["batch_size"] == 32
    assert segment.call_args.kwargs["neural_segmenter"] is shared_segmenter
    shared_segmenter.release.assert_called_once()


def test_filter_tessellate_config_construction(
    tmp_path, test_data_path, classifier_pkl_path, use_gpu, num_workers
):
    """Test that FilterTessellateConfig can be constructed with valid parameters."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    classifier_pkl = classifier_pkl_path
    output_h5_path = os.path.join(tmp_path, "filtered.h5")
    output_pt_path = os.path.join(tmp_path, "filtered.pt")

    seg_config = SegConfig(segment_threshold=0)
    cfg = FilterTessellateConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=classifier_pkl,
        classifier_threshold=0.75,
        model_type=ModelType.RESNET50,
        seg_config=seg_config,
        num_workers=num_workers,
        batch_size=32,
        use_gpu=use_gpu,
        keep_intermediate_files=False,
        save_features_to_h5=True,
    )

    # Verify config was constructed correctly
    assert cfg.slide_path == slide_path
    assert cfg.classifier_threshold == 0.75
    assert cfg.model_type == ModelType.RESNET50


def test_filter_tessellate_config_with_intermediate_files(
    tmp_path, test_data_path, classifier_pkl_path, use_gpu, num_workers
):
    """Test config construction with keep_intermediate_files=True."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    classifier_pkl = classifier_pkl_path
    output_h5_path = os.path.join(tmp_path, "filtered.h5")
    output_pt_path = os.path.join(tmp_path, "filtered.pt")

    seg_config = SegConfig(segment_threshold=0)
    cfg = FilterTessellateConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=classifier_pkl,
        classifier_threshold=0.75,
        model_type=ModelType.RESNET50,
        seg_config=seg_config,
        num_workers=num_workers,
        batch_size=32,
        use_gpu=use_gpu,
        keep_intermediate_files=True,
        save_features_to_h5=True,
    )

    # Verify config was constructed correctly
    assert cfg.keep_intermediate_files is True
    assert cfg.save_features_to_h5 is True


def _filter_tessellate(output_dir, test_data_path, use_gpu, num_workers, classifier_pkl, threshold):
    """filter_tessellate on 300 random tiles of the test slide; return (features, coords) it kept."""
    os.makedirs(output_dir, exist_ok=True)
    cfg = FilterTessellateConfig(
        slide_path=os.path.join(test_data_path, "948176.svs"),
        output_h5_path=os.path.join(output_dir, "filtered.h5"),
        output_pt_path=os.path.join(output_dir, "filtered.pt"),
        classifier_pkl=classifier_pkl,
        classifier_threshold=threshold,
        model_type=ModelType.RESNET50,
        seg_config=SegConfig(segment_threshold=0, max_tiles=300),
        num_workers=num_workers,
        batch_size=32,
        use_gpu=use_gpu,
        save_features_to_h5=True,
    )
    main(OmegaConf.create(cfg))
    with h5py.File(cfg.output_h5_path, "r") as f:
        features, coords = f["features"][:], f["coords"][:]
    pt = torch.load(cfg.output_pt_path, weights_only=True)
    np.testing.assert_allclose(pt.float().numpy(), features, rtol=1e-6, err_msg=".pt differs from .h5")
    return features, coords


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_filter_tessellate_keeps_exactly_the_passing_tiles(
    tmp_path, test_data_path, classifier_pkl_path, use_gpu, num_workers, cut_classifier
):
    """filter_tessellate writes exactly the tiles the classifier passes, with their features."""
    # Threshold 0: every tile passes, so this run gives every tile's features.
    all_feats, all_coords = _filter_tessellate(str(tmp_path / "all"), test_data_path, use_gpu, num_workers,
                                               classifier_pkl_path, threshold=0.0)
    assert all_feats.shape == (300, 1024)
    classifier_pkl, keep = cut_classifier(all_feats)
    kept = keep(all_feats)

    feats, coords = _filter_tessellate(str(tmp_path / "filtered"), test_data_path, use_gpu, num_workers,
                                       classifier_pkl, threshold=0.5)
    np.testing.assert_array_equal(coords, all_coords[kept], err_msg="wrong tiles kept")
    assert feats.shape == (kept.sum(), 1024)
    err = np.linalg.norm(feats - all_feats[kept], axis=1) / np.linalg.norm(all_feats[kept], axis=1)
    assert err.max() <= 1e-2, f"kept tiles' features differ by up to {err.max():.2e} of the row norm"
