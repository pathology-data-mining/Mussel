import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from mussel.cli.filter_tessellate import FilterTessellateConfig
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


@pytest.mark.slow
@pytest.mark.integration
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


@pytest.mark.slow
@pytest.mark.integration
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
