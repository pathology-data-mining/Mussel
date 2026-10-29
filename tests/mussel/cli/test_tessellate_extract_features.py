import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mussel.cli.tessellate import SegConfig

# isort: off
from mussel.cli.tessellate_extract_features import (
    TessellateExtractFeaturesConfig,
    _build_neural_segmenter,
    main,
)

# isort: on
from mussel.models import ModelType

_TEF_REQUIRED = dict(
    slide_path="test.svs",
    output_h5_path="test.h5",
    output_pt_path="test.pt",
)


@pytest.mark.parametrize(
    "model_type,expected_patch_size",
    [
        (ModelType.CONCH1_5, 512),
        (ModelType.VIRCHOW, 224),
        (ModelType.CLIP, 224),
        (ModelType.GOOGLEPATH, 224),
        (ModelType.GIGAPATH, SegConfig.DEFAULT_PATCH_SIZE),
    ],
)
def test_default_patch_size_for_model(model_type, expected_patch_size):
    """Test that patch size is automatically set based on model type."""
    cfg = TessellateExtractFeaturesConfig(
        **_TEF_REQUIRED,
        prefilter_model_type=model_type,
        seg_config=SegConfig(),
    )
    assert cfg.seg_config.patch_size == expected_patch_size


def test_explicit_patch_size_preserved():
    """Test that explicitly set patch size is not overridden."""
    cfg = TessellateExtractFeaturesConfig(
        **_TEF_REQUIRED,
        prefilter_model_type=ModelType.CONCH1_5,
        seg_config=SegConfig(patch_size=384),
    )
    assert cfg.seg_config.patch_size == 384


def _tiles_chosen_by_main(seg_config=None, **overrides):
    """seg_config.patch_size / mpp after main()'s model-based defaults (no slide is read)."""
    seen = {}

    def capture(cfg):
        seen["tiles"] = (cfg.seg_config.patch_size, cfg.seg_config.mpp)

    cfg = OmegaConf.create(TessellateExtractFeaturesConfig(
        **_TEF_REQUIRED, seg_config=seg_config or SegConfig(), **overrides))
    with patch("mussel.cli.tessellate_extract_features._main_single", side_effect=capture), \
            patch("mussel.cli.tessellate_extract_features._main_single_multi_model", side_effect=capture):
        main(cfg)
    return seen["tiles"]


@pytest.mark.parametrize(
    "overrides,expected",
    [
        (dict(model_type=ModelType.HOPTIMUS1), (224, 0.5)),
        (dict(model_type=ModelType.GPFM), (512, 0.25)),
        # Multi-model mode (both given): left unset for the per-model grouping, which
        # must not mistake a slide encoder's MPP for the user's.
        (dict(model_type=ModelType.CTRANSPATH, slide_model_type=ModelType.CHIEF_SLIDE), (256, 0.5)),
        (dict(model_type=ModelType.CONCH1_5, slide_model_type=ModelType.TITAN_SLIDE), (256, 0.5)),
        # Values other than the defaults are kept (an explicit 256 px / 0.5 µm/px looks like the default).
        (dict(model_type=ModelType.GPFM, seg_config=SegConfig(patch_size=384, mpp=0.75)), (384, 0.75)),
    ],
)
def test_main_tiles_at_the_models_size_and_mpp(overrides, expected):
    assert _tiles_chosen_by_main(**overrides) == expected


# Slide encoders whose tiles differ from their patch encoder's: CHIEF (CTransPath, 224 px)
# and MADELEINE (CONCH v1, 448 px) both need 256 px tiles at 1.0 µm/px.
_SLIDE_ENCODER_TILES = [
    (ModelType.CHIEF_SLIDE, ModelType.CTRANSPATH, (256, 1.0)),
    (ModelType.MADELEINE_SLIDE, ModelType.CONCH_V1, (256, 1.0)),
    (ModelType.TITAN_SLIDE, ModelType.CONCH1_5, (512, 0.5)),
]


@pytest.mark.parametrize("slide_model_type,patch_encoder,tiles", _SLIDE_ENCODER_TILES)
def test_single_slide_tiles_for_the_slide_encoder(
    slide_model_type, patch_encoder, tiles
):
    """_main_single keeps the slide encoder's tiles, not its patch encoder's."""
    seen = {}

    def capture(**kwargs):
        cfg = kwargs["cfg"]
        seen["tiles"] = (cfg.seg_config.patch_size, cfg.seg_config.mpp)
        seen["model_type"] = kwargs["model_type"]
        return None

    cfg = OmegaConf.create(
        TessellateExtractFeaturesConfig(
            **_TEF_REQUIRED, seg_config=SegConfig(), slide_model_type=slide_model_type
        )
    )
    with (
        patch(
            "mussel.cli.tessellate_extract_features.process_slide_tessellation_and_filtering",
            side_effect=capture,
        ),
        patch(
            "mussel.cli.tessellate_extract_features._build_segmentation_runtime",
            return_value=(None, None),
        ),
    ):
        main(cfg)
    assert seen == {"tiles": tiles, "model_type": patch_encoder}


@pytest.mark.parametrize("slide_model_type,patch_encoder,tiles", _SLIDE_ENCODER_TILES)
def test_batch_tiles_for_the_slide_encoder(
    tmp_path, slide_model_type, patch_encoder, tiles
):
    """_main_batch keeps the slide encoder's tiles, not its patch encoder's."""
    seen = []

    def capture(**kwargs):
        cfg = kwargs["cfg"]
        seen.append((cfg.seg_config.patch_size, cfg.seg_config.mpp))
        return None

    cfg = OmegaConf.create(
        TessellateExtractFeaturesConfig(
            slide_paths=["a.svs", "b.svs"],
            output_dir=str(tmp_path),
            seg_config=SegConfig(),
            slide_model_type=slide_model_type,
        )
    )
    with (
        patch(
            "mussel.cli.tessellate_extract_features.process_slide_tessellation_only",
            side_effect=capture,
        ),
        patch(
            "mussel.cli.tessellate_extract_features._build_segmentation_runtime",
            return_value=(None, None),
        ),
    ):
        main(cfg)
    assert seen == [tiles, tiles]


@pytest.mark.parametrize(
    "model_type,slide_model_type,expected",
    [
        (
            ModelType.CTRANSPATH,
            ModelType.CHIEF_SLIDE,
            {
                (ModelType.CTRANSPATH, None): (224, 0.5),
                (ModelType.CTRANSPATH, ModelType.CHIEF_SLIDE): (256, 1.0),
            },
        ),
        (
            ModelType.CONCH_V1,
            ModelType.MADELEINE_SLIDE,
            {
                (ModelType.CONCH_V1, None): (448, 0.5),
                (ModelType.CONCH_V1, ModelType.MADELEINE_SLIDE): (256, 1.0),
            },
        ),
        # Same tile size, different MPP: two tessellations.
        (
            ["UNI"],
            ["CHIEF_SLIDE"],
            {
                (ModelType.UNI, None): (256, 0.5),
                (ModelType.CTRANSPATH, ModelType.CHIEF_SLIDE): (256, 1.0),
            },
        ),
        # ABMIL_SLIDE takes the one patch model given, and its tiles.
        (
            ModelType.UNI,
            ModelType.ABMIL_SLIDE,
            {
                (ModelType.UNI, None): (256, 0.5),
                (ModelType.UNI, ModelType.ABMIL_SLIDE): (256, 0.5),
            },
        ),
        (
            ModelType.CONCH1_5,
            ModelType.ABMIL_SLIDE,
            {
                (ModelType.CONCH1_5, None): (512, 0.5),
                (ModelType.CONCH1_5, ModelType.ABMIL_SLIDE): (512, 0.5),
            },
        ),
        # A patch model without a target MPP gets the default, not the slide encoder's.
        (
            ModelType.VIRCHOW2,
            ModelType.MADELEINE_SLIDE,
            {
                (ModelType.VIRCHOW2, None): (224, 0.5),
                (ModelType.CONCH_V1, ModelType.MADELEINE_SLIDE): (256, 1.0),
            },
        ),
    ],
)
def test_multi_model_tiles_for_each_model(
    tmp_path, model_type, slide_model_type, expected
):
    """Multi-model mode groups slide encoders by their own tile size and MPP."""
    seen = _run_multi_model(tmp_path, model_type, slide_model_type)
    assert {k: v["tiles"] for k, v in seen.items()} == expected


@pytest.mark.parametrize(
    "model_type,slide_model_type,mpp,expected",
    [
        (
            ModelType.CTRANSPATH,
            ModelType.CHIEF_SLIDE,
            0.75,
            {
                (ModelType.CTRANSPATH, None): (224, 0.75),
                (ModelType.CTRANSPATH, ModelType.CHIEF_SLIDE): (256, 0.75),
            },
        ),
        (
            ["CONCH1_5", "VIRCHOW2"],
            ["TITAN_SLIDE"],
            0.25,
            {
                (ModelType.CONCH1_5, None): (512, 0.25),
                (ModelType.VIRCHOW2, None): (224, 0.25),
                (ModelType.CONCH1_5, ModelType.TITAN_SLIDE): (512, 0.25),
            },
        ),
    ],
)
def test_multi_model_keeps_explicit_mpp(
    tmp_path, model_type, slide_model_type, mpp, expected
):
    """An explicit seg_config.mpp is used for every model, target MPP or not."""
    seen = _run_multi_model(tmp_path, model_type, slide_model_type, SegConfig(mpp=mpp))
    assert {k: v["tiles"] for k, v in seen.items()} == expected


def _run_multi_model(tmp_path, model_type, slide_model_type, seg_config=None):
    """Tiles and output dirs main() hands _main_batch per (model, slide model)."""
    seen = {}

    def capture(cfg, patch_output_dir=None):
        seen[(cfg.model_type, cfg.slide_model_type)] = {
            "tiles": (cfg.seg_config.patch_size, cfg.seg_config.mpp),
            "output_dir": os.path.relpath(cfg.output_dir, tmp_path),
            "patch_output_dir": patch_output_dir
            and os.path.relpath(patch_output_dir, tmp_path),
        }

    cfg = OmegaConf.create(
        TessellateExtractFeaturesConfig(
            slide_paths=["a.svs"],
            output_dir=str(tmp_path),
            seg_config=seg_config or SegConfig(),
            model_type=model_type,
            slide_model_type=slide_model_type,
        )
    )
    with patch(
        "mussel.cli.tessellate_extract_features._main_batch", side_effect=capture
    ):
        main(cfg)
    return seen


@pytest.mark.parametrize(
    "model_type,slide_model_type,expected",
    [
        # Other tiles than the patch model's: patch encoder outputs get their own dir.
        (
            ModelType.CTRANSPATH,
            ModelType.CHIEF_SLIDE,
            {
                (ModelType.CTRANSPATH, None): ("CTRANSPATH", None),
                (ModelType.CTRANSPATH, ModelType.CHIEF_SLIDE): (
                    "CHIEF_SLIDE",
                    "CTRANSPATH_256px_1mpp",
                ),
            },
        ),
        (
            ModelType.CONCH_V1,
            ModelType.MADELEINE_SLIDE,
            {
                (ModelType.CONCH_V1, None): ("CONCH_V1", None),
                (ModelType.CONCH_V1, ModelType.MADELEINE_SLIDE): (
                    "MADELEINE_SLIDE",
                    "CONCH_V1_256px_1mpp",
                ),
            },
        ),
        # Same tiles: the patch encoder's outputs are shared, as before.
        (
            ModelType.CONCH1_5,
            ModelType.TITAN_SLIDE,
            {
                (ModelType.CONCH1_5, None): ("CONCH1_5", None),
                (ModelType.CONCH1_5, ModelType.TITAN_SLIDE): ("TITAN_SLIDE", "CONCH1_5"),
            },
        ),
        (
            ModelType.CONCH1_5,
            ModelType.ABMIL_SLIDE,
            {
                (ModelType.CONCH1_5, None): ("CONCH1_5", None),
                (ModelType.CONCH1_5, ModelType.ABMIL_SLIDE): ("ABMIL_SLIDE", "CONCH1_5"),
            },
        ),
    ],
)
def test_multi_model_patch_outputs_do_not_collide(
    tmp_path, model_type, slide_model_type, expected
):
    """A slide encoder's patch features of other tiles don't overwrite the patch model's."""
    seen = _run_multi_model(tmp_path, model_type, slide_model_type)
    assert {
        k: (v["output_dir"], v["patch_output_dir"]) for k, v in seen.items()
    } == expected


def test_single_slide_filter_fast_path_keeps_the_slide_encoder(tmp_path):
    """With a slide encoder, filtering with its own patch encoder still runs the slide encoder."""
    seen = {}

    def capture(**kwargs):
        seen["skip_second_extraction"] = kwargs["skip_second_extraction"]
        seen["model_type"] = kwargs["model_type"]
        return None

    classifier = tmp_path / "classifier.pkl"
    classifier.touch()
    cfg = OmegaConf.create(
        TessellateExtractFeaturesConfig(
            **_TEF_REQUIRED,
            seg_config=SegConfig(),
            classifier_pkl=str(classifier),
            prefilter_model_type=ModelType.CONCH1_5,
            slide_model_type=ModelType.TITAN_SLIDE,
        )
    )
    with (
        patch(
            "mussel.cli.tessellate_extract_features.process_slide_tessellation_and_filtering",
            side_effect=capture,
        ),
        patch(
            "mussel.cli.tessellate_extract_features._build_segmentation_runtime",
            return_value=(None, None),
        ),
    ):
        main(cfg)
    assert seen == {"skip_second_extraction": False, "model_type": ModelType.CONCH1_5}


def test_neural_segmenter_use_gpu_false_forces_cpu():
    seg_cfg = {"seg_model": "neural"}

    with patch("mussel.utils.neural_seg.NeuralTissueSegmenter") as MockSegmenter:
        result = _build_neural_segmenter(seg_cfg, use_gpu=False, gpu_device_id=1)

    MockSegmenter.assert_called_once_with(device="cpu")
    assert result is MockSegmenter.return_value


def test_explicit_neural_device_overrides_use_gpu():
    seg_cfg = {"seg_model": "neural"}
    neural_cfg = {"device": "cpu", "batch_size": 16}

    with patch("mussel.utils.neural_seg.NeuralTissueSegmenter") as MockSegmenter:
        result = _build_neural_segmenter(
            seg_cfg, use_gpu=True, neural_config=neural_cfg
        )

    MockSegmenter.assert_called_once_with(**neural_cfg)
    assert result is MockSegmenter.return_value


def test_explicit_neural_cuda_device_overrides_cpu_workflow():
    seg_cfg = {"seg_model": "neural"}
    neural_cfg = {"device": "cuda:2"}

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("mussel.utils.neural_seg.NeuralTissueSegmenter") as MockSegmenter,
    ):
        result = _build_neural_segmenter(
            seg_cfg, use_gpu=False, neural_config=neural_cfg
        )

    MockSegmenter.assert_called_once_with(**neural_cfg)
    assert result is MockSegmenter.return_value


def test_neural_segmenter_use_gpu_true_requires_cuda():
    seg_cfg = {"seg_model": "neural"}

    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            _build_neural_segmenter(seg_cfg, use_gpu=True)


@pytest.mark.parametrize(
    ("kwargs", "expected_device"),
    [
        ({}, "cuda"),
        ({"gpu_device_id": 2}, "cuda:2"),
        ({"gpu_device_ids": [3, 4]}, "cuda:3"),
    ],
)
def test_neural_segmenter_use_gpu_true_uses_cuda_device(kwargs, expected_device):
    seg_cfg = {"seg_model": "neural"}

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("mussel.utils.neural_seg.NeuralTissueSegmenter") as MockSegmenter,
    ):
        result = _build_neural_segmenter(seg_cfg, use_gpu=True, **kwargs)

    MockSegmenter.assert_called_once_with(device=expected_device)
    assert result is MockSegmenter.return_value


# The integration tests check the workflow, not throughput: a seeded random 300
# tiles of the test slide instead of all of them (segment_threshold=0). An
# explicit 224 px (not the 256 px default, which a model's size overrides) gives
# every run in a test the same tiles, whichever models it uses.
_SEG = dict(segment_threshold=0, max_tiles=300, patch_size=224)


def _run_single(output_dir, test_data_path, use_gpu, num_workers, **overrides):
    """Run tessellate_extract_features on the test slide; return (features, coords) of its .h5.

    Also checks the .pt holds the same features as the .h5.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_h5_path = os.path.join(output_dir, "features.h5")
    output_pt_path = os.path.join(output_dir, "features.pt")
    cfg = TessellateExtractFeaturesConfig(**{
        **dict(
            slide_path=os.path.join(test_data_path, "948176.svs"),
            output_h5_path=output_h5_path,
            output_pt_path=output_pt_path,
            classifier_pkl=None,
            prefilter_model_type=ModelType.RESNET50,
            seg_config=SegConfig(**_SEG),
            num_workers=num_workers,
            batch_size=32,
            use_gpu=use_gpu,
            keep_intermediate_files=False,
        ),
        **overrides,
    })
    main(OmegaConf.create(cfg))
    with h5py.File(output_h5_path, "r") as f:
        features, coords = f["features"][:], f["coords"][:]
    pt = torch.load(output_pt_path, weights_only=True)
    np.testing.assert_allclose(pt.float().numpy(), features, rtol=1e-6, err_msg=".pt differs from .h5")
    return features, coords


def _assert_same_features(actual, expected, what):
    """Same tiles' features from two runs of one model, up to GPU noise (1% of each row's norm)."""
    assert actual.shape == expected.shape, f"{what}: shape {actual.shape} != {expected.shape}"
    err = np.linalg.norm(actual - expected, axis=1) / np.maximum(np.linalg.norm(expected, axis=1), 1e-12)
    assert err.max() <= 1e-2, f"{what}: features differ by up to {err.max():.2e} of the row norm"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_tessellate_extract_features(tmp_path, test_data_path, use_gpu, num_workers, cut_classifier):
    """Filtering with the pre-filter model as the final model keeps exactly the passing tiles."""
    all_feats, all_coords = _run_single(str(tmp_path / "all"), test_data_path, use_gpu, num_workers)
    classifier_pkl, keep = cut_classifier(all_feats)
    kept = keep(all_feats)
    assert 0.25 * len(kept) <= kept.sum() <= 0.75 * len(kept)

    feats, coords = _run_single(str(tmp_path / "filtered"), test_data_path, use_gpu, num_workers,
                                classifier_pkl=classifier_pkl, classifier_threshold=0.5, model_type=None)
    np.testing.assert_array_equal(coords, all_coords[kept], err_msg="wrong tiles kept")
    _assert_same_features(feats, all_feats[kept], "RESNET50 features of the kept tiles")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_tessellate_extract_features_with_different_models(
    tmp_path, test_data_path, use_gpu, num_workers, cut_classifier
):
    """Filter with ResNet50, extract the kept tiles with a second model (KAIKO ViT-S/16)."""
    final = ModelType.KAIKO_VITS16
    resnet_feats, all_coords = _run_single(str(tmp_path / "resnet"), test_data_path, use_gpu, num_workers)
    final_feats, final_coords = _run_single(str(tmp_path / "final"), test_data_path, use_gpu, num_workers,
                                            prefilter_model_type=final)
    np.testing.assert_array_equal(final_coords, all_coords, err_msg="the two models' runs tiled differently")
    classifier_pkl, keep = cut_classifier(resnet_feats)
    kept = keep(resnet_feats)

    feats, coords = _run_single(str(tmp_path / "filtered"), test_data_path, use_gpu, num_workers,
                                classifier_pkl=classifier_pkl, classifier_threshold=0.5,
                                prefilter_model_type=ModelType.RESNET50, model_type=final)
    np.testing.assert_array_equal(coords, all_coords[kept], err_msg="wrong tiles kept")
    _assert_same_features(feats, final_feats[kept], f"{final.name} features of the kept tiles")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_tessellate_extract_features_with_intermediate_files(
    tmp_path, test_data_path, classifier_pkl_path, use_gpu, num_workers
):
    """Test the integrated workflow while keeping intermediate files."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    classifier_pkl = classifier_pkl_path
    output_h5_path = os.path.join(tmp_path, "features.h5")
    output_pt_path = os.path.join(tmp_path, "features.pt")

    seg_config = SegConfig(**_SEG)
    cfg = TessellateExtractFeaturesConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=classifier_pkl,
        classifier_threshold=0.75,
        prefilter_model_type=ModelType.RESNET50,
        seg_config=seg_config,
        num_workers=num_workers,
        batch_size=32,
        use_gpu=use_gpu,
        keep_intermediate_files=True,  # Keep intermediate files
    )

    main(OmegaConf.create(cfg))

    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)

    # Check that intermediate files were created
    base_path = Path(output_h5_path).parent
    tessellate_h5_path = base_path / f"{Path(slide_path).stem}.tessellate.h5"
    prefilter_features_h5_path = (
        base_path / f"{Path(slide_path).stem}.prefilter_features.h5"
    )

    assert os.path.exists(tessellate_h5_path)
    assert os.path.exists(prefilter_features_h5_path)
    # filtered_coords.h5 is only created when prefilter_model_type != model_type
    # (i.e. when a second extraction pass is needed). With the same model, the
    # prefilter_features.h5 already contains the filtered coordinates.


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_tessellate_extract_features_with_visualizations(
    tmp_path, test_data_path, classifier_pkl_path, use_gpu, num_workers
):
    """Test the workflow with optional visualization outputs."""
    slide_path = os.path.join(test_data_path, "948176.svs")
    classifier_pkl = classifier_pkl_path
    output_h5_path = os.path.join(tmp_path, "features.h5")
    output_pt_path = os.path.join(tmp_path, "features.pt")
    output_mask_path = os.path.join(tmp_path, "mask.png")
    output_grid_mask_path = os.path.join(tmp_path, "grid.png")
    output_thumbnail_path = os.path.join(tmp_path, "thumbnail.png")

    seg_config = SegConfig(**_SEG)
    cfg = TessellateExtractFeaturesConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=classifier_pkl,
        classifier_threshold=0.75,
        output_mask_path=output_mask_path,
        output_grid_mask_path=output_grid_mask_path,
        output_thumbnail_path=output_thumbnail_path,
        prefilter_model_type=ModelType.RESNET50,
        seg_config=seg_config,
        num_workers=num_workers,
        batch_size=32,
        use_gpu=use_gpu,
        keep_intermediate_files=False,
    )

    main(OmegaConf.create(cfg))

    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)
    assert os.path.exists(output_mask_path)
    assert os.path.exists(output_grid_mask_path)
    assert os.path.exists(output_thumbnail_path)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_tessellate_extract_features_without_filtering(tmp_path, test_data_path, use_gpu, num_workers):
    """Without a classifier every tessellated tile is extracted, once."""
    feats, coords = _run_single(str(tmp_path / "out"), test_data_path, use_gpu, num_workers)
    assert feats.shape == (_SEG["max_tiles"], 1024), "expected one RESNET50 row per tile"
    assert len(np.unique(coords, axis=0)) == len(coords), "a tile was extracted twice"
    assert np.all(np.isfinite(feats))


def test_tessellate_extract_features_with_slide_encoder_inference(
    tmp_path, test_data_path, use_gpu, num_workers
):
    """Test that model_type is inferred from slide_model_type when using model aggregation."""

    slide_path = os.path.join(test_data_path, "948176.svs")
    output_h5_path = os.path.join(tmp_path, "features.h5")
    output_pt_path = os.path.join(tmp_path, "features.pt")

    seg_config = SegConfig(**_SEG)
    cfg = TessellateExtractFeaturesConfig(
        slide_path=slide_path,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=None,  # No filtering
        prefilter_model_type=ModelType.RESNET50,
        # Note: model_type is NOT specified
        model_type=None,
        aggregation_method="model",
        slide_model_type=ModelType.GIGAPATH_SLIDE,  # Requires GIGAPATH patch encoder
        seg_config=seg_config,
        num_workers=num_workers,
        batch_size=32,
        use_gpu=use_gpu,
        keep_intermediate_files=False,
    )

    # Mock save_features to capture the model_type parameter
    with (
        patch(
            "mussel.cli.tessellate_extract_features_common.save_features"
        ) as mock_save_features,
        patch(
            "mussel.cli.tessellate_extract_features_common.segment_tissue"
        ) as mock_segment,
    ):

        # Mock segment_tissue to return fake data
        mock_coords = [[0, 0], [256, 0], [0, 256]]
        mock_polygon = MagicMock()
        mock_grid = MagicMock()
        mock_segment.return_value = (mock_polygon, mock_grid, mock_coords, None)

        # Run main
        main(OmegaConf.create(cfg))

        # Verify save_features was called with the correct model_type
        assert mock_save_features.called
        call_args = mock_save_features.call_args

        # The model_type should be GIGAPATH (inferred from GIGAPATH_SLIDE)
        assert (
            call_args.kwargs["model_type"] == ModelType.GIGAPATH
        ), f"Expected model_type to be GIGAPATH (inferred from slide_model_type), got {call_args.kwargs['model_type']}"
        assert call_args.kwargs["slide_model_type"] == ModelType.GIGAPATH_SLIDE
        assert call_args.kwargs["aggregation_method"] == "model"
