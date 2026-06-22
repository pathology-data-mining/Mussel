"""Tests for get_features pre-loaded model parameters."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import ml_dtypes
import numpy as np
import pytest

from mussel.cli.extract_features import ExtractFeaturesConfig
from mussel.models import ModelType
from mussel.utils.feature_extract import (
    H5DatasetProcessor,
    TileCoordProcessor,
    _parse_feature_dtype,
    get_features,
)


def _make_mock_model(feature_dim=384):
    """Return a mock model with the required interface."""
    mock = MagicMock()
    mock.get_preprocessing_fun.return_value = None
    mock.get_model_fun.return_value = MagicMock(
        side_effect=lambda x: __import__("torch").randn(len(x), feature_dim)
    )
    return mock


def _base_patches():
    """Context managers that stub out all I/O in get_features."""
    return [
        patch("mussel.utils.feature_extract.get_model_factory"),
        patch("mussel.utils.feature_extract.WholeSlideImageTileCoordDataset"),
        patch("mussel.utils.feature_extract.process_dataset"),
        patch("mussel.utils.feature_extract._make_dataloader"),
    ]


# -- model= parameter -------------------------------------------------------


def test_model_factory_not_called_when_model_provided():
    """get_model_factory must not be called when a pre-loaded model is given."""
    coords = np.zeros((10, 2), dtype=np.int32)
    attrs = {
        "patch_size": 256,
        "patch_level": 0,
        "mpp": 0.5,
        "patch_size_to_resize_to_for_desired_mpp": 224,
    }
    mock_model = _make_mock_model()

    with (
        patch("mussel.utils.feature_extract.get_model_factory") as mock_factory,
        patch("mussel.utils.feature_extract.WholeSlideImageTileCoordDataset"),
        patch("mussel.utils.feature_extract._make_dataloader"),
        patch("mussel.utils.feature_extract.process_dataset") as mock_proc,
    ):
        mock_proc.return_value = MagicMock(
            features=np.zeros((10, 384)), labels=np.zeros(10)
        )
        get_features(coords, "slide.svs", attrs, model=mock_model)

    mock_factory.assert_not_called()


def test_model_factory_called_when_model_not_provided():
    """get_model_factory must be called when no pre-loaded model is given."""
    coords = np.zeros((10, 2), dtype=np.int32)
    attrs = {
        "patch_size": 256,
        "patch_level": 0,
        "mpp": 0.5,
        "patch_size_to_resize_to_for_desired_mpp": 224,
    }

    with (
        patch("mussel.utils.feature_extract.get_model_factory") as mock_factory,
        patch("mussel.utils.feature_extract.WholeSlideImageTileCoordDataset"),
        patch("mussel.utils.feature_extract._make_dataloader"),
        patch("mussel.utils.feature_extract.process_dataset") as mock_proc,
    ):
        mock_model = _make_mock_model()
        mock_factory.return_value = MagicMock(
            get_model=MagicMock(return_value=mock_model)
        )
        mock_proc.return_value = MagicMock(
            features=np.zeros((10, 384)), labels=np.zeros(10)
        )
        get_features(coords, "slide.svs", attrs, model_type=ModelType.CTRANSPATH)

    mock_factory.assert_called_once()


def test_positional_args_unchanged():
    """Existing positional call pattern must still work after adding model= at end."""
    coords = np.zeros((5, 2), dtype=np.int32)
    attrs = {
        "patch_size": 256,
        "patch_level": 0,
        "mpp": 0.5,
        "patch_size_to_resize_to_for_desired_mpp": 224,
    }

    with (
        patch("mussel.utils.feature_extract.get_model_factory") as mock_factory,
        patch("mussel.utils.feature_extract.WholeSlideImageTileCoordDataset"),
        patch("mussel.utils.feature_extract._make_dataloader"),
        patch("mussel.utils.feature_extract.process_dataset") as mock_proc,
    ):
        mock_model = _make_mock_model()
        mock_factory.return_value = MagicMock(
            get_model=MagicMock(return_value=mock_model)
        )
        mock_proc.return_value = MagicMock(
            features=np.zeros((5, 384)), labels=np.zeros(5)
        )
        # Classic positional call: (coords, slide_path, attrs, model_type, model_path, batch_size)
        # batch_size must not be interpreted as model=
        features, labels = get_features(
            coords, "slide.svs", attrs, ModelType.CTRANSPATH, None, 32
        )
    assert features.shape == (5, 384)


def test_model_invalid_interface_raises_type_error():
    """Passing an object without the required methods must raise TypeError immediately."""
    coords = np.zeros((5, 2), dtype=np.int32)
    attrs = {
        "patch_size": 256,
        "patch_level": 0,
        "mpp": 0.5,
        "patch_size_to_resize_to_for_desired_mpp": 224,
    }
    bad_model = object()

    with pytest.raises(TypeError, match="get_preprocessing_fun"):
        get_features(coords, "slide.svs", attrs, model=bad_model)


# -- slide_model= parameter ------------------------------------------------


def test_slide_model_factory_not_called_when_slide_model_provided():
    """get_model_factory must not be loaded for the slide encoder when slide_model is given."""
    coords = np.zeros((10, 2), dtype=np.int32)
    attrs = {
        "patch_size": 256,
        "patch_level": 0,
        "mpp": 0.5,
        "patch_size_to_resize_to_for_desired_mpp": 224,
    }
    mock_patch_model = _make_mock_model()
    mock_slide_model = MagicMock()
    mock_slide_model.get_model_fun.return_value = MagicMock(
        return_value=__import__("torch").zeros(1, 512)
    )

    call_log = []

    def factory_side_effect(model_type):
        call_log.append(model_type)
        m = MagicMock()
        m.get_model.return_value = mock_patch_model
        return m

    with (
        patch(
            "mussel.utils.feature_extract.get_model_factory",
            side_effect=factory_side_effect,
        ),
        patch("mussel.utils.feature_extract.validate_slide_encoder_compatibility"),
        patch("mussel.utils.feature_extract.WholeSlideImageTileCoordDataset"),
        patch("mussel.utils.feature_extract._make_dataloader"),
        patch("mussel.utils.feature_extract.process_dataset") as mock_proc,
    ):
        mock_proc.return_value = MagicMock(
            features=np.zeros((10, 384)), labels=np.zeros(10)
        )
        get_features(
            coords,
            "slide.svs",
            attrs,
            model=mock_patch_model,
            use_slide_encoder=True,
            slide_model_type=ModelType.GIGAPATH_SLIDE,
            aggregation_method="model",
            slide_model=mock_slide_model,
        )

    # Only the patch encoder factory may be called (for auto-infer check), not the slide encoder
    slide_encoder_calls = [t for t in call_log if t == ModelType.GIGAPATH_SLIDE]
    assert (
        len(slide_encoder_calls) == 0
    ), "Slide encoder factory must not be called when slide_model is provided"


def test_slide_model_invalid_interface_raises_type_error():
    """Passing a slide_model without get_model_fun must raise TypeError."""
    coords = np.zeros((5, 2), dtype=np.int32)
    attrs = {
        "patch_size": 256,
        "patch_level": 0,
        "mpp": 0.5,
        "patch_size_to_resize_to_for_desired_mpp": 224,
    }
    bad_slide_model = object()

    with pytest.raises(TypeError, match="get_model_fun"):
        get_features(coords, "slide.svs", attrs, slide_model=bad_slide_model)


# -- compatibility validation ----b??-----------------------------------------


def test_compatibility_validated_with_preloaded_patch_model():
    """validate_slide_encoder_compatibility must be called even with pre-loaded patch model."""
    coords = np.zeros((5, 2), dtype=np.int32)
    attrs = {
        "patch_size": 256,
        "patch_level": 0,
        "mpp": 0.5,
        "patch_size_to_resize_to_for_desired_mpp": 224,
    }
    mock_patch_model = _make_mock_model()

    with (
        patch(
            "mussel.utils.feature_extract.validate_slide_encoder_compatibility"
        ) as mock_validate,
        patch("mussel.utils.feature_extract.WholeSlideImageTileCoordDataset"),
        patch("mussel.utils.feature_extract._make_dataloader"),
        patch("mussel.utils.feature_extract.process_dataset") as mock_proc,
    ):
        mock_proc.return_value = MagicMock(
            features=np.zeros((5, 384)), labels=np.zeros(5)
        )
        mock_slide = MagicMock()
        mock_slide.get_model_fun.return_value = MagicMock(
            return_value=__import__("torch").zeros(1, 512)
        )
        get_features(
            coords,
            "slide.svs",
            attrs,
            model_type=ModelType.GIGAPATH,
            model=mock_patch_model,
            use_slide_encoder=True,
            slide_model_type=ModelType.GIGAPATH_SLIDE,
            aggregation_method="model",
            slide_model=mock_slide,
        )

    mock_validate.assert_called_once_with(ModelType.GIGAPATH, ModelType.GIGAPATH_SLIDE)


# -- embedding precision tests -----------------------------------------------


def test_parse_feature_dtype_float32_returns_none():
    assert _parse_feature_dtype("float32") is None


def test_parse_feature_dtype_float16():
    assert _parse_feature_dtype("float16") == np.dtype(np.float16)


def test_parse_feature_dtype_bfloat16():
    assert _parse_feature_dtype("bfloat16") == np.dtype(ml_dtypes.bfloat16)


def test_parse_feature_dtype_invalid_raises():
    with pytest.raises(ValueError, match="Unsupported embedding_precision"):
        _parse_feature_dtype("float8")


def _make_h5_processor_inputs(feature_dim=4, n_patches=8, batch_size=4):
    """Return (mock_dataset, mock_loader, mock_model_fun) for H5DatasetProcessor tests."""
    import torch

    batches = []
    n_batches = n_patches // batch_size
    for _ in range(n_batches):
        features_tensor = torch.zeros(batch_size, feature_dim)
        coords_np = np.zeros((batch_size, 2), dtype=np.int32)
        batches.append((features_tensor, coords_np))

    mock_dataset = MagicMock()
    mock_loader = batches

    def mock_model_fun(batch):
        return torch.zeros(len(batch), feature_dim)

    return mock_dataset, mock_loader, mock_model_fun


def test_h5_dataset_processor_saves_float16():
    """H5DatasetProcessor with feature_dtype=np.float16 writes float16 to HDF5."""
    import h5py

    mock_dataset, mock_loader, mock_model_fun = _make_h5_processor_inputs()
    processor = H5DatasetProcessor()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "feats.h5")
        result = processor.process(
            dataset=mock_dataset,
            loader=mock_loader,
            model_fun=mock_model_fun,
            output_h5_path=out_path,
            is_test_run=False,
            feature_dtype=np.float16,
        )
        assert result.features.dtype == np.float16
        with h5py.File(out_path, "r") as f:
            assert f["features"].dtype == np.float16


def test_h5_dataset_processor_default_float32():
    """H5DatasetProcessor with feature_dtype=None keeps float32."""
    import h5py

    mock_dataset, mock_loader, mock_model_fun = _make_h5_processor_inputs()
    processor = H5DatasetProcessor()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "feats.h5")
        result = processor.process(
            dataset=mock_dataset,
            loader=mock_loader,
            model_fun=mock_model_fun,
            output_h5_path=out_path,
            is_test_run=False,
            feature_dtype=None,
        )
        assert result.features.dtype == np.float32
        with h5py.File(out_path, "r") as f:
            assert f["features"].dtype == np.float32


def test_tile_coord_processor_float16():
    """TileCoordProcessor with feature_dtype=np.float16 casts output after concat."""
    import torch

    batches = [(torch.zeros(4, 4), torch.zeros(4, dtype=torch.long))] * 2
    mock_dataset = MagicMock()
    mock_loader = batches

    def mock_model_fun(batch):
        return torch.zeros(len(batch), 4)

    processor = TileCoordProcessor()
    result = processor.process(
        dataset=mock_dataset,
        loader=mock_loader,
        model_fun=mock_model_fun,
        feature_dtype=np.float16,
    )
    assert result.features.dtype == np.float16


def test_h5_dataset_processor_saves_bfloat16():
    """H5DatasetProcessor with bfloat16 keeps in-memory dtype; HDF5 stores as 2-byte void."""
    import h5py

    bfloat16 = np.dtype(ml_dtypes.bfloat16)
    mock_dataset, mock_loader, mock_model_fun = _make_h5_processor_inputs()
    processor = H5DatasetProcessor()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "feats.h5")
        result = processor.process(
            dataset=mock_dataset,
            loader=mock_loader,
            model_fun=mock_model_fun,
            output_h5_path=out_path,
            is_test_run=False,
            feature_dtype=bfloat16,
        )
        # In-memory result preserves bfloat16 dtype.
        assert result.features.dtype == bfloat16
        # h5py stores bfloat16 as a 2-byte opaque type (|V2); verify storage size.
        with h5py.File(out_path, "r") as f:
            assert f["features"].dtype.itemsize == 2


def test_tile_coord_processor_bfloat16():
    """TileCoordProcessor with feature_dtype=ml_dtypes.bfloat16 casts output after concat."""
    import torch

    bfloat16 = np.dtype(ml_dtypes.bfloat16)
    batches = [(torch.zeros(4, 4), torch.zeros(4, dtype=torch.long))] * 2
    mock_dataset = MagicMock()
    mock_loader = batches

    def mock_model_fun(batch):
        return torch.zeros(len(batch), 4)

    processor = TileCoordProcessor()
    result = processor.process(
        dataset=mock_dataset,
        loader=mock_loader,
        model_fun=mock_model_fun,
        feature_dtype=bfloat16,
    )
    assert result.features.dtype == bfloat16


def test_extract_features_config_has_embedding_precision_field():
    """ExtractFeaturesConfig must include precision and model kwargs defaults."""
    cfg = ExtractFeaturesConfig()
    assert hasattr(cfg, "embedding_precision")
    assert cfg.embedding_precision == "float32"
    assert cfg.model_kwargs == {}
    assert cfg.slide_model_kwargs == {}


# -- slide pipeline precision semantics ----------------------------------------


def test_aggregate_slide_features_precision():
    """aggregate_slide_features saves output at the requested precision."""
    import h5py
    import tempfile
    from mussel.utils.feature_extract import aggregate_slide_features

    features = np.random.rand(10, 8).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        patch_h5 = os.path.join(tmpdir, "patches.h5")
        with h5py.File(patch_h5, "w") as f:
            f.create_dataset("features", data=features)

        for precision, expected_itemsize, expected_kind in [
            ("float32", 4, "f"),
            ("float16", 2, "f"),
            ("bfloat16", 2, "V"),
        ]:
            out_h5 = os.path.join(tmpdir, f"slide_{precision}.h5")
            aggregate_slide_features(
                patch_features_h5_path=patch_h5,
                output_h5_path=out_h5,
                aggregation_method="mean",
                embedding_precision=precision,
            )
            with h5py.File(out_h5, "r") as f:
                dtype = f["features"].dtype
                assert (
                    dtype.itemsize == expected_itemsize
                ), f"precision={precision}: expected itemsize {expected_itemsize}, got {dtype.itemsize}"
                if expected_kind != "V":
                    assert (
                        dtype.kind == expected_kind
                    ), f"precision={precision}: expected kind {expected_kind!r}, got {dtype.kind!r}"


def test_save_features_two_step_keeps_intermediate_float32():
    """In two-step (slide model) path, intermediate tile features must remain float32."""
    import h5py
    import tempfile
    from mussel.utils.feature_extract import _parse_feature_dtype, save_features

    features = np.random.rand(4, 8).astype(np.float32)
    coords = np.zeros((4, 2), dtype=np.int32)

    with tempfile.TemporaryDirectory() as tmpdir:
        patch_h5 = os.path.join(tmpdir, "patches.h5")
        with h5py.File(patch_h5, "w") as f:
            f.create_dataset("features", data=features)
            f.create_dataset("coords", data=coords)
            f["features"].attrs["patch_size"] = 256
            f["features"].attrs["patch_level"] = 0
            f["features"].attrs["patch_size_to_resize_to_for_desired_mpp"] = 224

        intermediate_calls = []

        def fake_extract_patch(
            patch_h5_path,
            slide_path,
            output_h5_path,
            embedding_precision="float32",
            **kwargs,
        ):
            intermediate_calls.append(embedding_precision)
            with (
                h5py.File(patch_h5_path, "r") as src,
                h5py.File(output_h5_path, "w") as dst,
            ):
                dst.create_dataset("features", data=src["features"][:])
                dst.create_dataset("coords", data=src["coords"][:])

        def fake_aggregate(
            patch_features_h5_path,
            output_h5_path,
            embedding_precision="float32",
            **kwargs,
        ):
            feature_dtype = _parse_feature_dtype(embedding_precision)
            with h5py.File(patch_features_h5_path, "r") as src:
                data = src["features"][:]
            if feature_dtype is not None:
                data = data.astype(feature_dtype)
            if output_h5_path:
                with h5py.File(output_h5_path, "w") as dst:
                    dst.create_dataset("features", data=data)
            return output_h5_path, None

        out_h5 = os.path.join(tmpdir, "slide_out.h5")

        with (
            patch(
                "mussel.utils.feature_extract.extract_patch_features",
                side_effect=fake_extract_patch,
            ),
            patch(
                "mussel.utils.feature_extract.aggregate_slide_features",
                side_effect=fake_aggregate,
            ),
        ):
            save_features(
                patch_h5_path=patch_h5,
                slide_path="dummy.svs",
                output_h5_path=out_h5,
                aggregation_method="model",
                embedding_precision="float16",
            )

        assert intermediate_calls == [
            "float32"
        ], f"Intermediate tile extraction must use float32, got {intermediate_calls}"
        with h5py.File(out_h5, "r") as f:
            assert f["features"].dtype.itemsize == 2
            assert f["features"].dtype.kind == "f"  # float16


def test_save_features_forwards_model_kwargs_to_correct_steps():
    """save_features forwards patch and slide model kwargs to their respective stages."""
    import h5py
    import tempfile
    from mussel.utils.feature_extract import save_features

    features = np.random.rand(4, 8).astype(np.float32)
    coords = np.zeros((4, 2), dtype=np.int32)

    with tempfile.TemporaryDirectory() as tmpdir:
        patch_h5 = os.path.join(tmpdir, "patches.h5")
        with h5py.File(patch_h5, "w") as f:
            f.create_dataset("features", data=features)
            f.create_dataset("coords", data=coords)
            f["features"].attrs["patch_size"] = 256

        seen_model_kwargs = []
        seen_slide_model_kwargs = []

        def fake_extract_patch(patch_h5_path, slide_path, output_h5_path, **kwargs):
            seen_model_kwargs.append(kwargs["model_kwargs"])
            with (
                h5py.File(patch_h5_path, "r") as src,
                h5py.File(output_h5_path, "w") as dst,
            ):
                dst.create_dataset("features", data=src["features"][:])
                dst.create_dataset("coords", data=src["coords"][:])
                dst["features"].attrs["patch_size"] = src["features"].attrs[
                    "patch_size"
                ]

        def fake_aggregate(patch_features_h5_path, output_h5_path, **kwargs):
            seen_slide_model_kwargs.append(kwargs["slide_model_kwargs"])
            with (
                h5py.File(patch_features_h5_path, "r") as src,
                h5py.File(output_h5_path, "w") as dst,
            ):
                dst.create_dataset("features", data=src["features"][:])
            return output_h5_path, None

        out_h5 = os.path.join(tmpdir, "slide_out.h5")
        with (
            patch(
                "mussel.utils.feature_extract.extract_patch_features",
                side_effect=fake_extract_patch,
            ),
            patch(
                "mussel.utils.feature_extract.aggregate_slide_features",
                side_effect=fake_aggregate,
            ),
        ):
            save_features(
                patch_h5_path=patch_h5,
                slide_path="dummy.svs",
                output_h5_path=out_h5,
                aggregation_method="model",
                model_kwargs={"patch_arg": "value"},
                slide_model_kwargs={"patch_oom": False},
            )

        assert seen_model_kwargs == [{"patch_arg": "value"}]
        assert seen_slide_model_kwargs == [{"patch_oom": False}]


def test_aggregate_slide_features_forwards_slide_model_kwargs():
    """aggregate_slide_features forwards slide_model_kwargs to the model factory."""
    import h5py
    import tempfile
    import torch
    from mussel.utils.feature_extract import aggregate_slide_features

    features = np.random.rand(4, 8).astype(np.float32)
    coords = np.zeros((4, 2), dtype=np.int32)

    with tempfile.TemporaryDirectory() as tmpdir:
        patch_h5 = os.path.join(tmpdir, "patches.h5")
        out_h5 = os.path.join(tmpdir, "slide_out.h5")
        with h5py.File(patch_h5, "w") as f:
            f.create_dataset("features", data=features)
            f.create_dataset("coords", data=coords)
            f["features"].attrs["patch_size"] = 256

        fake_model = MagicMock()
        fake_model.get_model_fun.return_value = (
            lambda features, coords, patch_size: torch.zeros(1, 3)
        )
        fake_factory = MagicMock()
        fake_factory.get_model.return_value = fake_model

        with patch(
            "mussel.utils.feature_extract.get_model_factory", return_value=fake_factory
        ):
            aggregate_slide_features(
                patch_features_h5_path=patch_h5,
                output_h5_path=out_h5,
                aggregation_method="model",
                model_type=ModelType.TITAN_SLIDE,
                use_gpu=False,
                slide_model_kwargs={"patch_oom": False},
            )

        fake_factory.get_model.assert_called_once_with(
            None, False, None, patch_oom=False
        )


def test_aggregate_slide_features_batch_precision():
    """aggregate_slide_features_batch casts output to the requested precision."""
    import h5py
    import tempfile
    from mussel.utils.feature_extract import aggregate_slide_features_batch

    features = np.random.rand(6, 8).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        patch_h5 = os.path.join(tmpdir, "patches.h5")
        with h5py.File(patch_h5, "w") as f:
            f.create_dataset("features", data=features)

        out_h5 = os.path.join(tmpdir, "out.h5")

        aggregate_slide_features_batch(
            patch_features_h5_paths=[patch_h5],
            output_h5_paths=[out_h5],
            aggregation_method="mean",
            embedding_precision="float16",
        )

        with h5py.File(out_h5, "r") as f:
            assert f["features"].dtype.itemsize == 2
            assert f["features"].dtype.kind == "f"


def test_aggregate_slide_features_config_has_embedding_precision():
    """AggregateSlideFeaturesConfig must expose precision and slide model kwargs."""
    from mussel.cli.aggregate_slide_features import AggregateSlideFeaturesConfig
    from omegaconf import OmegaConf

    cfg = OmegaConf.structured(AggregateSlideFeaturesConfig)
    assert hasattr(cfg, "embedding_precision")
    assert cfg.embedding_precision == "float32"
    assert cfg.slide_model_kwargs == {}


def test_tessellate_extract_features_config_has_embedding_precision():
    """TessellateExtractFeaturesConfig must expose precision and model kwargs."""
    from mussel.cli.tessellate_extract_features import TessellateExtractFeaturesConfig
    from mussel.cli.tessellate import SegConfig

    cfg = TessellateExtractFeaturesConfig(seg_config=SegConfig())
    assert hasattr(cfg, "embedding_precision")
    assert cfg.embedding_precision == "float32"
    assert cfg.model_kwargs == {}
    assert cfg.slide_model_kwargs == {}
