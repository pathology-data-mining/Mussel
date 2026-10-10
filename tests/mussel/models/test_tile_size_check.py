"""Recommended tile sizes: check_tile_size, TITAN's level-0 spacing, the JSON export."""
import json
import logging
import subprocess
import sys
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from mussel.models import (MODEL_PATCH_SIZES, ModelType, check_tile_size,
                           recommended_patch_sizes)
from mussel.utils.feature_extract import (_titan_patch_size_lv0,
                                          aggregate_slide_features_batch,
                                          extract_patch_features_batch)

# reef v2's TITAN input: CONCH v1.5 features of 224 px tiles at 0.5 µm/px.
REEF_224 = {"patch_size_to_resize_to_for_desired_mpp": 224, "mpp": 0.5,
            "patch_size": 425, "native_mpp": 0.2633}
TITAN_OK = {"patch_size_to_resize_to_for_desired_mpp": 512, "mpp": 0.5,
            "patch_size": 972, "native_mpp": 0.2633}


class TestCheckTileSize:
    def test_matching_tiles_pass(self):
        assert check_tile_size(ModelType.TITAN_SLIDE, TITAN_OK, "error") is None
        assert check_tile_size(ModelType.HOPTIMUS1, REEF_224, "error") is None

    def test_wrong_size_warns_by_default(self, caplog):
        with caplog.at_level(logging.WARNING):
            msg = check_tile_size(ModelType.TITAN_SLIDE, REEF_224, context="s1")
        assert "s1: TITAN_SLIDE tiles are 224 px, recommended 512 px" in msg
        assert msg in caplog.text

    def test_wrong_size_raises_in_error_mode(self):
        with pytest.raises(ValueError, match="224 px, recommended 512"):
            check_tile_size(ModelType.CONCH1_5, REEF_224, "error")

    def test_wrong_mpp_flagged_for_models_with_a_target(self):
        attrs = {**TITAN_OK, "mpp": 0.25}
        with pytest.raises(ValueError, match="0.25 µm/px, recommended 0.5"):
            check_tile_size(ModelType.TITAN_SLIDE, attrs, "error")
        # No target MPP recorded for H-optimus-1: only its size is checked.
        assert check_tile_size(ModelType.HOPTIMUS1, {**REEF_224, "mpp": 0.25}, "error") is None

    def test_off_and_missing_attrs_skip(self):
        assert check_tile_size(ModelType.TITAN_SLIDE, REEF_224, "off") is None
        assert check_tile_size(ModelType.TITAN_SLIDE, {}, "error") is None
        assert check_tile_size(ModelType.TITAN_SLIDE, None, "error") is None

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="tile_size_check"):
            check_tile_size(ModelType.TITAN_SLIDE, TITAN_OK, "strict")


class TestTitanPatchSizeLv0:
    def test_recorded_spacing_wins(self):
        assert _titan_patch_size_lv0(972, TITAN_OK) == 972

    def test_derived_from_mpp_on_40x_slide(self):
        attrs = {k: v for k, v in TITAN_OK.items() if k != "patch_size"}
        assert _titan_patch_size_lv0(None, attrs) == round(512 * 0.5 / 0.2633)

    def test_no_guess_without_metadata(self):
        with pytest.raises(ValueError, match="level-0 tile spacing"):
            _titan_patch_size_lv0(None, {})


def _patch_h5(path, attrs, features=False):
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("coords", data=np.zeros((3, 2), np.int64))
        for k, v in attrs.items():
            ds.attrs[k] = v
        if features:
            f.create_dataset("features", data=np.zeros((3, 8), np.float32))
    return str(path)


def test_patch_extraction_error_mode_fails_before_loading_the_model(tmp_path):
    h5 = _patch_h5(tmp_path / "s.patch.h5", REEF_224)
    with patch("mussel.utils.feature_extract.get_model_factory") as factory:
        with pytest.raises(ValueError, match="CONCH1_5 tiles are 224 px"):
            extract_patch_features_batch([h5], ["s.svs"], [str(tmp_path / "o.h5")],
                                         model_type=ModelType.CONCH1_5,
                                         tile_size_check="error")
    factory.assert_not_called()


def test_slide_aggregation_error_mode_fails_before_loading_the_model(tmp_path):
    h5 = _patch_h5(tmp_path / "s.patch_features.h5", REEF_224, features=True)
    with patch("mussel.utils.feature_extract.get_model_factory") as factory:
        with pytest.raises(ValueError, match="TITAN_SLIDE tiles are 224 px"):
            aggregate_slide_features_batch([h5], aggregation_method="model",
                                           model_type=ModelType.TITAN_SLIDE,
                                           use_gpu=False, tile_size_check="error")
    factory.assert_not_called()


def test_export_lists_every_model():
    out = subprocess.run([sys.executable, "-m", "mussel.cli.model_patch_sizes"],
                         capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    assert data["patch_sizes"] == recommended_patch_sizes()
    assert len(data["patch_sizes"]) == len(MODEL_PATCH_SIZES)
    assert data["patch_sizes"]["titan_slide"] == 512
    assert data["target_mpp"]["titan_slide"] == 0.5
