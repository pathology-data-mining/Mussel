"""extract_features batch mode refuses duplicate slide IDs (outputs are named by slide ID)."""
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

import mussel.cli.extract_features
from mussel.cli.extract_features import ExtractFeaturesConfig
from mussel.models import ModelType


@pytest.mark.parametrize("aggregation_method", ["identity", "mean"])
def test_duplicate_slide_ids_are_refused(tmp_path, aggregation_method):
    paths = []
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "S1.svs").touch()
        (tmp_path / d / "S1.patch.h5").touch()
        paths.append(tmp_path / d)
    cfg = ExtractFeaturesConfig(
        patch_h5_paths=[str(p / "S1.patch.h5") for p in paths],
        slide_paths=[str(p / "S1.svs") for p in paths],
        output_dir=str(tmp_path / "out"), model_type=ModelType.CONCH1_5,
        aggregation_method=aggregation_method, use_gpu=False, num_workers=0,
    )
    with patch("mussel.cli.extract_features.extract_patch_features_batch") as extract, \
            patch("mussel.cli.extract_features.aggregate_slide_features_batch"):
        with pytest.raises(ValueError, match=r"slide IDs must be unique.*\['S1'\]"):
            mussel.cli.extract_features.main(OmegaConf.create(cfg))
    extract.assert_not_called()


def test_explicit_slide_ids_disambiguate(tmp_path):
    paths = []
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "S1.svs").touch()
        (tmp_path / d / "S1.patch.h5").touch()
        paths.append(tmp_path / d)
    cfg = ExtractFeaturesConfig(
        patch_h5_paths=[str(p / "S1.patch.h5") for p in paths],
        slide_paths=[str(p / "S1.svs") for p in paths], slide_ids=["a_S1", "b_S1"],
        output_dir=str(tmp_path / "out"), model_type=ModelType.CONCH1_5, use_gpu=False, num_workers=0,
    )
    with patch("mussel.cli.extract_features.extract_patch_features_batch") as extract, \
            patch("mussel.cli.extract_features._write_pts"):
        mussel.cli.extract_features.main(OmegaConf.create(cfg))
    outputs = extract.call_args.kwargs["output_h5_paths"]
    assert [p.split("/")[-1] for p in outputs] == ["a_S1.features.h5", "b_S1.features.h5"]
