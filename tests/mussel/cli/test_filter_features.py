import os
from omegaconf import OmegaConf
import h5py
import torch

from mussel.cli.filter_features import FilterFeaturesConfig, main


def test_filter_features(tmp_path):
    features_h5_path = "tests/testdata/948176.features.h5"
    classifier_pkl = "tests/testdata/simple_classifier.pkl"
    output_h5_path = os.path.join(tmp_path, "filtered.features.h5")
    output_pt_path = os.path.join(tmp_path, "filtered.features.pt")

    cfg = FilterFeaturesConfig(
        features_h5_path=features_h5_path,
        features_pt_path=None,
        output_h5_path=output_h5_path,
        output_pt_path=output_pt_path,
        classifier_pkl=classifier_pkl,
        classifier_threshold=0.75,
        save_features_to_h5=True,
    )

    main(OmegaConf.create(cfg))

    assert os.path.exists(output_h5_path)
    assert os.path.exists(output_pt_path)

    # Check that the output files are not empty
    with h5py.File(output_h5_path, "r") as f:
        assert "features" in f
        assert f["features"].shape[0] > 0

    data = torch.load(output_pt_path, weights_only=True)
    assert data.shape[0] > 0
