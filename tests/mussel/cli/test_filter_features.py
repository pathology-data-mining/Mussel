import os
import pickle

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from mussel.cli.filter_features import FilterFeaturesConfig, main


class SimpleClassifier:
    """A classifier that scores tiles on a linear ramp so roughly half pass a 0.75 threshold."""

    def predict_proba(self, X):
        n = len(X)
        probs = np.column_stack([
            np.linspace(0.9, 0.1, n),  # class 0
            np.linspace(0.1, 0.9, n),  # class 1
        ])
        return probs


def _create_test_data(tmp_path):
    """Create test features H5 and a simple classifier pickle."""
    features_h5_path = os.path.join(tmp_path, "features.h5")
    num_tiles = 20
    feat_dim = 64
    features = np.random.randn(num_tiles, feat_dim).astype(np.float32)
    coords = np.array([[i * 256, 0] for i in range(num_tiles)])

    with h5py.File(features_h5_path, "w") as f:
        f.create_dataset("features", data=features)
        f.create_dataset("coords", data=coords)

    classifier_pkl = os.path.join(tmp_path, "classifier.pkl")
    with open(classifier_pkl, "wb") as f:
        pickle.dump(SimpleClassifier(), f)

    return features_h5_path, classifier_pkl


def test_filter_features(tmp_path):
    features_h5_path, classifier_pkl = _create_test_data(tmp_path)
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
