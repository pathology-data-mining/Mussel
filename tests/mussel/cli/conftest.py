"""
Shared pytest configuration and fixtures for CLI tests.

Fixtures from tests/conftest.py are automatically available here via pytest's
conftest discovery — no explicit imports needed.
"""

import pickle

import numpy as np
import pytest


@pytest.fixture
def test_data_path_cli(test_data_path: str) -> str:
    """Return path to test data directory (alias for CLI tests)."""
    return test_data_path


@pytest.fixture
def cut_classifier(tmp_path):
    """Build a tile classifier whose kept tiles are known, for testing filtering.

    ``make(features)`` takes an unfiltered run's pre-filter features (tiles x
    dim) and saves a logistic regression that, at ``classifier_threshold=0.5``,
    keeps the tiles whose score along a fixed direction is above a cut. It
    returns ``(classifier_pkl_path, keep)``, where ``keep(features)`` is the mask
    of tiles the filter must keep. The cut sits in the widest gap between scores
    in the middle half, far from any tile, so GPU noise in the features can't
    move a tile across it.

    It is pickled because ``classifier_pkl`` is read with ``pickle.load``
    (``mussel.utils.file.load_classifier``); the test pickles only this object,
    which it built itself, into its own temp directory.

    The test data's ``simple_classifier.pkl`` can't be used for this: it gives
    every tile P ~ 0.945, so it filters nothing at any usual threshold.
    """
    from sklearn.linear_model import LogisticRegression

    def make(features):
        features = np.asarray(features, dtype=np.float64)
        direction = features[0] - features.mean(0)
        scores = np.sort(features @ direction)
        mid = np.arange(len(scores) // 4, 3 * len(scores) // 4)
        k = mid[np.argmax(scores[mid + 1] - scores[mid])]
        cut = (scores[k] + scores[k + 1]) / 2
        classifier = LogisticRegression()
        classifier.classes_ = np.array([0, 1])
        classifier.coef_ = direction[None]
        classifier.intercept_ = np.array([-cut])
        classifier.n_features_in_ = features.shape[1]
        path = tmp_path / "cut_classifier.pkl"
        path.write_bytes(pickle.dumps(classifier))

        def keep(f):
            return np.asarray(f, dtype=np.float64) @ direction > cut

        return str(path), keep

    return make
