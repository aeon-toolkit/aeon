"""Unit tests for clustering base class functionality."""

import numpy as np
import numpy.random

from aeon.clustering.base import BaseClusterer


class _TestClusterer(BaseClusterer):
    """Clusterer for testing base class fit/predict/predict_proba."""

    def _fit(self, X, y=None):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return np.zeros(shape=(len(X),), dtype=int)

    def _score(self, X, y=None):
        return 1.0


def test_base_clusterer():
    """Test with no clusters."""
    clst = _TestClusterer()
    X = np.random.random(size=(10, 1, 20))
    clst.fit(X)
    assert clst.is_fitted
    preds = clst._predict_proba(X)
    assert preds.shape == (10, 1)
