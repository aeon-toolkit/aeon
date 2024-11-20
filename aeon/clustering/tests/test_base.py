"""Unit tests for clustering base class functionality."""

import numpy as np
import numpy.random
import pytest

from aeon.clustering.base import BaseClusterer
from aeon.testing.mock_estimators import MockCluster


def test_correct_input():
    """Tests errors raised with wrong inputs: X and/or y."""
    dummy = MockCluster()

    X = ["list", "of", "invalid", "test", "strings"]
    msg1 = r"ERROR passed a list containing <class 'str'>"
    with pytest.raises(TypeError, match=msg1):
        dummy.fit(X)

    # dict X
    X = {
        0: "invalid",
        1: "input",
        2: "dict",
    }
    msg2 = r"ERROR passed input of type <class 'dict'>"
    with pytest.raises(TypeError, match=msg2):
        dummy.fit(X)

    # 2d list of int X
    X = [[1, 1, 1], [1, 3, 4]]
    msg3 = r"lists should either 2D numpy arrays or pd.DataFrames"
    with pytest.raises(TypeError, match=msg3):
        dummy.fit(X)

    # correct X
    X = np.random.randn(5, 5)
    dummy.fit(X)
    assert (dummy.predict(X)).shape == (5,)
    assert (dummy.predict_proba(X)).shape == (5,)


class _TestClusterer(BaseClusterer):
    """Clusterer for testing base class fit/predict/predict_proba."""

    def __init__(self):
        super().__init__()

    def _fit(self, X, y=None):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return np.zeros(shape=(len(X),), dtype=int)


def test_base_clusterer():
    """Test with no clusters."""
    clst = _TestClusterer()
    X = np.random.random(size=(10, 1, 20))
    clst.fit(X)
    assert clst.is_fitted
    preds = clst._predict_proba(X)
    assert preds.shape == (10, 1)
