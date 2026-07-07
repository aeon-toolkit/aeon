"""Unit tests for clustering base class functionality."""

import numpy as np
import numpy.random
import pytest
from sklearn.exceptions import NotFittedError

from aeon.clustering.base import BaseClusterer
from aeon.testing.mock_estimators import MockCluster, MockTransductiveCluster


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


class _CountingClusterer(BaseClusterer):
    """Clusterer counting _predict calls, for testing fit_predict independence."""

    def __init__(self):
        self.predict_calls = 0
        super().__init__()

    def _fit(self, X, y=None):
        self.labels_ = np.zeros(shape=(len(X),), dtype=int)
        return self

    def _predict(self, X):
        self.predict_calls += 1
        return np.zeros(shape=(len(X),), dtype=int)


def test_fit_predict_returns_labels():
    """Test fit_predict returns the same labels as fit(X).labels_."""
    X = np.random.random(size=(10, 1, 20))

    fitted = MockTransductiveCluster().fit(X)
    labels = MockTransductiveCluster().fit_predict(X)
    assert np.array_equal(labels, fitted.labels_)

    fitted = MockCluster().fit(X)
    labels = MockCluster().fit_predict(X)
    assert np.array_equal(labels, fitted.labels_)


def test_fit_predict_does_not_call_predict():
    """Test fit_predict does not route through predict/_predict."""
    X = np.random.random(size=(10, 1, 20))

    clst = _CountingClusterer()
    clst.fit_predict(X)
    assert clst.predict_calls == 0

    # the transductive mock raises if _predict is ever reached
    labels = MockTransductiveCluster().fit_predict(X)
    assert labels.shape == (10,)


def test_predict_without_capability_raises():
    """Test predict raises a clear error when capability:predict is False."""
    X = np.random.random(size=(10, 1, 20))
    clst = MockTransductiveCluster()

    assert not clst.get_tag("capability:predict")

    # unfitted estimators still raise the standard fitted-state error
    with pytest.raises(NotFittedError, match="has not been fitted"):
        clst.predict(X)

    clst.fit(X)
    msg = (
        "MockTransductiveCluster does not support out-of-sample prediction. "
        r"Use fit_predict\(X\) to cluster a collection, or inspect labels_ "
        r"after fit\(X\)."
    )
    with pytest.raises(NotImplementedError, match=msg):
        clst.predict(X)
    with pytest.raises(NotImplementedError, match=msg):
        clst.predict_proba(X)


def test_predict_with_capability():
    """Test clusterers with out-of-sample prediction support are unaffected."""
    X = np.random.random(size=(10, 1, 20))
    clst = MockCluster()

    assert clst.get_tag("capability:predict")

    clst.fit(X)
    assert clst.predict(X).shape == (10,)
    assert clst.predict_proba(X).shape == (10,)
