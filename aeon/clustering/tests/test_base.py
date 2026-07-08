"""Unit tests for clustering base class functionality."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from aeon.clustering.base import BaseClusterer
from aeon.testing.mock_estimators import MockCluster, MockTransductiveCluster


def test_correct_input():
    """Test fit rejects invalid X and accepts a valid 2D array.

    Exercises the input validation in ``BaseClusterer.fit``: a list of strings,
    a dict, and a 2D list of ints each raise a ``TypeError`` with a specific
    message, while a valid 2D numpy array fits and can then predict.
    """
    dummy = MockCluster()

    X = ["list", "of", "invalid", "test", "strings"]
    msg = r"ERROR passed a list containing <class 'str'>"
    with pytest.raises(TypeError, match=msg):
        dummy.fit(X)

    X = {0: "invalid", 1: "input", 2: "dict"}
    with pytest.raises(TypeError, match=r"ERROR passed input of type <class 'dict'>"):
        dummy.fit(X)

    X = [[1, 1, 1], [1, 3, 4]]
    with pytest.raises(TypeError, match=r"lists should either 2D numpy arrays"):
        dummy.fit(X)

    X = np.random.randn(5, 5)
    dummy.fit(X)
    assert dummy.predict(X).shape == (5,)
    assert dummy.predict_proba(X).shape == (5,)


class _DefaultProbaClusterer(BaseClusterer):
    """Clusterer that assigns every case to cluster 0.

    It overrides only ``_fit`` and ``_predict`` and deliberately does not
    override ``_predict_proba``, so it exercises the ``BaseClusterer`` default
    ``_predict_proba`` (which one-hot encodes the ``_predict`` output).
    """

    def __init__(self):
        super().__init__()

    def _fit(self, X, y=None):
        self.labels_ = np.zeros(len(X), dtype=int)
        return self

    def _predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_base_predict_proba_is_one_hot_of_predict():
    """Test the base default ``_predict_proba`` one-hot encodes ``_predict``.

    A clusterer that does not override ``_predict_proba`` falls back to the
    base implementation: run ``_predict`` and put probability 1 on the assigned
    cluster. With a single cluster this is an ``(n_cases, 1)`` column of ones.
    """
    X = np.random.random(size=(10, 1, 20))
    clst = _DefaultProbaClusterer().fit(X)
    proba = clst._predict_proba(X)
    assert proba.shape == (10, 1)
    assert np.all(proba == 1.0)


def test_fit_predict_returns_labels_from_its_own_fit():
    """Test ``fit_predict(X)`` returns ``labels_`` from the fit it performs.

    ``fit_predict`` is defined as ``fit(X)`` followed by returning ``labels_``
    (the assignment of the fitted collection); it must not return ``predict(X)``.
    ``MockTransductiveCluster`` has a known ``labels_`` pattern (alternating 0/1)
    and a ``_predict`` that raises, so matching that pattern shows ``fit_predict``
    returned ``labels_`` rather than routing through ``predict``. The check is
    repeated on ``MockCluster`` to confirm the same for a predict-capable
    clusterer.
    """
    X = np.random.random(size=(10, 1, 20))

    labels = MockTransductiveCluster().fit_predict(X)
    assert np.array_equal(labels, np.arange(10) % 2)

    clst = MockCluster()
    assert np.array_equal(clst.fit_predict(X), clst.labels_)


def test_fit_predict_valid_without_predict_capability():
    """Test ``fit_predict`` works for a clusterer without out-of-sample predict.

    Because ``fit_predict`` never routes through ``_predict``, it is valid even
    when ``capability:predict`` is False. ``MockTransductiveCluster._predict``
    raises, so ``fit_predict`` completing normally (rather than erroring) proves
    ``_predict`` was never called.
    """
    X = np.random.random(size=(10, 1, 20))
    clst = MockTransductiveCluster()
    assert not clst.get_tag("capability:predict")

    labels = clst.fit_predict(X)
    assert labels.shape == (10,)


def test_predict_without_capability_raises():
    """Test ``predict``/``predict_proba`` raise for a transductive clusterer.

    When ``capability:predict`` is False, ``predict`` and ``predict_proba`` must
    raise ``NotFittedError`` before fit and ``NotImplementedError`` after fit
    (out-of-sample prediction is unsupported). Only the distinctive part of the
    message is matched, so the test is robust to wording of the accompanying
    guidance text.
    """
    X = np.random.random(size=(10, 1, 20))
    clst = MockTransductiveCluster()
    assert not clst.get_tag("capability:predict")

    with pytest.raises(NotFittedError, match="has not been fitted"):
        clst.predict(X)

    clst.fit(X)
    msg = "does not support out-of-sample prediction"
    with pytest.raises(NotImplementedError, match=msg):
        clst.predict(X)
    with pytest.raises(NotImplementedError, match=msg):
        clst.predict_proba(X)


def test_predict_with_capability():
    """Test ``predict``/``predict_proba`` work when ``capability:predict`` is True.

    The default, non-transductive case: after ``fit`` both ``predict`` and
    ``predict_proba`` return one row per case without raising.
    """
    X = np.random.random(size=(10, 1, 20))
    clst = MockCluster()
    assert clst.get_tag("capability:predict")

    clst.fit(X)
    assert clst.predict(X).shape == (10,)
    assert clst.predict_proba(X).shape == (10,)
