"""Unit tests for clustering base class functionality."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from aeon.clustering.base import BaseClusterer
from aeon.testing.mock_estimators import (
    MockCluster,
    MockLabelCluster,
    MockTransductiveCluster,
)


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


class _NClustersClusterer(BaseClusterer):
    """Clusterer exposing an ``n_clusters`` attribute, assigning cases round-robin.

    The base ``_predict_proba`` sizes its probability matrix from ``n_clusters``
    when that attribute is present, and falls back to ``max(labels_) + 1`` when
    it is ``None``. This mock lets a test drive both branches;
    ``_predict`` returns three distinct clusters (labels 0, 1, 2).
    """

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        super().__init__()

    def _fit(self, X, y=None):
        self.labels_ = np.arange(len(X)) % 3
        return self

    def _predict(self, X):
        return np.arange(len(X)) % 3


def test_base_predict_proba_width_from_n_clusters_attribute():
    """Test base ``_predict_proba`` matrix width follows the ``n_clusters`` attribute.

    When a clusterer defines ``n_clusters``, the probability matrix has exactly
    that many columns (even if wider than the number of clusters actually
    predicted). When ``n_clusters`` is ``None`` the base falls back to
    ``max(labels_) + 1`` (here 3, for labels 0, 1, 2). Either way each row is a
    one-hot of the assigned cluster.
    """
    X = np.random.random(size=(9, 1, 20))
    expected = np.arange(9) % 3

    # explicit n_clusters -> that many columns (line: n_clusters = self.n_clusters)
    proba = _NClustersClusterer(n_clusters=4).fit(X)._predict_proba(X)
    assert proba.shape == (9, 4)
    assert np.all(proba.sum(axis=1) == 1)
    assert np.array_equal(np.argmax(proba, axis=1), expected)

    # n_clusters is None -> fall back to max(labels_) + 1 == 3
    proba = _NClustersClusterer(n_clusters=None).fit(X)._predict_proba(X)
    assert proba.shape == (9, 3)
    assert np.array_equal(np.argmax(proba, axis=1), expected)


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


def test_base_predict_proba_columns_fixed_when_cluster_missing():
    """Test ``predict_proba`` columns match ``predict`` when ``X`` misses a cluster.

    Columns follow the fitted clusters, so the argmax of ``predict_proba`` equals
    ``predict`` and the width stays the same on any subset of the data.
    """
    X = np.random.random(size=(9, 1, 20))
    for n_clusters in [3, None]:
        clst = MockLabelCluster(n_clusters=n_clusters).fit(X)
        sub = X[clst.predict(X) != 0]
        preds = clst.predict(sub)
        proba = clst.predict_proba(sub)
        assert proba.shape == (len(sub), 3)
        assert np.array_equal(proba.argmax(axis=1), preds)
        assert clst.predict_proba(sub[:1]).shape == (1, 3)


def test_base_predict_proba_negative_prediction_raises():
    """Test a negative predicted label raises a ``ValueError``.

    Negative labels such as ``-1`` noise are not clusters, so they have no
    ``predict_proba`` column.
    """
    X = np.random.random(size=(6, 1, 20))
    clst = MockLabelCluster(labels=(-1, 0, 1)).fit(X)
    assert np.array_equal(clst.predict(X), [-1, 0, 1, -1, 0, 1])
    with pytest.raises(ValueError, match="negative labels"):
        clst.predict_proba(X)


def test_base_predict_proba_ignores_noise_in_labels():
    """Test negative labels in ``labels_`` get no ``predict_proba`` column.

    Like ``MeanShift(cluster_all=False)``, ``labels_`` contains ``-1`` for noise
    but ``predict`` assigns every case to a cluster, so the columns are the
    clusters and the argmax matches ``predict``.
    """
    X = np.random.random(size=(8, 1, 20))
    for n_clusters in [3, None]:
        clst = MockLabelCluster(
            labels=(0, 1, 2), fit_labels=(-1, 0, 1, 2), n_clusters=n_clusters
        ).fit(X)
        proba = clst.predict_proba(X)
        assert proba.shape == (8, 3)
        assert np.array_equal(proba.argmax(axis=1), clst.predict(X))

    # an empty cluster alongside noise keeps cluster indices as columns
    clst = MockLabelCluster(labels=(0, 2), fit_labels=(-1, 0, 2)).fit(X)
    proba = clst.predict_proba(X)
    assert proba.shape == (8, 3)
    assert np.array_equal(proba.argmax(axis=1), clst.predict(X))


def test_meanshift_noise_predict_proba_matches_predict():
    """Test a wrapped ``MeanShift`` with noise in ``labels_`` agrees with predict."""
    from sklearn.cluster import MeanShift

    from aeon.clustering.feature_based import SummaryClusterer
    from aeon.testing.data_generation import make_example_3d_numpy

    X, _ = make_example_3d_numpy(
        n_cases=40, n_channels=1, n_timepoints=20, random_state=0
    )
    clst = SummaryClusterer(estimator=MeanShift(cluster_all=False, bandwidth=0.5)).fit(
        X
    )
    assert -1 in clst.labels_
    preds = clst.predict(X)
    proba = clst.predict_proba(X)
    assert proba.shape == (40, clst.labels_.max() + 1)
    assert np.array_equal(proba.argmax(axis=1), preds)


def test_base_predict_proba_labels_above_n_clusters():
    """Test labels at or above ``n_clusters`` widen the matrix instead of failing.

    Labels are column indices, so the width is ``max(labels_) + 1`` when that is
    larger than ``n_clusters``.
    """
    X = np.random.random(size=(8, 1, 20))
    clst = MockLabelCluster(labels=(22, 3, 32, 22), n_clusters=3).fit(X)
    proba = clst.predict_proba(X)
    assert proba.shape == (8, 33)
    assert np.array_equal(proba.argmax(axis=1), clst.predict(X))


def test_kmeans_predict_proba_matches_predict_on_subset():
    """Test a clusterer using the base default agrees with ``predict`` on a subset."""
    from aeon.clustering import TimeSeriesKMeans
    from aeon.testing.data_generation import make_example_3d_numpy

    X, _ = make_example_3d_numpy(
        n_cases=20, n_channels=1, n_timepoints=10, random_state=0
    )
    km = TimeSeriesKMeans(n_clusters=3, random_state=0).fit(X)
    sub = X[km.predict(X) != 0]
    proba = km.predict_proba(sub)
    assert proba.shape == (len(sub), 3)
    assert np.array_equal(proba.argmax(axis=1), km.predict(sub))
