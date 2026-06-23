"""Tests for Catch22 Clusterer."""

import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator, ClusterMixin

from aeon.clustering.feature_based import Catch22Clusterer
from aeon.datasets import load_basic_motions, load_gunpoint
from aeon.testing.data_generation import make_example_3d_numpy


class _MockClusterer(ClusterMixin, BaseEstimator):
    """Minimal sklearn-style clusterer exposing n_jobs and predict_proba."""

    def __init__(self, n_clusters=2, n_jobs=1):
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        rng = np.random.RandomState(0)
        self.labels_ = rng.randint(0, self.n_clusters, size=X.shape[0])
        return self

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X):
        proba = np.zeros((X.shape[0], self.n_clusters))
        proba[:, 0] = 1.0
        return proba


def test_catch22_multivariate():
    """Test Catch22 Clusterer with univariate data."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 12

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    catach22 = Catch22Clusterer(
        catch24=False,
        random_state=1,
    )
    train_result = catach22.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_result)
    test_result = catach22.predict(X_test)
    test_score = metrics.rand_score(y_test, test_result)
    ari_test = metrics.adjusted_rand_score(y_test, test_result)
    ari_train = metrics.adjusted_rand_score(y_train, train_result)
    predict_proba = catach22.predict_proba(X_test)

    assert len(predict_proba) == 12
    assert ari_test == 0.1927353595255745
    assert ari_train == 0.09810791871058164
    assert len(predict_proba) == 12
    assert np.array_equal(
        test_result,
        [3, 4, 7, 7, 7, 7, 0, 7, 0, 4, 2, 2],
    )
    assert np.array_equal(
        train_result,
        [7, 3, 0, 5, 6, 4, 7, 7, 4, 7, 1, 2],
    )
    assert train_score == 0.4090909090909091
    assert test_score == 0.5
    assert test_result.shape == (12,)
    assert train_result.shape == (12,)


def test_catch22_univariate():
    """Test Catch22 Clusterer with multivariate data."""
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    num_points = 8

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    catach22 = Catch22Clusterer(
        catch24=False,
        random_state=1,
    )
    train_result = catach22.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_result)
    test_result = catach22.predict(X_test)
    test_score = metrics.rand_score(y_test, test_result)
    ari_test = metrics.adjusted_rand_score(y_test, test_result)
    ari_train = metrics.adjusted_rand_score(y_train, train_result)
    predict_proba = catach22.predict_proba(X_test)

    assert len(predict_proba) == 8
    assert ari_test == 0.023255813953488372
    assert ari_train == 0.0
    assert np.array_equal(
        test_result,
        [3, 0, 1, 3, 7, 5, 2, 2],
    )
    assert np.array_equal(
        train_result,
        [5, 0, 3, 7, 4, 6, 2, 1],
    )
    assert train_score == 0.42857142857142855
    assert test_score == 0.5714285714285714
    assert test_result.shape == (8,)
    assert train_result.shape == (8,)


def test_catch22_custom_estimator():
    """Test Catch22Clusterer with an estimator exposing n_jobs and predict_proba."""
    X = make_example_3d_numpy(8, 1, 12, return_y=False, random_state=1)
    clst = Catch22Clusterer(
        estimator=_MockClusterer(n_clusters=2), catch24=False, random_state=1
    )
    clst.fit(X)
    # the wrapped estimator's n_jobs should be set from the clusterer
    assert clst._estimator.n_jobs == clst._n_jobs
    preds = clst.predict(X)
    assert preds.shape == (8,)
    proba = clst.predict_proba(X)
    assert proba.shape == (8, 2)


def test_catch22_get_test_params():
    """Test the testing parameter set produces a valid estimator."""
    params = Catch22Clusterer._get_test_params()
    assert "features" in params
    X = make_example_3d_numpy(8, 1, 12, return_y=False, random_state=1)
    assert Catch22Clusterer(**params).fit(X) is not None
