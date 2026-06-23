"""Tests for Summary Clusterer."""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from aeon.clustering.feature_based import SummaryClusterer
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


def test_all_summary_stat_uni():
    """Test Summary Clusterer with all summary stat."""
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    num_points = 8

    X_train = X_train[:num_points]
    X_test = X_test[:num_points]
    summary_stats_options = ["default", "quantiles", "bowley", "tukey"]
    for summary_stat in summary_stats_options:
        summary = SummaryClusterer(random_state=1, summary_stats=summary_stat)
        train_result = summary.fit_predict(X_train)
        test_result = summary.predict(X_test)
        predict_proba = summary.predict_proba(X_test)
        assert len(predict_proba) == 8
        assert not np.isnan(train_result).any()
        assert not np.isnan(test_result).any()
        assert test_result.shape == (8,)
        assert train_result.shape == (8,)


def test_summary_custom_estimator():
    """Test SummaryClusterer with an estimator exposing n_jobs and predict_proba."""
    X = make_example_3d_numpy(8, 1, 12, return_y=False, random_state=1)
    clst = SummaryClusterer(estimator=_MockClusterer(n_clusters=2), random_state=1)
    clst.fit(X)
    assert clst._estimator.n_jobs == clst._n_jobs
    preds = clst.predict(X)
    assert preds.shape == (8,)
    proba = clst.predict_proba(X)
    assert proba.shape == (8, 2)


def test_all_summary_stat_multi():
    """Test Summary Clusterer with all summary stat."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 8

    X_train = X_train[:num_points]
    X_test = X_test[:num_points]
    summary_stats_options = ["default", "quantiles", "bowley", "tukey"]
    for summary_stat in summary_stats_options:
        summary = SummaryClusterer(random_state=1, summary_stats=summary_stat)
        train_result = summary.fit_predict(X_train)
        test_result = summary.predict(X_test)
        predict_proba = summary.predict_proba(X_test)
        assert len(predict_proba) == 8
        assert not np.isnan(train_result).any()
        assert not np.isnan(test_result).any()
        assert test_result.shape == (8,)
        assert train_result.shape == (8,)
