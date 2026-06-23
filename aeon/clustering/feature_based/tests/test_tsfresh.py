"""Tests for TSFresh Clusterer."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClusterMixin

from aeon.clustering.feature_based import TSFreshClusterer
from aeon.datasets import load_basic_motions, load_gunpoint
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


class _MockClusterer(ClusterMixin, BaseEstimator):
    """Minimal sklearn-style clusterer exposing n_jobs and predict_proba."""

    def __init__(self, n_clusters=None, n_jobs=1):
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


@pytest.mark.skipif(
    not _check_soft_dependencies(["tsfresh"], severity="none"),
    reason="TSFresh soft dependency unavailable.",
)
def test_all_fc_parameters_uni():
    """Test TSFresh Clusterer with all FC parameters."""
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    num_points = 5

    X_train = X_train[:num_points]
    X_test = X_test[:num_points]
    fc_parameters = ["minimal", "efficient", "comprehensive"]
    for fc in fc_parameters:
        tsfresh = TSFreshClusterer(
            n_clusters=2, random_state=1, default_fc_parameters=fc
        )

        train_result = tsfresh.fit_predict(X_train)
        test_result = tsfresh.predict(X_test)
        predict_proba = tsfresh.predict_proba(X_test)
        assert len(predict_proba) == 5
        assert not np.isnan(train_result).any()
        assert not np.isnan(test_result).any()
        assert test_result.shape == (5,)
        assert train_result.shape == (5,)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tsfresh"], severity="none"),
    reason="TSFresh soft dependency unavailable.",
)
def test_all_fc_parameters_multi():
    """Test TSFresh Clusterer with all FC parameters."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 5

    X_train = X_train[:num_points]
    X_test = X_test[:num_points]
    fc_parameters = ["minimal", "efficient", "comprehensive"]
    for fc in fc_parameters:
        tsfresh = TSFreshClusterer(
            n_clusters=2, random_state=1, default_fc_parameters=fc
        )

        train_result = tsfresh.fit_predict(X_train)
        test_result = tsfresh.predict(X_test)
        predict_proba = tsfresh.predict_proba(X_test)
        assert len(predict_proba) == 5
        assert not np.isnan(train_result).any()
        assert not np.isnan(test_result).any()
        assert test_result.shape == (5,)
        assert train_result.shape == (5,)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tsfresh"], severity="none"),
    reason="TSFresh soft dependency unavailable.",
)
def test_tsfresh_custom_estimator():
    """Test TSFreshClusterer with an estimator exposing n_jobs and predict_proba.

    Passing an estimator with ``n_clusters=None`` also exercises the branch that
    fills in ``n_clusters`` from the clusterer.
    """
    X = make_example_3d_numpy(6, 1, 12, return_y=False, random_state=1)
    clst = TSFreshClusterer(
        n_clusters=2,
        estimator=_MockClusterer(n_clusters=None),
        default_fc_parameters="minimal",
        random_state=1,
    )
    clst.fit(X)
    assert clst._estimator.n_jobs == clst._n_jobs
    assert clst._estimator.n_clusters == 2
    preds = clst.predict(X)
    assert preds.shape == (6,)
    proba = clst.predict_proba(X)
    assert proba.shape == (6, 2)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tsfresh"], severity="none"),
    reason="TSFresh soft dependency unavailable.",
)
def test_tsfresh_get_test_params():
    """Test the testing parameter set produces a valid estimator."""
    params = TSFreshClusterer._get_test_params()
    assert params["n_clusters"] == 3
    X = make_example_3d_numpy(6, 1, 12, return_y=False, random_state=1)
    assert TSFreshClusterer(**params).fit(X) is not None
