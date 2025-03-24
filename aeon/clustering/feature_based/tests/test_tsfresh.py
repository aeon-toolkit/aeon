"""Tests for TSFresh Clusterer."""

import numpy as np
from sklearn import metrics

from aeon.clustering.feature_based import TSFreshClusterer
from aeon.datasets import load_basic_motions, load_gunpoint

def test_tsfresh_univariate():
    """Test TSFresh Clusterer with univariate data."""
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    num_points = 20

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    tsfresh = TSFreshClusterer(
        random_state=1,
        n_clusters=2,
    )
    train_result = tsfresh.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_result)
    test_result = tsfresh.predict(X_test)
    test_score = metrics.rand_score(y_test, test_result)

    assert np.array_equal(
        train_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    )
    assert np.array_equal(
        test_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    assert train_score == 0.49473684210526314
    assert test_score == 0.4789473684210526
    assert test_result.shape == (20,)
    assert train_result.shape == (20,)


def test_tsfresh_multivariate():
    """Test TSFresh Clusterer with multivariate data."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 20

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    tsfresh = TSFreshClusterer(
        random_state=1,
        n_clusters=2,
    )
    train_result = tsfresh.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_result)
    test_result = tsfresh.predict(X_test)
    test_score = metrics.rand_score(y_test, test_result)

    assert np.array_equal(
        train_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    )
    assert np.array_equal(
        test_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    )
    assert train_score == 1.0
    assert test_score == 1.0
    assert test_result.shape == (20,)
    assert train_result.shape == (20,)
