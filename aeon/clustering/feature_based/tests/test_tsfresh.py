"""Tests for TSFresh Clusterer"""
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
        [3, 7, 5, 0, 0, 5, 7, 5, 0, 4, 0, 5, 6, 0, 0, 1, 6, 5, 3, 2],
    )
    assert np.array_equal(
        test_result,
        [3, 3, 0, 5, 0, 4, 6, 7, 0, 3, 5, 5, 0, 6, 0, 5, 0, 5, 0, 7],

    )
    assert train_score == 0.48947368421052634
    assert test_score == 0.5210526315789473
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
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 3, 4, 6, 7, 2, 5, 5, 3, 1],
    )
    assert np.array_equal(
        test_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 3, 3, 1, 6, 5, 1, 1, 3],
    )
    assert train_score == 0.7842105263157895
    assert test_score == 0.8263157894736842
    assert test_result.shape == (20,)
    assert train_result.shape == (20,)


