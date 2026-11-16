"""Tests for Catch22 Clusterer."""

import numpy as np
from sklearn import metrics

from aeon.clustering.feature_based import Catch22Clusterer
from aeon.datasets import load_basic_motions, load_gunpoint


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
