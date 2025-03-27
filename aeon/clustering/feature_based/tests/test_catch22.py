"""Tests for Catch22 Clusterer."""

import numpy as np
from sklearn import metrics

from aeon.clustering.feature_based import Catch22Clusterer
from aeon.datasets import load_basic_motions, load_gunpoint


def test_catch24_multivariate():
    """Test Catch24 Clusterer with univariate data."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 20

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    catach24 = Catch22Clusterer(
        random_state=1,
    )
    catach24.fit(X_train)
    train_result = catach24.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_result)
    test_result = catach24.predict(X_test)
    test_score = metrics.rand_score(y_test, test_result)
    predict_proba = catach24.predict_proba(X_test)

    assert len(predict_proba) == 20
    assert np.array_equal(
        test_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 3, 3, 6, 5, 3, 1, 1, 3],
    )
    assert np.array_equal(
        train_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 7, 2, 4, 3, 3, 6, 1],
    )
    assert train_score == 0.6684210526315789
    assert test_score == 0.8263157894736842
    assert test_result.shape == (20,)
    assert train_result.shape == (20,)


def test_catch24_univariate():
    """Test Catch24 Clusterer with multivariate data."""
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    num_points = 20

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    catach24 = Catch22Clusterer(
        random_state=1,
    )
    train_result = catach24.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_result)
    test_result = catach24.predict(X_test)
    test_score = metrics.rand_score(y_test, test_result)
    ari_test = metrics.adjusted_rand_score(y_test, test_result)
    ari_train = metrics.adjusted_rand_score(y_train, train_result)
    predict_proba = catach24.predict_proba(X_test)

    assert len(predict_proba) == 20
    assert ari_test == 0.036247577795508946
    assert ari_train == 0.16466826538768986
    assert np.array_equal(
        test_result,
        [1, 3, 4, 6, 7, 3, 5, 5, 6, 3, 3, 1, 3, 1, 1, 7, 3, 0, 6, 3],
    )
    assert np.array_equal(
        train_result,
        [3, 3, 7, 7, 0, 3, 2, 4, 1, 1, 6, 1, 5, 1, 3, 1, 6, 3, 1, 5],
    )
    assert train_score == 0.5947368421052631
    assert test_score == 0.531578947368421
    assert test_result.shape == (20,)
    assert train_result.shape == (20,)


def test_catch22_multivariate():
    """Test Catch22 Clusterer with univariate data."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 20

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

    assert len(predict_proba) == 20
    assert ari_test == 0.6451612903225806
    assert ari_train == 0.32639279684862127
    assert len(predict_proba) == 20
    assert np.array_equal(
        test_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 3, 3, 6, 5, 3, 1, 1, 3],
    )
    assert np.array_equal(
        train_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 7, 2, 4, 3, 3, 6, 1],
    )
    assert train_score == 0.6684210526315789
    assert test_score == 0.8263157894736842
    assert test_result.shape == (20,)
    assert train_result.shape == (20,)


def test_catch22_univariate():
    """Test Catch22 Clusterer with multivariate data."""
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    num_points = 20

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

    assert len(predict_proba) == 20
    assert ari_test == 0.036247577795508946
    assert ari_train == 0.16466826538768986
    assert np.array_equal(
        test_result,
        [1, 3, 4, 6, 7, 3, 5, 5, 6, 3, 3, 1, 3, 1, 1, 7, 3, 0, 6, 3],
    )
    assert np.array_equal(
        train_result,
        [3, 3, 7, 7, 0, 3, 2, 4, 1, 1, 6, 1, 5, 1, 3, 1, 6, 3, 1, 5],
    )
    assert train_score == 0.5947368421052631
    assert test_score == 0.531578947368421
    assert test_result.shape == (20,)
    assert train_result.shape == (20,)
