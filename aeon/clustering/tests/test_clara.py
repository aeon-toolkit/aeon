"""Tests for time series k-medoids."""

import numpy as np
from sklearn import metrics

from aeon.clustering._clara import TimeSeriesCLARA
from aeon.datasets import load_basic_motions, load_gunpoint


def test_clara_uni():
    """Test implementation of CLARA."""
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    num_points = 20

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    clara = TimeSeriesCLARA(
        random_state=1,
        n_samples=10,
        n_init=2,
        max_iter=5,
        init="first",
        distance="euclidean",
        n_clusters=2,
    )
    train_medoids_result = clara.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = clara.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = clara.predict_proba(X_test)
    assert np.array_equal(
        test_medoids_result,
        [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    )
    assert np.array_equal(
        train_medoids_result,
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1],
    )
    assert test_score == 0.5210526315789473
    assert train_score == 0.5578947368421052
    assert np.isclose(clara.inertia_, 78.79839208236065)
    assert clara.n_iter_ == 2
    assert np.array_equal(
        clara.labels_, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]
    )
    assert isinstance(clara.cluster_centers_, np.ndarray)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


def test_clara_multi():
    """Test implementation of CLARA."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 20

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    clara = TimeSeriesCLARA(
        random_state=1,
        n_samples=10,
        n_init=2,
        max_iter=5,
        init="first",
        distance="euclidean",
        n_clusters=2,
    )
    train_medoids_result = clara.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = clara.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = clara.predict_proba(X_test)
    assert np.array_equal(
        test_medoids_result,
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    assert np.array_equal(
        train_medoids_result,
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
    )
    assert test_score == 0.4789473684210526
    assert train_score == 0.5578947368421052
    assert np.isclose(clara.inertia_, 1900.6752544011563)
    assert clara.n_iter_ == 3
    assert np.array_equal(
        clara.labels_, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1]
    )
    assert isinstance(clara.cluster_centers_, np.ndarray)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1
