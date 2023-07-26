# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics

from aeon.clustering.clarans import TimeSeriesCLARANS
from aeon.datasets import load_basic_motions, load_gunpoint


def test_clarans_uni():
    """Test implementation of CLARANS."""
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    num_points = 20

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    clarans = TimeSeriesCLARANS(
        random_state=1,
        n_init=2,
        init_algorithm="first",
        distance="euclidean",
        n_clusters=2,
    )
    train_medoids_result = clarans.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = clarans.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = clarans.predict_proba(X_test)
    assert np.array_equal(
        test_medoids_result,
        [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    )
    assert np.array_equal(
        train_medoids_result,
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1],
    )
    assert test_score == 0.5578947368421052
    assert train_score == 0.49473684210526314
    assert np.isclose(clarans.inertia_, 94.22886366617668)
    assert clarans.n_iter_ == 0
    assert np.array_equal(
        clarans.labels_, [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]
    )
    assert isinstance(clarans.cluster_centers_, np.ndarray)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


def test_clara_multi():
    """Test implementation of CLARANS."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 20

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    clarans = TimeSeriesCLARANS(
        random_state=1,
        n_init=2,
        init_algorithm="first",
        distance="euclidean",
        n_clusters=2,
    )

    train_medoids_result = clarans.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = clarans.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = clarans.predict_proba(X_test)
    assert np.array_equal(
        test_medoids_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    )
    assert np.array_equal(
        train_medoids_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    )
    assert test_score == 0.4789473684210526
    assert train_score == 0.4789473684210526
    assert np.isclose(clarans.inertia_, 1762.5323632597904)
    assert clarans.n_iter_ == 0
    assert np.array_equal(
        clarans.labels_, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    )
    assert isinstance(clarans.cluster_centers_, np.ndarray)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1
