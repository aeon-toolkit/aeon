"""Tests for CLARANS."""

import numpy as np
from sklearn import metrics
from sklearn.utils import check_random_state

from aeon.clustering._clarans import TimeSeriesCLARANS
from aeon.clustering.tests.test_k_medoids import check_value_in_every_cluster
from aeon.datasets import load_basic_motions, load_gunpoint
from aeon.distances import euclidean_distance


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
        init="first",
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
        init="first",
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


def test_medoids_init():
    """Test init algorithms."""
    X_train, _ = load_gunpoint(split="train")
    X_train = X_train[:10]

    num_clusters = 8
    kmedoids = TimeSeriesCLARANS(
        random_state=1,
        n_init=1,
        init="first",
        distance="euclidean",
        n_clusters=num_clusters,
    )

    kmedoids._random_state = check_random_state(kmedoids.random_state)
    kmedoids._distance_cache = np.full((len(X_train), len(X_train)), np.inf)
    kmedoids._distance_callable = euclidean_distance
    first_medoids_result = kmedoids._first_center_initializer(X_train)
    check_value_in_every_cluster(num_clusters, first_medoids_result)
    random_medoids_result = kmedoids._random_center_initializer(X_train)
    check_value_in_every_cluster(num_clusters, random_medoids_result)
    kmedoids_plus_plus_medoids_result = kmedoids._kmedoids_plus_plus_center_initializer(
        X_train
    )
    check_value_in_every_cluster(num_clusters, kmedoids_plus_plus_medoids_result)
    kmedoids_build_result = kmedoids._pam_build_center_initializer(X_train)
    check_value_in_every_cluster(num_clusters, kmedoids_build_result)

    # Test setting manual init centres
    num_clusters = 8
    custom_init_centres = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    kmedoids = TimeSeriesCLARANS(
        random_state=1,
        n_init=1,
        init=custom_init_centres,
        distance="euclidean",
        n_clusters=num_clusters,
    )
    kmedoids.fit(X_train)
    assert np.array_equal(kmedoids.cluster_centers_, X_train[custom_init_centres])
