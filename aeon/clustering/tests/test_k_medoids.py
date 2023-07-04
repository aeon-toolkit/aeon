# -*- coding: utf-8 -*-
"""Tests for time series k-medoids."""
import numpy as np
from sklearn import metrics
from sklearn.utils import check_random_state

from aeon.clustering.k_medoids import TimeSeriesKMedoids
from aeon.datasets import load_basic_motions, load_gunpoint
from aeon.distances import euclidean_distance


def test_kmedoids_uni():
    """Test implementation of Kmedoids."""
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")

    num_points = 10

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]
    _alternate_uni_medoids(X_train, y_train, X_test, y_test)
    _pam_uni_medoids(X_train, y_train, X_test, y_test)


def test_kmedoids_multi():
    """Test implementation of Kmedoids for multivariate."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 10

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]
    _alternate_multi_medoids(X_train, y_train, X_test, y_test)
    _pam_multi_medoids(X_train, y_train, X_test, y_test)


def _pam_uni_medoids(X_train, y_train, X_test, y_test):
    kmedoids = TimeSeriesKMedoids(
        random_state=1,
        n_init=2,
        max_iter=5,
        init_algorithm="first",
        distance="euclidean",
        method="pam",
    )
    train_medoids_result = kmedoids.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = kmedoids.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = kmedoids.predict_proba(X_test)
    assert np.array_equal(test_medoids_result, [1, 3, 7, 1, 3, 5, 1, 6, 2, 5])
    assert np.array_equal(train_medoids_result, [0, 2, 2, 3, 4, 5, 6, 7, 1, 1])
    assert test_score == 0.5777777777777777
    assert train_score == 0.4222222222222222
    assert np.isclose(kmedoids.inertia_, 10.61819833154311)
    assert kmedoids.n_iter_ == 2
    assert np.array_equal(kmedoids.labels_, [0, 2, 2, 3, 4, 5, 6, 7, 1, 1])
    assert isinstance(kmedoids.cluster_centers_, np.ndarray)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


def _alternate_uni_medoids(X_train, y_train, X_test, y_test):
    kmedoids = TimeSeriesKMedoids(
        random_state=1,
        n_init=2,
        max_iter=5,
        method="alternate",
        init_algorithm="first",
        distance="euclidean",
    )
    train_medoids_result = kmedoids.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = kmedoids.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = kmedoids.predict_proba(X_test)
    assert np.array_equal(test_medoids_result, [6, 1, 7, 6, 3, 5, 6, 6, 2, 5])
    assert np.array_equal(train_medoids_result, [0, 1, 2, 3, 4, 5, 6, 7, 6, 6])
    assert test_score == 0.5333333333333333
    assert train_score == 0.4444444444444444
    assert np.isclose(kmedoids.inertia_, 13.942955492757196)
    assert kmedoids.n_iter_ == 2
    assert np.array_equal(kmedoids.labels_, [0, 1, 2, 3, 4, 5, 6, 7, 6, 6])
    assert isinstance(kmedoids.cluster_centers_, np.ndarray)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


def _pam_multi_medoids(X_train, y_train, X_test, y_test):
    kmedoids = TimeSeriesKMedoids(
        random_state=1,
        n_init=2,
        max_iter=5,
        init_algorithm="first",
        distance="euclidean",
        method="pam",
    )
    train_medoids_result = kmedoids.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = kmedoids.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = kmedoids.predict_proba(X_test)
    assert np.array_equal(test_medoids_result, [1, 5, 1, 0, 5, 3, 5, 5, 3, 4])
    assert np.array_equal(train_medoids_result, [0, 1, 2, 3, 5, 5, 6, 7, 5, 4])
    assert test_score == 0.17777777777777778
    assert train_score == 0.06666666666666667
    assert np.isclose(kmedoids.inertia_, 15.512444347419628)
    assert kmedoids.n_iter_ == 2
    assert np.array_equal(kmedoids.labels_, [0, 1, 2, 3, 5, 5, 6, 7, 5, 4])
    assert isinstance(kmedoids.cluster_centers_, np.ndarray)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


def _alternate_multi_medoids(X_train, y_train, X_test, y_test):
    kmedoids = TimeSeriesKMedoids(
        random_state=1,
        n_init=2,
        max_iter=5,
        init_algorithm="first",
        method="alternate",
        distance="euclidean",
    )
    train_medoids_result = kmedoids.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = kmedoids.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = kmedoids.predict_proba(X_test)

    assert np.array_equal(test_medoids_result, [1, 5, 1, 0, 5, 3, 5, 5, 3, 4])
    assert np.array_equal(train_medoids_result, [0, 1, 2, 3, 4, 5, 6, 7, 4, 4])
    assert test_score == 0.17777777777777778
    assert train_score == 0.06666666666666667
    assert np.isclose(kmedoids.inertia_, 20.66525401745263)
    assert kmedoids.n_iter_ == 2
    assert np.array_equal(kmedoids.labels_, [0, 1, 2, 3, 4, 5, 6, 7, 4, 4])
    assert isinstance(kmedoids.cluster_centers_, np.ndarray)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


def check_value_in_every_cluster(num_clusters, initial_medoids):
    """Check that every cluster has at least one value."""
    original_length = len(initial_medoids)
    assert original_length == num_clusters
    assert original_length == len(set(initial_medoids))


def test_medoids_init():
    """Test implementation of Kmedoids."""
    X_train, y_train = load_gunpoint(split="train")
    X_train = X_train[:10]

    num_clusters = 8
    kmedoids = TimeSeriesKMedoids(
        random_state=1,
        n_init=1,
        max_iter=5,
        init_algorithm="first",
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
