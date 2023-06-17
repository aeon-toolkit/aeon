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

    X_test = X_test[:num_points]
    y_test = y_test[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    kmedoids = TimeSeriesKMedoids(
        random_state=1,
        n_init=2,
        max_iter=5,
        init_algorithm="first",
        distance="euclidean",
    )
    train_medoids_result = kmedoids.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = kmedoids.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = kmedoids.predict_proba(X_test)
    assert np.array_equal(test_medoids_result, [6, 5, 7, 6, 5, 5, 6, 3, 2, 5])
    assert np.array_equal(train_medoids_result,
                          [0, 1, 2, 5, 4, 5, 6, 7, 3, 6, 6, 6, 6, 6, 1, 6, 2, 1, 6, 3,
                           0, 6, 6, 7, 4, 3, 6, 6, 3, 7, 6, 7, 3, 2, 5, 5, 6, 1, 7, 6,
                           2, 6, 6, 6, 3, 5, 5, 1, 5, 7])
    assert test_score == 0.5777777777777777
    assert train_score == 0.5795918367346938
    assert np.isclose(kmedoids.inertia_, 118.9405520596505)
    assert kmedoids.n_iter_ == 4
    assert np.array_equal(kmedoids.labels_,
                          [0, 1, 2, 5, 4, 5, 6, 7, 3, 6, 6, 6, 6, 6, 1, 6, 2, 1, 6, 3,
                           0, 6, 6, 7, 4, 3, 6, 6, 3, 7, 6, 7, 3, 2, 5, 5, 6, 1, 7, 6,
                           2, 6, 6, 6, 3, 5, 5, 1, 5, 7])
    assert isinstance(kmedoids.cluster_centers_, np.ndarray)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


def test_kmedoids_multi():
    """Test implementation of Kmedoids for multivariate."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 10

    X_test = X_test[:num_points]
    y_test = y_test[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    kmedoids = TimeSeriesKMedoids(
        random_state=1,
        n_init=2,
        max_iter=5,
        init_algorithm="first",
        distance="euclidean",
    )
    train_medoids_result = kmedoids.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = kmedoids.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = kmedoids.predict_proba(X_test)
    assert np.array_equal(test_medoids_result, [4, 5, 4, 0, 5, 5, 5, 5, 4, 4])
    assert np.array_equal(train_medoids_result,
                          [0, 4, 4, 5, 4, 5, 7, 7, 4, 4, 0, 0, 6, 1, 3, 1, 1, 5, 0, 3,
                           4, 6, 3, 1, 7, 3, 2, 2, 1, 3, 6, 0, 7, 2, 6, 6, 7, 2, 0, 6])
    assert test_score == 0.35555555555555557
    assert train_score == 0.7461538461538462
    assert np.isclose(kmedoids.inertia_, 3474.432529578885)
    assert kmedoids.n_iter_ == 3
    assert np.array_equal(kmedoids.labels_,
                          [0, 4, 4, 5, 4, 5, 7, 7, 4, 4, 0, 0, 6, 1, 3, 1, 1, 5, 0, 3,
                           4, 6, 3, 1, 7, 3, 2, 2, 1, 3, 6, 0, 7, 2, 6, 6, 7, 2, 0, 6])
    assert isinstance(kmedoids.cluster_centers_, np.ndarray)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


def check_value_in_every_cluster(num_clusters, initial_medoids):
    """Check that every cluster has at least one value."""
    original_length = len(initial_medoids)
    assert original_length == num_clusters
    original_length == len(set(initial_medoids))


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
        n_clusters=num_clusters
    )
    kmedoids._random_state = check_random_state(kmedoids.random_state)
    kmedoids._distance_cache = np.full((len(X_train), len(X_train)), np.inf)
    kmedoids._distance_callable = euclidean_distance
    first_medoids_result = kmedoids._first_center_initializer(X_train)
    check_value_in_every_cluster(num_clusters, first_medoids_result)
    random_medoids_result = kmedoids._random_center_initializer(X_train)
    check_value_in_every_cluster(num_clusters, random_medoids_result)
    kmedoids_plus_plus_medoids_result = kmedoids._kmedoids_plus_plus(X_train)
    check_value_in_every_cluster(num_clusters, kmedoids_plus_plus_medoids_result)


def time_kmedoids(X_train, X_test, dataset):
    import time
    kmedoids = TimeSeriesKMedoids(
        random_state=1,
        init_algorithm="random",
        distance="dtw",
    )
    start_time = time.time()
    kmedoids.fit(X_train)
    test_medoids_result = kmedoids.predict(X_test)
    end_time = time.time()
    print(f"{dataset} Time elapsed: {end_time - start_time} seconds")


def test_timing_experiment():
    from aeon.datasets import load_basic_motions, load_gunpoint, load_italy_power_demand, load_acsf1
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    time_kmedoids(X_train, X_test, "GunPoint")
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    time_kmedoids(X_train, X_test, "BasicMotions")
    X_train, y_train = load_italy_power_demand(split="train")
    X_test, y_test = load_italy_power_demand(split="test")
    time_kmedoids(X_train, X_test, "ItalyPowerDemand")
    X_train, y_train = load_acsf1(split="train")
    X_test, y_test = load_acsf1(split="test")
    time_kmedoids(X_train, X_test, "ACSF1")



def print_assertion(test_expected_result, train_expected_result, test_score,
                    train_expected_score, clusterer):
    print("")
    print(f"assert np.array_equal(test_medoids_result, {list(test_expected_result)})")
    print(f"assert np.array_equal(train_medoids_result, {list(train_expected_result)})")
    print(f"assert test_score == {test_score}")
    print(f"assert train_score == {train_expected_score}")
    print(f"assert np.isclose(kmedoids.inertia_, {clusterer.inertia_})")
    print(f"assert kmedoids.n_iter_ == {clusterer.n_iter_}")
    print(f"assert np.array_equal(kmedoids.labels_, {list(clusterer.labels_)})")
    print(f"assert isinstance(kmedoids.cluster_centers_, np.ndarray)")
    print(f"for val in proba:")
    print(f"    assert np.count_nonzero(val == 1.0) == 1")
