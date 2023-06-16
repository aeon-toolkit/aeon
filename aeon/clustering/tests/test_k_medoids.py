# -*- coding: utf-8 -*-
"""Tests for time series k-medoids."""
import numpy as np
from sklearn import metrics

from aeon.clustering.k_medoids import TimeSeriesKMedoids
from aeon.datasets import load_basic_motions

expected_results = {
    "medoids": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        4,
        0,
        3,
        0,
        0,
        0,
        5,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
}

expected_score = {"medoids": 0.3153846153846154}

train_expected_score = {"medoids": 0.48717948717948717}

expected_inertia = {"medoids": 2383.9806075295196}

expected_iters = {"medoids": 2}

expected_labels = {
    "medoids": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3,
        0,
        0,
        5,
        2,
        4,
        1,
        0,
        4,
        4,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        7,
        0,
        0,
        6,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
}


# def test_kmedoids():
#     """Test implementation of Kmedoids."""
#     X_train, y_train = load_basic_motions(split="train")
#     X_test, y_test = load_basic_motions(split="test")
#
#     kmedoids = TimeSeriesKMedoids(
#         random_state=1,
#         n_init=2,
#         max_iter=5,
#         init_algorithm="random",
#         distance="euclidean",
#     )
#     train_predict = kmedoids.fit_predict(X_train)
#     train_score = metrics.rand_score(y_train, train_predict)
#     test_medoids_result = kmedoids.predict(X_test)
#     medoids_score = metrics.rand_score(y_test, test_medoids_result)
#     proba = kmedoids.predict_proba(X_test)
#     assert np.array_equal(test_medoids_result, expected_results["medoids"])
#     assert medoids_score == expected_score["medoids"]
#     assert train_score == train_expected_score["medoids"]
#     assert np.isclose(kmedoids.inertia_, expected_inertia["medoids"])
#     assert kmedoids.n_iter_ == expected_iters["medoids"]
#     assert np.array_equal(kmedoids.labels_, expected_labels["medoids"])
#     assert isinstance(kmedoids.cluster_centers_, np.ndarray)
#     assert proba.shape == (40, 8)
#
#     for val in proba:
#         assert np.count_nonzero(val == 1.0) == 1
#
def test_kmedoids_multi():
    """Test implementation of Kmedoids."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    kmedoids = TimeSeriesKMedoids(
        random_state=1,
        n_init=2,
        max_iter=5,
        init_algorithm="random",
        distance="euclidean",
    )
    train_predict = kmedoids.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_predict)
    test_medoids_result = kmedoids.predict(X_test)
    medoids_score = metrics.rand_score(y_test, test_medoids_result)
    from numpy.testing import assert_almost_equal
    joe = ""

    assert_almost_equal(train_predict,
                        [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 5, 3, 7, 1, 3, 3,
                         6, 6, 6, 6, 1, 6, 6, 6, 6, 6, 6, 6, 4, 6, 6, 0, 6, 6, 6, 2])
    assert train_score == 0.5205128205128206
    assert medoids_score == 0.367948717948718
    assert_almost_equal(test_medoids_result,
                        [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 6, 6, 1, 1, 6, 6, 6, 6, 1, 6,
                         6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1])


def test_new_kmedoids():
    from aeon.datasets import load_gunpoint as load_data
    X_train, y_train = load_data(split="train")
    X_test, y_test = load_data(split="test")

    kmedoids = TimeSeriesKMedoids(
        n_clusters=len(set(y_train)),
        random_state=1,
        n_init=2,
        max_iter=5,
        init_algorithm="random",
        distance="euclidean",
    )
    train_predict = kmedoids.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_predict)
    test_medoids_result = kmedoids.predict(X_test)
    medoids_score = metrics.rand_score(y_test, test_medoids_result)
    proba = kmedoids.predict_proba(X_test)
    joe = ""
