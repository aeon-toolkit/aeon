"""Tests for naive Euclidean distance profile."""

__maintainer__ = []

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from aeon.distances import euclidean_distance, get_distance_function
from aeon.similarity_search.distance_profiles.naive_distance_profile import (
    naive_distance_profile,
    normalized_naive_distance_profile,
)
from aeon.utils.numba.general import sliding_mean_std_one_series

DATATYPES = ["float64"]
DISTANCES = ["euclidean", "dtw", "lcss"]


@pytest.mark.parametrize("dtype", DATATYPES)
@pytest.mark.parametrize("distance_str", DISTANCES)
def test_naive_distance(dtype, distance_str):
    X = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=dtype)
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    mask = np.ones((X.shape[1] - q.shape[1] + 1), dtype=bool)
    distance = get_distance_function(distance_str)
    dist_profile = naive_distance_profile(X, q, mask, distance).sum(axis=0)

    expected = np.array(
        [
            distance(q, X[:, i : i + q.shape[-1]])
            for i in range(X.shape[1] - q.shape[1] + 1)
        ]
    )
    assert_array_almost_equal(dist_profile, expected)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_naive_euclidean_constant_case(dtype):
    # Test constant case
    X = np.ones((1, 10), dtype=dtype)
    q = np.zeros((1, 3), dtype=dtype)

    mask = np.ones((X.shape[1] - q.shape[1] + 1), dtype=bool)
    dist_profile = naive_distance_profile(X, q, mask, euclidean_distance).sum(axis=0)
    # Should be full array for sqrt(3) as q is zeros of length 3 and X is full ones
    search_space_size = X.shape[1] - q.shape[1] + 1
    expected = np.array([3**0.5] * search_space_size)
    assert_array_almost_equal(dist_profile, expected)


def test_non_alteration_of_inputs_naive_euclidean():
    X = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]])
    X_copy = np.copy(X)
    q = np.asarray([[3, 4, 5]])
    q_copy = np.copy(q)

    mask = np.ones((X.shape[1] - q.shape[1] + 1), dtype=bool)
    _ = naive_distance_profile(X, q, mask, euclidean_distance)
    assert_array_equal(q, q_copy)
    assert_array_equal(X, X_copy)


@pytest.mark.parametrize("dtype", DATATYPES)
@pytest.mark.parametrize("distance_str", DISTANCES)
def test_normalized_naive_distance(dtype, distance_str):
    X = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=dtype)
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search_space_size = X.shape[1] - q.shape[1] + 1

    X_means, X_stds = sliding_mean_std_one_series(X, q.shape[1], 1)

    q_means = q.mean(axis=-1)
    q_stds = q.std(axis=-1)
    mask = np.ones((X.shape[1] - q.shape[1] + 1), dtype=bool)

    distance = get_distance_function(distance_str)
    dist_profile = normalized_naive_distance_profile(
        X, q, mask, X_means, X_stds, q_means, q_stds, distance
    )
    dist_profile = dist_profile.sum(axis=0)

    _q = q.copy()
    for k in range(q.shape[0]):
        _q[k] = (_q[k] - q_means[k]) / q_stds[k]

    expected = np.full(dist_profile.shape, np.inf)

    for j in range(search_space_size):
        _C = X[:, j : j + q.shape[1]].copy()
        for k in range(X.shape[0]):
            _C[k] = (_C[k] - X_means[k, j]) / X_stds[k, j]
        expected[j] = distance(_q, _C)

    assert_array_almost_equal(dist_profile, expected)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_normalized_naive_euclidean_constant_case(dtype):
    # Test constant case
    X = np.ones((2, 10), dtype=dtype)
    q = np.zeros((2, 3), dtype=dtype)

    search_space_size = X.shape[1] - q.shape[1] + 1

    q_means = q.mean(axis=1)
    q_stds = q.std(axis=1)

    X_means, X_stds = sliding_mean_std_one_series(X, q.shape[1], 1)

    mask = np.ones((X.shape[1] - q.shape[1] + 1), dtype=bool)
    dist_profile = normalized_naive_distance_profile(
        X, q, mask, X_means, X_stds, q_means, q_stds, euclidean_distance
    ).sum(axis=0)

    expected = np.zeros(search_space_size)
    assert_array_almost_equal(dist_profile, expected)


def test_non_alteration_of_inputs_normalized_naive_euclidean():
    X = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]])
    X_copy = np.copy(X)
    q = np.asarray([[3, 4, 5]])
    q_copy = np.copy(q)

    X_means, X_stds = sliding_mean_std_one_series(X, q.shape[1], 1)

    q_means = q.mean(axis=1)
    q_stds = q.std(axis=1)

    mask = np.ones((X.shape[1] - q.shape[1] + 1), dtype=bool)
    _ = normalized_naive_distance_profile(
        X, q, mask, X_means, X_stds, q_means, q_stds, euclidean_distance
    )

    assert_array_equal(q, q_copy)
    assert_array_equal(X, X_copy)
