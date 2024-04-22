"""Tests for naive Euclidean distance profile."""

__maintainer__ = []

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from aeon.distances import get_distance_function
from aeon.similarity_search.distance_profiles.euclidean_distance_profile import (
    euclidean_distance_profile,
    normalized_euclidean_distance_profile,
)
from aeon.similarity_search.distance_profiles.naive_distance_profile import (
    naive_distance_profile,
    normalized_naive_distance_profile,
)
from aeon.utils.numba.general import sliding_mean_std_one_series

DATATYPES = ["float64", "int64"]


@pytest.mark.parametrize("dtype", DATATYPES)
def test_euclidean_distance(dtype):
    """Test Euclidean distance."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    mask = np.ones((X.shape[0], X.shape[2] - q.shape[1] + 1), dtype=bool)
    distance = get_distance_function("euclidean")
    expected = naive_distance_profile(X, q, mask, distance)
    dist_profile = euclidean_distance_profile(X, q, mask)

    assert_array_almost_equal(dist_profile, expected)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_euclidean_constant_case(dtype):
    """Test Euclidean distance profile calculation."""
    X = np.ones((2, 1, 10), dtype=dtype)
    q = np.zeros((1, 3), dtype=dtype)

    mask = np.ones((X.shape[0], X.shape[2] - q.shape[1] + 1), dtype=bool)
    distance = get_distance_function("euclidean")
    expected = naive_distance_profile(X, q, mask, distance)
    dist_profile = euclidean_distance_profile(X, q, mask)

    assert_array_almost_equal(dist_profile, expected)


def test_non_alteration_of_inputs_euclidean():
    """Test if input is altered during Euclidean distance profile."""
    X = np.asarray([[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]])
    X_copy = np.copy(X)
    q = np.asarray([[3, 4, 5]])
    q_copy = np.copy(q)

    mask = np.ones((X.shape[0], X.shape[2] - q.shape[1] + 1), dtype=bool)
    _ = euclidean_distance_profile(X, q, mask)
    assert_array_equal(q, q_copy)
    assert_array_equal(X, X_copy)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_normalized_euclidean_distance(dtype):
    """Test normalised Euclidean distance profile calculation."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search_space_size = X.shape[-1] - q.shape[-1] + 1

    X_means = np.zeros((X.shape[0], X.shape[1], search_space_size))
    X_stds = np.zeros((X.shape[0], X.shape[1], search_space_size))

    for i in range(X.shape[0]):
        _mean, _std = sliding_mean_std_one_series(X[i], q.shape[-1], 1)
        X_stds[i] = _std
        X_means[i] = _mean

    q_means = q.mean(axis=-1)
    q_stds = q.std(axis=-1)
    mask = np.ones((X.shape[0], X.shape[2] - q.shape[1] + 1), dtype=bool)

    distance = get_distance_function("euclidean")

    dist_profile = normalized_euclidean_distance_profile(
        X, q, mask, X_means, X_stds, q_means, q_stds
    )
    expected = normalized_naive_distance_profile(
        X, q, mask, X_means, X_stds, q_means, q_stds, distance
    )

    assert_array_almost_equal(dist_profile, expected)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_normalized_euclidean_constant_case(dtype):
    """Test normalised Euclidean distance profile calculation."""
    X = np.ones((2, 2, 10), dtype=dtype)
    q = np.zeros((2, 3), dtype=dtype)

    search_space_size = X.shape[-1] - q.shape[-1] + 1

    q_means = q.mean(axis=-1)
    q_stds = q.std(axis=-1)

    X_means = np.zeros((X.shape[0], X.shape[1], search_space_size))
    X_stds = np.zeros((X.shape[0], X.shape[1], search_space_size))
    for i in range(X.shape[0]):
        _mean, _std = sliding_mean_std_one_series(X[i], q.shape[-1], 1)
        X_stds[i] = _std
        X_means[i] = _mean

    mask = np.ones((X.shape[0], X.shape[2] - q.shape[1] + 1), dtype=bool)
    distance = get_distance_function("euclidean")

    dist_profile = normalized_euclidean_distance_profile(
        X, q, mask, X_means, X_stds, q_means, q_stds
    )
    expected = normalized_naive_distance_profile(
        X, q, mask, X_means, X_stds, q_means, q_stds, distance
    )

    assert_array_almost_equal(dist_profile, expected)


def test_non_alteration_of_inputs_normalized_euclidean():
    """Test if input is altered during normalized Euclidean distance profile."""
    X = np.asarray([[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]])
    X_copy = np.copy(X)
    q = np.asarray([[3, 4, 5]])
    q_copy = np.copy(q)

    search_space_size = X.shape[-1] - q.shape[-1] + 1

    X_means = np.zeros((X.shape[0], X.shape[1], search_space_size))
    X_stds = np.zeros((X.shape[0], X.shape[1], search_space_size))

    for i in range(X.shape[0]):
        _mean, _std = sliding_mean_std_one_series(X[i], q.shape[-1], 1)
        X_stds[i] = _std
        X_means[i] = _mean

    q_means = q.mean(axis=-1)
    q_stds = q.std(axis=-1)

    mask = np.ones((X.shape[0], X.shape[2] - q.shape[1] + 1), dtype=bool)
    _ = normalized_euclidean_distance_profile(
        X, q, mask, X_means, X_stds, q_means, q_stds
    )

    assert_array_equal(q, q_copy)
    assert_array_equal(X, X_copy)
