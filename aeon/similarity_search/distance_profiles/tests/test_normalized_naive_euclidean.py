# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 12:21:00 2023

@author: antoi
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from aeon.distances import euclidean_distance
from aeon.similarity_search.distance_profiles.normalized_naive_euclidean import (
    normalized_naive_euclidean_profile,
)
from aeon.utils.numba.general import sliding_mean_std_one_series

DATATYPES = ["int64", "float64"]


@pytest.mark.parametrize("dtype", DATATYPES)
def test_normalized_naive_euclidean(dtype):
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    Q = np.asarray([[3, 4, 5]], dtype=dtype)

    search_space_size = X.shape[-1] - Q.shape[-1] + 1

    X_means = np.zeros((X.shape[0], X.shape[1], search_space_size))
    X_stds = np.zeros((X.shape[0], X.shape[1], search_space_size))

    for i in range(X.shape[0]):
        _mean, _std = sliding_mean_std_one_series(X[i], Q.shape[-1], 1)
        X_stds[i] = _std
        X_means[i] = _mean

    Q_means = Q.mean(axis=-1)
    Q_stds = Q.std(axis=-1)
    dist_profile = normalized_naive_euclidean_profile(
        X, Q, X_means, X_stds, Q_means, Q_stds
    )

    _Q = Q.copy()
    for k in range(Q.shape[0]):
        _Q[k] = (_Q[k] - Q_means[k]) / Q_stds[k]

    expected = np.full(dist_profile.shape, np.inf)
    for i in range(X.shape[0]):
        for j in range(search_space_size):
            _C = X[i, :, j : j + Q.shape[-1]].copy()
            for k in range(X.shape[1]):
                _C[k] = (_C[k] - X_means[i, k, j]) / X_stds[i, k, j]
            expected[i, j] = euclidean_distance(_Q, _C)

    assert_array_almost_equal(dist_profile, expected)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_normalized_naive_euclidean_constant_case(dtype):
    # Test constant case
    X = np.ones((2, 2, 10), dtype=dtype)
    Q = np.zeros((2, 3), dtype=dtype)

    search_space_size = X.shape[-1] - Q.shape[-1] + 1

    Q_means = Q.mean(axis=-1, keepdims=True)
    Q_stds = Q.std(axis=-1, keepdims=True)

    X_means = np.zeros((X.shape[0], X.shape[1], search_space_size))
    X_stds = np.zeros((X.shape[0], X.shape[1], search_space_size))
    for i in range(X.shape[0]):
        _mean, _std = sliding_mean_std_one_series(X[i], Q.shape[-1], 1)
        X_stds[i] = _std
        X_means[i] = _mean

    dist_profile = normalized_naive_euclidean_profile(
        X, Q, X_means, X_stds, Q_means, Q_stds
    )
    # Should be full array for 0

    expected = np.array([[0] * search_space_size] * X.shape[0])
    assert_array_almost_equal(dist_profile, expected)


def test_non_alteration_of_inputs_normalized_naive_euclidean():
    X = np.asarray([[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]])
    X_copy = np.copy(X)
    Q = np.asarray([[3, 4, 5]])
    Q_copy = np.copy(Q)

    search_space_size = X.shape[-1] - Q.shape[-1] + 1

    X_means = np.zeros((X.shape[0], X.shape[1], search_space_size))
    X_stds = np.zeros((X.shape[0], X.shape[1], search_space_size))

    for i in range(X.shape[0]):
        _mean, _std = sliding_mean_std_one_series(X[i], Q.shape[-1], 1)
        X_stds[i] = _std
        X_means[i] = _mean

    Q_means = Q.mean(axis=-1, keepdims=True)
    Q_stds = Q.std(axis=-1, keepdims=True)

    _ = normalized_naive_euclidean_profile(X, Q, X_means, X_stds, Q_means, Q_stds)

    assert_array_equal(Q, Q_copy)
    assert_array_equal(X, X_copy)
