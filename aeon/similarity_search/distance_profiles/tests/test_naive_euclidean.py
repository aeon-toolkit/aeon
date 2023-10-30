"""Tests for naive Euclidean distance profile."""

__author__ = ["baraline"]

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from aeon.distances import euclidean_distance
from aeon.similarity_search.distance_profiles.naive_euclidean import (
    naive_euclidean_profile,
)

DATATYPES = ["float64"]


@pytest.mark.parametrize("dtype", DATATYPES)
def test_naive_euclidean(dtype):
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    mask = np.ones(X.shape, dtype=bool)
    dist_profile = naive_euclidean_profile(X, q, mask).sum(axis=1)

    expected = np.array(
        [
            [
                euclidean_distance(q, X[j, :, i : i + q.shape[-1]])
                for i in range(X.shape[-1] - q.shape[-1] + 1)
            ]
            for j in range(X.shape[0])
        ]
    )
    assert_array_almost_equal(dist_profile, expected)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_naive_euclidean_constant_case(dtype):
    # Test constant case
    X = np.ones((2, 1, 10), dtype=dtype)
    q = np.zeros((1, 3), dtype=dtype)

    mask = np.ones(X.shape, dtype=bool)
    dist_profile = naive_euclidean_profile(X, q, mask).sum(axis=1)
    # Should be full array for sqrt(3) as q is zeros of length 3 and X is full ones
    search_space_size = X.shape[-1] - q.shape[-1] + 1
    expected = np.array([[3**0.5] * search_space_size] * X.shape[0])
    assert_array_almost_equal(dist_profile, expected)


def test_non_alteration_of_inputs_naive_euclidean():
    X = np.asarray([[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]])
    X_copy = np.copy(X)
    q = np.asarray([[3, 4, 5]])
    q_copy = np.copy(q)

    mask = np.ones(X.shape, dtype=bool)
    _ = naive_euclidean_profile(X, q, mask)
    assert_array_equal(q, q_copy)
    assert_array_equal(X, X_copy)
