"""Test _commons.py functions."""

__maintainer__ = ["baraline"]

import numpy as np
from numpy.testing import assert_array_almost_equal

from aeon.similarity_search._commons import (
    fft_sliding_dot_product,
    naive_squared_distance_profile,
    naive_squared_matrix_profile,
)


def test_fft_sliding_dot_product():
    """Test the fft_sliding_dot_product function."""
    X = np.random.rand(1, 10)
    q = np.random.rand(1, 5)

    values = fft_sliding_dot_product(X, q)

    assert_array_almost_equal(
        values[0],
        [np.dot(q[0], X[0, i : i + 5]) for i in range(X.shape[1] - 5 + 1)],
    )


def test_naive_squared_distance_profile():
    """Test naive squared distance profile computation is correct."""
    X = np.zeros((1, 1, 6))
    X[0, 0] = np.arange(6)
    Q = np.array([[1, 2, 3]])
    query_length = Q.shape[1]
    mask = np.ones((X.shape[0], X.shape[2] - query_length + 1), dtype=bool)
    dist_profile = naive_squared_distance_profile(X, Q, mask)
    assert_array_almost_equal(dist_profile[0], np.array([3.0, 0.0, 3.0, 12.0]))


def test_naive_squared_matrix_profile():
    """Test naive squared matrix profile computation is correct."""
    X = np.zeros((1, 1, 6))
    X[0, 0] = np.arange(6)
    Q = np.zeros((1, 6))

    Q[0] = np.arange(6, 12)
    query_length = 3
    mask = np.ones((X.shape[0], X.shape[2] - query_length + 1), dtype=bool)
    matrix_profile = naive_squared_matrix_profile(X, Q, query_length, mask)
    assert_array_almost_equal(matrix_profile, np.array([27.0, 48.0, 75.0, 108.0]))
