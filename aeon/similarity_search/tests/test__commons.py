"""Test _commons.py functions."""

__maintainer__ = ["baraline"]

import numpy as np
from numpy.testing import assert_array_almost_equal

from aeon.similarity_search._commons import (
    extract_top_k_and_threshold_from_distance_profiles,
    fft_sliding_dot_product,
    naive_squared_distance_profile,
    naive_squared_matrix_profile,
    numba_roll_1D_no_warparound,
    numba_roll_2D_no_warparound,
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


def test_extract_top_k_and_threshold_from_distance_profiles():
    """Test the extract_top_k_and_threshold_from_distance_profiles function."""
    k = 2
    X = np.array(
        [
            [0.48656398, 0.42053769, 0.67763485, 0.80750033],
            [0.29294077, 0.85502115, 0.17414422, 0.87988586],
            [0.02714461, 0.57553083, 0.53823929, 0.08922194],
        ]
    )

    p, q = extract_top_k_and_threshold_from_distance_profiles(X, k)
    assert_array_almost_equal(p, np.array([0.02714461, 0.08922194]))


# has bugs
# def test_extract_top_k_and_threshold_from_distance_profiles_one_series():
#     pass


def test_numba_roll_2D_no_warparound():
    """Test the numba_roll_2D_no_warparound function."""
    shift = 2
    warparound = 14
    X = np.array(
        [[0.93306621, 0.46541855, 0.80534776], [0.86205769, 0.07086389, 0.38304427]]
    )
    result = numba_roll_2D_no_warparound(X, shift, warparound)
    assert_array_almost_equal(
        result, np.array([[14.0, 14.0, 0.93306621], [14.0, 14.0, 0.86205769]])
    )


def test_numba_roll_1D_no_warpaorund():
    """Test the numba_roll_1D_no_warparound function."""
    shift = 2
    warparound = 23
    X = np.array([0.73828259, 0.6035077, 0.31581101, 0.03536085, 0.22670591])
    result = numba_roll_1D_no_warparound(X, shift, warparound)
    assert_array_almost_equal(
        result, np.array([23.0, 23.0, 0.73828259, 0.6035077, 0.31581101])
    )
