"""Test _commons.py functions."""

__maintainer__ = ["baraline"]

import numpy as np
from numba.typed import List
from numpy.testing import assert_array_almost_equal, assert_array_equal

from aeon.similarity_search.subsequence_search._commons import (
    _inverse_distance_profile_list,
    fft_sliding_dot_product,
    get_ith_products,
)
from aeon.testing.data_generation import (
    make_example_2d_numpy_list,
    make_example_2d_numpy_series,
)


def test_fft_sliding_dot_product():
    """Test the fft_sliding_dot_product function."""
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=4)

    values = fft_sliding_dot_product(X, Q)
    # Compare values[0] only as input is univariate
    assert_array_almost_equal(
        values[0],
        [np.dot(Q[0], X[0, i : i + 5]) for i in range(X.shape[1] - 5 + 1)],
    )


def test_get_ith_products():
    """Test i-th dot product of a subsequence of size L."""
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    L = 5

    values = get_ith_products(X, Q, L, 0)
    # Compare values[0] only as input is univariate
    assert_array_almost_equal(
        values[0],
        [np.dot(Q[0, 0:L], X[0, i : i + L]) for i in range(X.shape[1] - L + 1)],
    )

    values = get_ith_products(X, Q, L, 4)
    # Compare values[0] only as input is univariate
    assert_array_almost_equal(
        values[0],
        [np.dot(Q[0, 4 : 4 + L], X[0, i : i + L]) for i in range(X.shape[1] - L + 1)],
    )


def test__inverse_distance_profile_list():
    """Test method to inverse a TypedList of distance profiles."""
    X = make_example_2d_numpy_list(n_cases=2, return_y=False)
    T = _inverse_distance_profile_list(List(X))
    assert_array_almost_equal(1 / (X[0] + 1e-8), T[0])
    assert_array_almost_equal(1 / (X[1] + 1e-8), T[1])


def test__extract_top_k_from_dist_profile():
    """Test method to esxtract the top k candidates from a list of distance profiles."""
    X = List([
        [5,4,3,3,1,3,2,5,1,4,1,0,1,2,2,7,8,1,5],
        [5,1,0,1,0,0,5,4,3,5,6,1,4,2],
    ])

    top_k_indexes, top_k_distances = _extract_top_k_from_dist_profile(
        X,
        1,
        np.inf,
        False,
        3
    )
    assert_array_equal(top_k_indexes, [[0,11]])
    assert_array_equal(top_k_indexes, [0])
    
    top_k_indexes, top_k_distances = _extract_top_k_from_dist_profile(
        X,
        5,
        np.inf,
        False,
        3
    )
    assert_array_equal(top_k_indexes, [[0,11],[1,2],[0,4],[0,17],[1,11]])
    assert_array_equal(top_k_indexes, [0,0,1,1,1])

    top_k_indexes, top_k_distances = _extract_top_k_from_dist_profile(
        X,
        5,
        np.inf,
        True,
        3
    )
    assert_array_equal(top_k_indexes, [[0,11],[1,2],[1,4],[1,5],[0,4]])
    assert_array_equal(top_k_indexes, [0,0,0,0,1])

    top_k_indexes, top_k_distances = _extract_top_k_from_dist_profile(
        X,
        5,
        0.5,
        True,
        3
    )
    assert_array_equal(top_k_indexes, [[0,11],[1,2],[1,4],[1,5]])
    assert_array_equal(top_k_indexes, [0,0,0,0])

    top_k_indexes, top_k_distances = _extract_top_k_from_dist_profile(
        X,
        5,
        0.5,
        False,
        3
    )
    assert_array_equal(top_k_indexes, [[0,11],[1,2]])
    assert_array_equal(top_k_indexes, [0,0])
