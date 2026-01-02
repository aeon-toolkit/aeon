"""Test _commons.py functions."""

__maintainer__ = ["baraline"]
import numpy as np
import pytest
from numpy.testing import assert_, assert_array_almost_equal

from aeon.similarity_search.subsequence._commons import (
    _extract_top_k_from_dist_profile_one_series,
    _inverse_distance_profile,
    fft_sliding_dot_product,
    get_ith_products,
)
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
)

K_VALUES = [1, 3, 5]
THRESHOLDS = [np.inf, 1.5]
NN_MATCHES = [False, True]
EXCLUSION_SIZE = [3, 5]


def test_fft_sliding_dot_product():
    """Test the fft_sliding_dot_product function."""
    L = 4
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=L)

    values = fft_sliding_dot_product(X, Q)
    # Compare values[0] only as input is univariate
    assert_array_almost_equal(
        values[0],
        [np.dot(Q[0], X[0, i : i + L]) for i in range(X.shape[1] - L + 1)],
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


def test__inverse_distance_profile():
    """Test method to inverse a TypedList of distance profiles."""
    X = make_example_1d_numpy()
    X_inv = _inverse_distance_profile(X)
    assert_array_almost_equal(1 / (X + 1e-8), X_inv)


@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("allow_nn_matches", NN_MATCHES)
@pytest.mark.parametrize("exclusion_size", EXCLUSION_SIZE)
def test__extract_top_k_from_dist_profile_one_series(
    k, threshold, allow_nn_matches, exclusion_size
):
    """Test method to esxtract the top k candidates from a list of distance profiles."""
    X = make_example_1d_numpy(n_timepoints=30)
    X_sort = np.argsort(X)
    exclusion_size = 3
    top_k_indexes, top_k_distances = _extract_top_k_from_dist_profile_one_series(
        X, k, threshold, allow_nn_matches, exclusion_size
    )

    if len(top_k_indexes) == 0 or len(top_k_distances) == 0:
        raise AssertionError(
            "_extract_top_k_from_dist_profile_one_series returned empty"
        )
    for i, index in enumerate(top_k_indexes):
        assert_(X[index] == top_k_distances[i])

    assert_(np.all(top_k_distances <= threshold))

    if allow_nn_matches:
        assert_(np.all(top_k_distances <= X[X_sort[len(top_k_indexes) - 1]]))

    if not allow_nn_matches:
        same_X = np.sort(top_k_indexes)
        if len(same_X) > 1:
            assert_(np.all(np.diff(same_X) >= exclusion_size))
