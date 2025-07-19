"""
Tests for stomp algorithm.

We do not test equality for returned indexes due to the unstable nature of argsort
and the fact that the "kind=stable" parameter is not yet supported in numba. We instead
test that the returned index match the expected distance value.
"""

__maintainer__ = ["baraline"]

import numpy as np
from numpy.testing import assert_almost_equal

from aeon.similarity_search.series.neighbors._dummy import (
    _naive_squared_distance_profile,
)
from aeon.testing.data_generation import make_example_2d_numpy_series
from aeon.utils.numba.general import get_all_subsequences


def test__naive_squared_distance_profile():
    """Test Euclidean distance with brute force."""
    L = 3
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=L)

    dist_profile = _naive_squared_distance_profile(get_all_subsequences(X, L, 1), Q)

    for i_t in range(X.shape[1] - L + 1):
        S = X[:, i_t : i_t + L]
        assert_almost_equal(dist_profile[i_t], np.sum((S - Q) ** 2))
