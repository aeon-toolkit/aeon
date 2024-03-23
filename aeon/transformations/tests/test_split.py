"""Tests for SplitsTimeSeries."""

import numpy as np
import pytest

from aeon.transformations._split import SplitsTimeSeries

X = np.arange(10)
testdata = [
    (X, 2, [np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9])]),
    (X, 3, [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8, 9])]),
]


@pytest.mark.parametrize("X,n_intervals,expected", testdata)
def test_split_(X, n_intervals, expected):
    """Test the splitting of a time series into multiple intervals."""
    splitter = SplitsTimeSeries()
    splitter.n_intervals = n_intervals
    res = splitter._split(X)

    assert len(res) == n_intervals
    for x, y in zip(res, expected):
        np.testing.assert_array_equal(x, y)
