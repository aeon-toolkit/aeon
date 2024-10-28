"""Tests for MovingAverageTransformer."""

__maintainer__ = ["Datadote"]

import numpy as np
import pytest

from aeon.transformations.series._moving_average import MovingAverageSeriesTransformer

TEST_DATA = [np.array([-3, -2, -1, 0, 1, 2, 3]), np.array([[-3, -2, -1, 0, 1, 2, 3]])]
EXPECTED_RESULTS = [
    np.array([[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]]),
    np.array([[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]]),
]


def test_window_size_greater_than_zero():
    """Test window sizes > 0."""
    ma = MovingAverageSeriesTransformer(window_size=1)
    xt = ma.fit_transform(TEST_DATA[0])
    np.testing.assert_array_almost_equal(xt, xt, decimal=2)

    ma = MovingAverageSeriesTransformer(window_size=2)
    for i in range(len(TEST_DATA)):
        xt = ma.fit_transform(TEST_DATA[i])
        np.testing.assert_array_almost_equal(xt, EXPECTED_RESULTS[i], decimal=2)


def test_window_size_equal_zero():
    """Test window size == 0."""
    with pytest.raises(ValueError):
        MovingAverageSeriesTransformer(window_size=0)


def test_window_size_less_than_zero():
    """Test window sizes < 0."""
    with pytest.raises(ValueError):
        MovingAverageSeriesTransformer(window_size=-1)
