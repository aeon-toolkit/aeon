"""Tests for ExpSmoothingSeriesTransformer."""

__maintainer__ = ["Datadote"]

import numpy as np
import pytest

from aeon.transformations.series._exp_smoothing import ExpSmoothingSeriesTransformer

TEST_DATA = [np.array([-2, -1, 0, 1, 2]), np.array([[1, 2, 3, 4], [10, 9, 8, 7]])]
EXPECTED_RESULTS = [
    np.array([[-2.0, -1.5, -0.75, 0.125, 1.0625]]),
    np.array([[1.0, 1.5, 2.25, 3.125], [10.0, 9.5, 8.75, 7.875]]),
]


def test_input_1d_array():
    """Test inputs of dimension 1."""
    transformer = ExpSmoothingSeriesTransformer(0.5)
    idx_data = 0
    Xt = transformer.fit_transform(TEST_DATA[idx_data])
    np.testing.assert_almost_equal(Xt, EXPECTED_RESULTS[idx_data], decimal=5)


def test_input_2d_array():
    """Test inputs of dimension 2."""
    transformer = ExpSmoothingSeriesTransformer(0.5)
    idx_data = 1
    Xt = transformer.fit_transform(TEST_DATA[idx_data])
    np.testing.assert_almost_equal(Xt, EXPECTED_RESULTS[idx_data], decimal=5)


@pytest.mark.parametrize("alpha_window", [(0.2, 9), (0.5, 3), (1, 1)])
def test_window_size_matches_alpha(alpha_window):
    """Check same output results using equivalent alpha and window_size."""
    alpha, window_size = alpha_window
    transformer1 = ExpSmoothingSeriesTransformer(alpha=alpha)
    transformer2 = ExpSmoothingSeriesTransformer(window_size=window_size)
    for i in range(len(TEST_DATA)):
        Xt1 = transformer1.fit_transform(TEST_DATA[i])
        Xt2 = transformer2.fit_transform(TEST_DATA[i])
        np.testing.assert_array_almost_equal(Xt1, Xt2, decimal=5)


def test_alpha_less_than_zero():
    """Test alpha less than zero."""
    with pytest.raises(ValueError):
        ExpSmoothingSeriesTransformer(-0.5)


def test_alpha_greater_than_one():
    """Test alpha greater than one."""
    with pytest.raises(ValueError):
        ExpSmoothingSeriesTransformer(2.0)


def test_window_size_than_one():
    """Test window_size < 0."""
    with pytest.raises(ValueError):
        ExpSmoothingSeriesTransformer(window_size=0)
