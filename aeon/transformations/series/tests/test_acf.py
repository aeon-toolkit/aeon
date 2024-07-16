"""Test ACF series transformer."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from aeon.transformations.series._acf import AutoCorrelationSeriesTransformer


def test_acf():
    """Test ACF series transformer returned size."""
    n_lags = 3
    acf = AutoCorrelationSeriesTransformer(n_lags=n_lags)
    # Output format, should return an array length n_timepoints-lags for univariate
    # and multivariate time series
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    if len(x) - n_lags < 3:
        with pytest.raises(ValueError, match="The number of lags is too large"):
            y = acf.fit_transform(x)
    y = acf.fit_transform(x)
    assert y.shape == (1, n_lags)
    y = acf.fit_transform(x, axis=0)
    assert y.shape == (n_lags, 1)
    # Test with axis = 0


TEST_DATA = [
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    np.array([-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]),
    np.array([1.0, 5.0, 3.0, 4.0, 2.0, 1.0, 3.0, 4.0, 5.0, 2.0]),
]
EXPECTED_RESULTS = [
    np.array([1.0, 1.0, 1.0, 1.0]),
    np.array([1.0, 1.0, 1.0, 1.0]),
    np.array([-0.18797908, -0.22454436, -0.70898489, -0.11094004]),
]


def test_acf_against_expected():
    """Test ACF series transformer against expected results."""
    acf = AutoCorrelationSeriesTransformer(n_lags=4)
    for i in range(len(TEST_DATA)):
        xt = acf.fit_transform(TEST_DATA[i])
        xt = xt.squeeze()
        assert_array_almost_equal(xt, EXPECTED_RESULTS[i], decimal=5)


NORMED_TEST_DATA = [
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    np.array([-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]),
    np.array([1.0, 5.0, 3.0, 4.0, 2.0, 1.0, 3.0, 4.0, 5.0, 2.0]),
]
NORMED_EXPECTED_RESULTS = [
    np.array([1.0, 1.0, 1.0, 1.0]),
    np.array([1.0, 1.0, 1.0, 1.0]),
    np.array([-0.18797908, -0.22454436, -0.70898489, -0.11094004]),
]


def test_multivariate():
    """Test multiple univariate calls the same as a multivariate one."""
    x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    x2 = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    x3 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    n_lags = 3
    acf = AutoCorrelationSeriesTransformer(n_lags=n_lags)
    y1 = acf.fit_transform(x1)
    y2 = acf.fit_transform(x2)
    y3 = acf.fit_transform(x3)
    c2 = np.vstack([y1, y2, y3])
    combined = np.vstack([x1, x2, x3])
    c3 = acf.fit_transform(combined)
    assert np.allclose(c2, c3)
