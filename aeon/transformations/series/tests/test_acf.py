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


def test_acf_n_lags_clamped_to_one():
    """Test n_lags below 1 is clamped to 1."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    acf = AutoCorrelationSeriesTransformer(n_lags=0)
    y = acf.fit_transform(x)
    assert y.shape == (1, 1)


def test_acf_raises_when_lags_too_large_for_series():
    """Test a ValueError is raised when n_lags leaves too few observations."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    acf = AutoCorrelationSeriesTransformer(n_lags=3)
    with pytest.raises(ValueError, match="The number of lags is too large"):
        acf.fit_transform(x)


def test_acf_one_zero_variance_window_gives_zero_correlation():
    """Test a lag where only one window has zero variance returns 0."""
    x = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 9.0, 3.0, 7.0, 2.0])
    acf = AutoCorrelationSeriesTransformer(n_lags=5)
    y = acf.fit_transform(x)
    assert y[0, -1] == 0.0


def test_acf_default_n_lags_is_quarter_series_length():
    """Test n_lags defaults to max(1, n_timepoints / 4) when not specified."""
    x = np.arange(20, dtype=float)
    acf = AutoCorrelationSeriesTransformer()
    y = acf.fit_transform(x)
    assert y.shape == (1, 5)


def test_acf_get_test_params():
    """Test the default test parameters are valid and usable."""
    for params in AutoCorrelationSeriesTransformer._get_test_params():
        transformer = AutoCorrelationSeriesTransformer(**params)
        assert isinstance(transformer, AutoCorrelationSeriesTransformer)
