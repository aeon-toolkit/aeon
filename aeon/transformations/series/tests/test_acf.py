"""Test ACF series transformer."""

import numpy as np
import pytest

from aeon.transformations.series._acf import AutoCorrelationTransformer


def test_acf():
    """Test ACF series transformer returned size."""
    n_lags = 3
    acf = AutoCorrelationTransformer(n_lags=n_lags)
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


def test_multivariate():
    """Test multiple univariate calls the same as a multivariate one."""
    pass
    # x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # x2 = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    # x3 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # n_lags = 3
    # acf = AutoCorrelationTransformer(n_lags=n_lags)
    # y1 = acf.fit_transform(x1)
    # y2 = acf.fit_transform(x2)
    # y3 = acf.fit_transform(x3)
