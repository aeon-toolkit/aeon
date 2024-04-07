"""Test ACF series transformer."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from aeon.transformations.series._acf import AutoCorrelationTransformer
from aeon.utils.validation._dependencies import _check_soft_dependencies


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


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_acf_against_statsmodels():
    """Test ACF series transformer against statsmodels."""
    from statsmodels.tsa.stattools import acf

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sm = acf(x, nlags=3, fft=False)
    acf = AutoCorrelationTransformer(n_lags=4)
    xt = acf.fit_transform(x)
    xt = xt.squeeze()
    assert_array_almost_equal(xt, sm, decimal=5)


def test_multivariate():
    """Test multiple univariate calls the same as a multivariate one."""
    x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    x2 = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    x3 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    n_lags = 3
    acf = AutoCorrelationTransformer(n_lags=n_lags)
    y1 = acf.fit_transform(x1)
    y2 = acf.fit_transform(x2)
    y3 = acf.fit_transform(x3)
    c2 = np.vstack([y1, y2, y3])
    combined = np.vstack([x1, x2, x3])
    c3 = acf.fit_transform(combined)
    assert np.allclose(c2, c3)
