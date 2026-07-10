"""Test for incorrect input for ARCoefficientTransformer transformer."""

import numpy as np
import pytest

from aeon.transformations.collection import AutocorrelationFunctionTransformer


def test_acf():
    """Test ACF Transformer exceptions."""
    X = np.random.random((2, 1, 10))
    acf = AutocorrelationFunctionTransformer(n_lags=100)
    with pytest.raises(ValueError, match=r"must be smaller than n_timepoints - 1"):
        acf.fit_transform(X)


def test_acf_default_n_lags():
    """Test the default n_lags=None resolves to an integer n_timepoints // 4."""
    X = np.random.random((4, 2, 20))
    default = AutocorrelationFunctionTransformer().fit_transform(X)
    # default lags = n_timepoints / 4 = 5
    assert default.shape == (4, 2, 5)
    explicit = AutocorrelationFunctionTransformer(n_lags=5).fit_transform(X)
    np.testing.assert_allclose(default, explicit)
