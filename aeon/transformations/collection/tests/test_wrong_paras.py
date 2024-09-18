"""Test for incorrect input for Collections transformers."""

import numpy as np
import pytest

from aeon.transformations.collection import (
    ARCoefficientTransformer,
    AutocorrelationFunctionTransformer,
    HOG1DTransformer,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency statsmodels not available",
)
def test_ar_coefficient():
    """Test AR Coefficient Transformer exceptions."""
    X = np.random.random((2, 1, 10))
    ar = ARCoefficientTransformer(order=0)
    Xt = ar.fit_transform(X)
    assert Xt.shape[2] == 1
    ar = ARCoefficientTransformer(order=-10)
    Xt = ar.fit_transform(X)
    assert Xt.shape[2] == 1
    ar = ARCoefficientTransformer(order=100)
    with pytest.raises(ValueError, match=r"must be smaller than n_timepoints - 1"):
        ar.fit_transform(X)
    ar = ARCoefficientTransformer(order=6, min_values=5)
    Xt = ar.fit_transform(X)
    assert Xt.shape[2] == 5


def test_acf():
    """Test ACF Transformer exceptions."""
    X = np.random.random((2, 1, 10))
    acf = AutocorrelationFunctionTransformer(n_lags=100)
    with pytest.raises(ValueError, match=r"must be smaller than n_timepoints - 1"):
        acf.fit_transform(X)


def test_hog1d():
    """Test HOG1D Transformer exceptions."""
    X = np.random.random((2, 1, 10))
    hog = HOG1DTransformer(n_bins=0)
    with pytest.raises(ValueError, match=r"num_bins must have the value of at least 1"):
        hog.fit_transform(X)
    hog = HOG1DTransformer(n_bins=0.5)
    with pytest.raises(TypeError, match=r"must be an 'int'"):
        hog.fit_transform(X)
    hog = HOG1DTransformer(scaling_factor="Bob")
    with pytest.raises(TypeError, match=r"scaling_factor must be a 'number'"):
        hog.fit_transform(X)
