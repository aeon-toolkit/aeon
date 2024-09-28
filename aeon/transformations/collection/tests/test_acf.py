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
