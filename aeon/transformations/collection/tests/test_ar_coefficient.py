"""Test for incorrect input for ARCoefficientTransformer transformer."""

import numpy as np
import pytest

from aeon.transformations.collection import ARCoefficientTransformer
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
