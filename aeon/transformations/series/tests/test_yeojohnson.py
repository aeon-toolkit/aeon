"""Tests for YeoJohnsonTransformer."""

__maintainer__ = []
__all__ = []

import numpy as np
import pytest
from scipy.stats import yeojohnson

from aeon.datasets import load_airline
from aeon.transformations.series._yeojohnson import YeoJohnsonTransformer


def test_yeojohnson_against_scipy():
    """Test YeoJohnsonTransformer against scipy implementation."""
    y = load_airline()

    t = YeoJohnsonTransformer()
    actual = t.fit_transform(y)

    excepted, expected_lambda = yeojohnson(y)
    np.testing.assert_almost_equal(actual, excepted, decimal=12)
    assert t._lambda == expected_lambda


@pytest.mark.parametrize("lmbda", [0.0, 0.5, 2.0])
def test_yeojohnson_supplied_lambda_against_scipy(lmbda):
    """Supplied lambdas match SciPy for positive and negative observations."""
    y = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

    actual = YeoJohnsonTransformer(lmbda=lmbda).fit_transform(y)
    expected = yeojohnson(y, lmbda=lmbda)

    np.testing.assert_allclose(actual.squeeze(), expected)
