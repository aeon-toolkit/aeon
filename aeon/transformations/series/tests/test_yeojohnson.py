"""Tests for YeoJohnsonTransformer."""

__maintainer__ = []
__all__ = []

import numpy as np
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


def test_yeojohnson_with_supplied_lambda():
    """Test a supplied lmbda is used directly instead of being estimated."""
    y = load_airline()
    t = YeoJohnsonTransformer(lmbda=0.5)
    t.fit(y)
    assert t._lambda == 0.5


def test_yeojohnson_lambda_zero_uses_log():
    """Test lambda=0 uses the log branch for non-negative values."""
    y = np.array([[1.0, 2.0, 3.0]])
    t = YeoJohnsonTransformer(lmbda=0.0)
    t.fit(y)
    Xt = t.transform(y)
    np.testing.assert_allclose(Xt.squeeze(), np.log(y.squeeze() + 1))


def test_yeojohnson_lambda_two_uses_log_for_negative_values():
    """Test lambda=2 uses the log branch for negative values."""
    y = np.array([[-1.0, -2.0, -3.0]])
    t = YeoJohnsonTransformer(lmbda=2.0)
    t.fit(y)
    Xt = t.transform(y)
    np.testing.assert_allclose(Xt.squeeze(), -np.log(-y.squeeze() + 1))
