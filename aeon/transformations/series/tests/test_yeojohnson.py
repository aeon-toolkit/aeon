"""Tests for YeoJohnsonTransformer."""

__maintainer__ = []
__all__ = []

import numpy as np
from scipy.stats import yeojohnson

from aeon.datasets import load_airline
from aeon.transformations.series.yeojohnson import YeoJohnsonTransformer


def test_yeojohnson_against_scipy():
    y = load_airline()

    t = YeoJohnsonTransformer()
    actual = t.fit_transform(y)

    excepted, expected_lambda = yeojohnson(y.values)
    np.testing.assert_almost_equal(actual, excepted, decimal=12)
    assert t.lambda_ == expected_lambda
