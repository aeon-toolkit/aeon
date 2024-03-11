"""Tests for dummy series transformer."""

import numpy as np

from aeon.transformations.series import DummySeriesTransformer


def test_DummySeriesTransformer():
    constant = 1
    X = np.zeros((2, 5))
    transformer = DummySeriesTransformer(constant=constant).fit(X)
    Xt = transformer.transform(X)
    assert np.all(Xt == constant + transformer.random_value_) and Xt.shape == (2, 5)

    X = np.zeros((1, 5))
    transformer = DummySeriesTransformer(constant=constant).fit(X)
    Xt = transformer.transform(X)
    assert np.all(Xt == constant + transformer.random_value_) and Xt.shape == (1, 5)

    X = np.zeros(5)
    transformer = DummySeriesTransformer(constant=constant).fit(X)
    Xt = transformer.transform(X)
    assert np.all(Xt == constant + transformer.random_value_) and Xt.shape == (1, 5)
