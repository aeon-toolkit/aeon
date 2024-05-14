"""Tests for dummy series transformer."""

__maintainer__ = ["baraline"]

import numpy as np
import pytest

from aeon.testing.mock_estimators._mock_series_transformers import (
    MockSeriesTransformerNoFit,
    MockUnivariateSeriesTransformer,
)


def test_mock_uni_mock_transformer():
    """Test dummy transformer fit, transform and inverse_transform methods."""
    constant = 1
    X = np.zeros((1, 10))
    transformer = MockUnivariateSeriesTransformer(constant=constant).fit(X)
    Xt = transformer.transform(X)
    assert Xt.shape[0] == 10
    for i in range(X.shape[0]):
        assert np.all(Xt[i] == constant + transformer.random_values_[i])
    Xit = transformer.inverse_transform(Xt)
    assert np.all(Xit == X)


INPUT_SHAPES = [(2, 5), (1, 5)]


@pytest.mark.parametrize("input_shape", INPUT_SHAPES)
def test_mock_series_no_fit(input_shape):
    """Test dummy transform and inverse_transform methods with fit_empty."""
    constant = 1
    X = np.zeros(input_shape)
    transformer = MockSeriesTransformerNoFit(constant=constant).fit(X)
    Xt = transformer.transform(X)
    assert np.all(Xt == constant) and Xt.shape == input_shape
    Xit = transformer.inverse_transform(Xt)
    assert np.all(Xit == X) and Xit.shape == input_shape
