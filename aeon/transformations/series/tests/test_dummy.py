"""Tests for dummy series transformer."""

__maintainer__ = ["baraline"]

import numpy as np
import pytest

from aeon.transformations.series import (
    DummySeriesTransformer,
    DummySeriesTransformerNoFit,
)

INPUT_SHAPES = [(2, 5), (1, 5)]


@pytest.mark.parametrize("input_shape", INPUT_SHAPES)
def test_DummySeriesTransformer(input_shape):
    """
    Test dummy transformer fit, transform and inverse_transform methods.
    """
    constant = 1
    X = np.zeros(input_shape)
    transformer = DummySeriesTransformer(constant=constant).fit(X)
    Xt = transformer.transform(X)
    assert Xt.shape == input_shape
    for i in range(X.shape[0]):
        assert np.all(Xt[i] == constant + transformer.random_values_[i])
    Xit = transformer.inverse_transform(Xt)
    assert np.all(Xit == X) and Xit.shape == input_shape


def test_DummySeriesTransformer_1D_convertion():
    """
    Test that dummy transformer correctly handle 1D data series.
    """
    constant = 1
    X = np.zeros(5)
    transformer = DummySeriesTransformer(constant=constant).fit(X)
    Xt = transformer.transform(X)
    assert np.all(Xt == constant + transformer.random_values_) and Xt.shape == (1, 5)
    Xit = transformer.inverse_transform(Xt)
    assert np.all(Xit == X) and Xit.shape == (1, 5)


@pytest.mark.parametrize("input_shape", INPUT_SHAPES)
def test_DummySeriesTransformerNoFit(input_shape):
    """
    Test dummy transform and inverse_transform methods with fit_empty.
    """
    constant = 1
    X = np.zeros(input_shape)
    transformer = DummySeriesTransformerNoFit(constant=constant).fit(X)
    Xt = transformer.transform(X)
    assert np.all(Xt == constant) and Xt.shape == input_shape
    Xit = transformer.inverse_transform(Xt)
    assert np.all(Xit == X) and Xit.shape == input_shape
