"""Tests for dummy series transformer."""

import numpy as np
import pytest

from aeon.transformations.series import (
    DummySeriesTransformer,
    DummySeriesTransformer_no_fit,
)

INPUT_SHAPES = [(2, 5), (1, 5), (5)]


@pytest.mark.parametrize("input_shape", INPUT_SHAPES)
def test_DummySeriesTransformer(input_shape):
    constant = 1
    X = np.zeros(input_shape)
    transformer = DummySeriesTransformer(constant=constant).fit(X)
    Xt = transformer.transform(X)
    assert (
        np.all(Xt == constant + transformer.random_value_) and Xt.shape == input_shape
    )
    Xit = transformer.inverse_transform(Xt)
    assert np.all(Xit == X) and Xit.shape == input_shape


@pytest.mark.parametrize("input_shape", INPUT_SHAPES)
def test_DummySeriesTransformer_no_fit(input_shape):
    constant = 1
    X = np.zeros(input_shape)
    transformer = DummySeriesTransformer_no_fit(constant=constant).fit(X)
    Xt = transformer.transform(X)
    assert np.all(Xt == constant) and Xt.shape == input_shape
    Xit = transformer.inverse_transform(Xt)
    assert np.all(Xit == X) and Xit.shape == input_shape
