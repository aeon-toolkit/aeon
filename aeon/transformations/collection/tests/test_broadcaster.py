"""Tests for broadcaster transformer"""

import numpy as np
import pytest

from aeon.transformations.collection import BroadcastTransformer
from aeon.transformations.series import (
    DummySeriesTransformer,
    DummySeriesTransformer_no_fit,
)

INPUT_SHAPES = [(2, 1, 10), (2, 2, 10)]


def test_BroadcastTransformer_tag_inheritance():
    broadcaster = BroadcastTransformer(DummySeriesTransformer_no_fit())
    broadcaster_tags = broadcaster.get_tags()
    dummy_tags = DummySeriesTransformer_no_fit().get_tags()
    for key in broadcaster._tags_to_inherit:
        assert broadcaster_tags[key] == dummy_tags[key]


@pytest.mark.parametrize("input_shape", INPUT_SHAPES)
def test_BroadcastTransformer_fit(input_shape):
    constant = 1
    broadcaster = BroadcastTransformer(DummySeriesTransformer(constant=constant))
    X = np.zeros(input_shape)
    broadcaster.fit(X)
    Xt = broadcaster.transform(X)
    Xit = broadcaster.inverse_transform(Xt)
    assert hasattr(broadcaster, "series_transformers") and len(
        broadcaster.series_transformers
    ) == len(X)
    for i in range(len(X)):
        assert np.all(
            Xt[i] == constant + broadcaster.series_transformers[i].random_value_
        )
    np.testing.assert_array_equal(Xit, X)
    assert Xt.shape == Xit.shape == X.shape


@pytest.mark.parametrize("input_shape", INPUT_SHAPES)
def test_BroadcastTransformer_no_fit(input_shape):
    constant = 1
    broadcaster = BroadcastTransformer(DummySeriesTransformer_no_fit(constant=constant))
    X = np.zeros(input_shape)
    broadcaster.fit(X)
    Xt = broadcaster.transform(X)
    Xit = broadcaster.inverse_transform(Xt)
    assert not hasattr(broadcaster, "series_transformers")
    for i in range(len(X)):
        assert np.all(Xt[i] == constant)
    np.testing.assert_array_equal(Xit, X)
    assert Xt.shape == Xit.shape == X.shape
