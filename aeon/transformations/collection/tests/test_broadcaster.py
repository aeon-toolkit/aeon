"""Tests for broadcaster transformer"""

import numpy as np

from aeon.transformations.collection import BroadcastTransformer
from aeon.transformations.series import (
    DummySeriesTransformer,
    DummySeriesTransformer_no_fit,
)


def test_BroadcastTransformer_fit():
    constant = 1
    broadcaster = BroadcastTransformer(DummySeriesTransformer(constant=constant))
    X = np.zeros((2, 1, 10))
    broadcaster.fit(X)
    Xt = broadcaster.transform(X)
    assert hasattr(
        broadcaster, "series_transformers"
    ) and broadcaster.series_transformers == len(X)

    for i in range(len(X)):
        assert np.all(
            Xt[i] == constant + broadcaster.series_transformers[i].random_value_
        )
    assert Xt.shape == (2, 1, 10)


def test_BroadcastTransformer_no_fit():
    constant = 1
    broadcaster = BroadcastTransformer(DummySeriesTransformer_no_fit(constant=constant))
    X = np.zeros((2, 1, 10))
    broadcaster.fit(X)
    Xt = broadcaster.transform(X)
    assert not hasattr(broadcaster, "series_transformers")
    for i in range(len(X)):
        assert np.all(Xt[i] == constant)
    assert Xt.shape == (2, 1, 10)
