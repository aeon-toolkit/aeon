"""Tests for broadcaster transformer"""

__maintainer__ = ["baraline"]

import pytest
from numpy.testing import assert_array_almost_equal

from aeon.testing.utils.data_gen import make_example_3d_numpy
from aeon.transformations.collection._series_wrapper import SeriesToCollectionWrapper
from aeon.transformations.series import (
    DummySeriesTransformer,
    DummySeriesTransformerNoFit,
)


def test_SeriesToCollectionWrapper_tag_inheritance():
    broadcaster = SeriesToCollectionWrapper(DummySeriesTransformerNoFit())
    broadcaster_tags = broadcaster.get_tags()
    dummy_tags = DummySeriesTransformerNoFit().get_tags()
    for key in broadcaster._tags_to_inherit:
        assert broadcaster_tags[key] == dummy_tags[key]


@pytest.mark.parametrize(
    "data_gen",
    [
        make_example_3d_numpy,
    ],
)
def test_SeriesToCollectionWrapper(data_gen):
    X, y = data_gen()
    constant = 1
    broadcaster = SeriesToCollectionWrapper(DummySeriesTransformer(constant=constant))
    broadcaster.fit(X, y)
    Xt = broadcaster.transform(X)
    Xit = broadcaster.inverse_transform(Xt)
    assert hasattr(broadcaster, "series_transformers") and len(
        broadcaster.series_transformers
    ) == len(X)

    for i in range(len(X)):
        for j in range(broadcaster.series_transformers[i].n_features_):
            assert_array_almost_equal(
                Xt[i, j],
                X[i, j]
                + constant
                + broadcaster.series_transformers[i].random_values_[j],
            )

    assert_array_almost_equal(Xit, X)


@pytest.mark.parametrize(
    "data_gen",
    [make_example_3d_numpy],
)
def test_SeriesToCollectionWrapper_no_fit(data_gen):
    X, y = data_gen()
    constant = 1
    broadcaster = SeriesToCollectionWrapper(
        DummySeriesTransformerNoFit(constant=constant)
    )
    broadcaster.fit(X, y)
    Xt = broadcaster.transform(X)
    Xit = broadcaster.inverse_transform(Xt)
    assert not hasattr(broadcaster, "series_transformers")
    assert_array_almost_equal(Xt, X + constant)
    assert_array_almost_equal(Xit, X)
