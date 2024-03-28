"""Tests for SeriesToCollectionWrapper transformer."""

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
    """Test the ability to inherit tags from the BaseSeriesTransformer."""
    wrapper = SeriesToCollectionWrapper(DummySeriesTransformerNoFit())
    wrapper_tags = wrapper.get_tags()
    dummy_tags = DummySeriesTransformerNoFit().get_tags()
    for key in wrapper._tags_to_inherit:
        assert wrapper_tags[key] == dummy_tags[key]


@pytest.mark.parametrize(
    "data_gen",
    [
        make_example_3d_numpy,
    ],
)
def test_SeriesToCollectionWrapper(data_gen):
    """Test the wrapper fit, transform and inverse transform method."""
    X, y = data_gen()
    constant = 1
    wrapper = SeriesToCollectionWrapper(DummySeriesTransformer(constant=constant))
    wrapper.fit(X, y)
    Xt = wrapper.transform(X)
    Xit = wrapper.inverse_transform(Xt)
    assert hasattr(wrapper, "series_transformers") and len(
        wrapper.series_transformers
    ) == len(X)

    for i in range(len(X)):
        for j in range(wrapper.series_transformers[i].n_features_):
            assert_array_almost_equal(
                Xt[i, j],
                X[i, j] + constant + wrapper.series_transformers[i].random_values_[j],
            )

    assert_array_almost_equal(Xit, X)


@pytest.mark.parametrize(
    "data_gen",
    [make_example_3d_numpy],
)
def test_SeriesToCollectionWrapper_no_fit(data_gen):
    """Test the wrapper for transformers with fit_empty."""
    X, y = data_gen()
    constant = 1
    wrapper = SeriesToCollectionWrapper(DummySeriesTransformerNoFit(constant=constant))
    wrapper.fit(X, y)
    Xt = wrapper.transform(X)
    Xit = wrapper.inverse_transform(Xt)
    assert not hasattr(wrapper, "series_transformers")
    assert_array_almost_equal(Xt, X + constant)
    assert_array_almost_equal(Xit, X)
