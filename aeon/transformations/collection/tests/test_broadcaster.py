"""Tests for SeriesToCollectionBroadcaster transformer."""

__maintainer__ = ["baraline"]

import pytest
from numpy.testing import assert_array_almost_equal

from aeon.testing.mock_estimators._mock_series_transformers import (
    MockSeriesTransformer,
    MockSeriesTransformerNoFit,
)
from aeon.testing.utils.data_gen import (
    make_example_3d_numpy,
    make_example_unequal_length,
)
from aeon.transformations.collection._broadcaster import SeriesToCollectionBroadcaster


def test_broadcaster_tag_inheritance():
    """Test the ability to inherit tags from the BaseSeriesTransformer.

    The broadcaster should always keep the tags related to the collection, namely
        "input_data_type": "Collection",
        "output_data_type": "Collection",
        "capability:unequal_length": True,
        "X_inner_type": ["numpy3D", "np-list"],
    """
    trans = MockSeriesTransformerNoFit()
    class_tags = SeriesToCollectionBroadcaster._tags

    bc = SeriesToCollectionBroadcaster(trans)

    post_constructor_tags = bc.get_tags()
    mock_tags = trans.get_tags()
    # constructor_tags should match class_tags or, if not present, tags in transformer
    for key in post_constructor_tags:
        if key in class_tags:
            assert post_constructor_tags[key] == class_tags[key]
        elif key in mock_tags:
            assert post_constructor_tags[key] == mock_tags[key]


@pytest.mark.parametrize(
    "data_gen",
    [
        make_example_3d_numpy,
        make_example_unequal_length,
    ],
)
def test_broadcaster_methods(data_gen):
    """Test the broadcaster fit, transform and inverse transform method."""
    X, y = data_gen()
    constant = 1
    broadcaster = SeriesToCollectionBroadcaster(
        MockSeriesTransformer(constant=constant)
    )
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
def test_broadcaster_no_fit(data_gen):
    """Test the wrapper for transformers with fit_empty."""
    X, y = data_gen()
    constant = 1
    wrapper = SeriesToCollectionBroadcaster(
        MockSeriesTransformerNoFit(constant=constant)
    )
    wrapper.fit(X, y)
    Xt = wrapper.transform(X)
    Xit = wrapper.inverse_transform(Xt)
    assert not hasattr(wrapper, "series_transformers")
    assert_array_almost_equal(Xt, X + constant)
    assert_array_almost_equal(Xit, X)
