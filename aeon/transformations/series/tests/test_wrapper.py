"""Tests for CollectionToSeriesWrapper transformer."""

import numpy as np

from aeon.testing.mock_estimators import MockCollectionTransformer
from aeon.transformations.collection.compose import CollectionId
from aeon.transformations.series import CollectionToSeriesWrapper


def test_broadcaster_tag_inheritance():
    """Test the ability to inherit tags from the BaseCollectionTransformer.

    The broadcaster should always keep some tags related to single series
    """
    trans = MockCollectionTransformer()
    class_tags = CollectionToSeriesWrapper._tags

    bc = CollectionToSeriesWrapper(trans)

    post_constructor_tags = bc.get_tags()
    mock_tags = trans.get_tags()
    # constructor_tags should match class_tags or, if not present, tags in transformer
    for key in post_constructor_tags:
        if key in class_tags:
            assert post_constructor_tags[key] == class_tags[key]
        elif key in mock_tags:
            assert post_constructor_tags[key] == mock_tags[key]


def test_broadcaster_returns_a_series():
    """The wrapper takes a series and gives a series back, not a collection of one.

    It reshapes the 2D input to a collection of one case for the wrapped
    collection transformer. Nothing reshaped the output back, so callers were
    handed a 3D array from a series transformer.
    """
    X = np.random.RandomState(0).normal(size=(1, 20))

    bc = CollectionToSeriesWrapper(MockCollectionTransformer())
    assert bc.fit_transform(X).shape == X.shape

    bc = CollectionToSeriesWrapper(MockCollectionTransformer())
    bc.fit(X)
    assert bc.transform(X).shape == X.shape

    # a second transformer, because the first is a mock and this one is not
    bc = CollectionToSeriesWrapper(CollectionId())
    bc.fit(X)
    assert bc.transform(X).shape == X.shape
