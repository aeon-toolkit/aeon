"""Tests for SeriesToCollectionBroadcaster transformer."""

from aeon.testing.mock_estimators import MockCollectionTransformer
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
