"""Legacy data generators."""

__all__ = [
    "make_example_long_table",
    "_make_collection",
    "_make_collection_X",
    "_make_classification_y",
]

from aeon.testing.data_generation._legacy._collection import (
    _make_classification_y,
    _make_collection,
    _make_collection_X,
    make_example_long_table,
)
