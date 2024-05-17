"""Compositions for collection transforms."""

__all__ = [
    "CollectionTransformerPipeline",
    "Id",
]

from aeon.transformations.collection.compose._identity import Id
from aeon.transformations.collection.compose._pipeline import (
    CollectionTransformerPipeline,
)
