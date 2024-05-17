"""Compositions for collection transforms."""

__all__ = [
    "CollectionTransformerPipeline",
    "CollectionId",
]

from aeon.transformations.collection.compose._identity import CollectionId
from aeon.transformations.collection.compose._pipeline import (
    CollectionTransformerPipeline,
)
