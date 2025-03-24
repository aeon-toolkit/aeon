"""Mock collection transformers useful for testing and debugging."""

__maintainer__ = []
__all__ = [
    "MockCollectionTransformer",
]

from aeon.transformations.collection import BaseCollectionTransformer


class MockCollectionTransformer(BaseCollectionTransformer):
    """BasecollectionTransformer for testing tags."""

    _tags = {
        "capability:multivariate": True,
    }

    def __init__(self) -> None:
        super().__init__()

    def _fit(self, X, y=None):
        """Mock fit."""
        return self

    def _transform(self, X, y=None):
        """Mock transform."""
        return X
