"""Mock similarity searchers useful for testing and debugging."""

__maintainer__ = ["baraline"]
__all__ = [
    "MockSimilaritySearch",
]

from aeon.similarity_search.base import BaseSimilaritySearch


class MockSimilaritySearch(BaseSimilaritySearch):
    """Mock similarity search for testing base class predict."""

    def _fit(self, X, y=None):
        """_fit dummy."""
        self.X_ = X
        return self

    def predict(self, X):
        """Predict dummy."""
        return [(0, 0)]
