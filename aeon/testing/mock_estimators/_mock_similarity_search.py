"""Mock similarity search useful for testing and debugging.

Used in tests for the query search base class.
"""

from aeon.similarity_search.base import BaseSimilaritySearch


class MocksimilaritySearch(BaseSimilaritySearch):
    """Mock similarity search for testing base class predict."""

    def _fit(self, X, y=None):
        """_fit dummy."""
        self.X_ = X
        return self

    def predict(self, X):
        """Predict dummy."""
        return [(0, 0)]
