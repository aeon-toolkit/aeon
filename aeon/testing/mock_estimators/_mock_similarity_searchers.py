"""Mock series transformers useful for testing and debugging."""

__maintainer__ = ["baraline"]
__all__ = [
    "MockSeriesSimilaritySearch",
    "MockCollectionSimilaritySearch",
]

from aeon.similarity_search.collection._base import BaseCollectionSimilaritySearch
from aeon.similarity_search.series._base import BaseSeriesSimilaritySearch


class MockSeriesSimilaritySearch(BaseSeriesSimilaritySearch):
    """Mock estimator for BaseMatrixProfile."""

    def __init__(self):
        super().__init__()

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        """top-1 motif start timestamp index in X, and distances to the match in X_."""
        return [0], [0.1]


class MockCollectionSimilaritySearch(BaseCollectionSimilaritySearch):
    """Mock estimator for BaseMatrixProfile."""

    def __init__(self):
        super().__init__()

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        """top-1 motif start timestamp index in X, and distances to the match in X_."""
        return [0, 0], [0.1]
