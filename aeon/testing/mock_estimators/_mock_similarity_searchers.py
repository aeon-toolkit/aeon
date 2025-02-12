"""Mock series transformers useful for testing and debugging."""

__maintainer__ = ["baraline"]
__all__ = ["MockSeriesSimilaritySearch", "MockCollectionSimilaritySearch"]

from aeon.similarity_search.collection._base import BaseCollectionSimilaritySearch
from aeon.similarity_search.series._base import BaseSeriesSimilaritySearch


class MockSeriesSimilaritySearch(BaseSeriesSimilaritySearch):
    """Mock estimator for BaseMatrixProfile."""

    def __init__(self):
        super().__init__()

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        """Compute matrix profiles between X_ and X or between all series in X_."""
        return [0], [0.1]


class MockCollectionSimilaritySearch(BaseCollectionSimilaritySearch):
    """Mock estimator for BaseMatrixProfile."""

    def __init__(self):
        super().__init__()

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        """Compute matrix profiles between X_ and X or between all series in X_."""
        return [0 for _ in range(len(X))], [0.1 for _ in range(len(X))]
