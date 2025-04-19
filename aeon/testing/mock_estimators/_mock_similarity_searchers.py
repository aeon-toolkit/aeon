"""Mock series transformers useful for testing and debugging."""

__maintainer__ = ["baraline"]
__all__ = [
    "MockSeriesMotifSearch",
    "MockSeriesNeighborsSearch",
    "MockCollectionMotifsSearch",
    "MockCollectionNeighborsSearch",
]

from aeon.similarity_search.collection._base import (
    BaseCollectionMotifs,
    BaseCollectionNeighbors,
)
from aeon.similarity_search.series._base import BaseSeriesMotifs, BaseSeriesNeighbors


class MockSeriesMotifSearch(BaseSeriesMotifs):
    """Mock estimator for BaseMatrixProfile."""

    def __init__(self):
        super().__init__()

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        """top-1 motif start timestamp index in X, and distances to the match in X_."""
        return [0], [0.1]


class MockSeriesNeighborsSearch(BaseSeriesNeighbors):
    """Mock estimator for BaseMatrixProfile."""

    def __init__(self):
        super().__init__()

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        """top-1 neighbor start timestamp index in X_, and distances to the query."""
        return [0], [0.1]


class MockCollectionMotifsSearch(BaseCollectionMotifs):
    """Mock estimator for BaseMatrixProfile."""

    def __init__(self):
        super().__init__()

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        """top-1 motif start timestamp index in X, and distances to the match in X_."""
        return [0, 0], [0.1]


class MockCollectionNeighborsSearch(BaseCollectionNeighbors):
    """Mock estimator for BaseMatrixProfile."""

    def __init__(self):
        super().__init__()

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        """top-1 neighbor sample index in X_, and distances to the each query."""
        return [0 for _ in range(len(X))], [0.1 for _ in range(len(X))]
