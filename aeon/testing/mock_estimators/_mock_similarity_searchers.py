"""Mock similarity search estimators useful for testing and debugging."""

__maintainer__ = ["baraline"]
__all__ = [
    "MockSubsequenceSearch",
    "MockDistanceProfileSearch",
    "MockWholeSeriesSearch",
]

import numpy as np

from aeon.similarity_search.subsequence._base import (
    BaseDistanceProfileSearch,
    BaseSubsequenceSearch,
)
from aeon.similarity_search.whole_series._base import BaseWholeSeriesSearch


class MockSubsequenceSearch(BaseSubsequenceSearch):
    """Mock estimator for BaseSubsequenceSearch."""

    _tags = {
        "capability:multivariate": True,
    }

    def __init__(self, length=10):
        super().__init__(length=length)

    def _fit(self, X, y=None):
        return self

    def _predict(self, X, k=1, **kwargs):
        """Return dummy predictions."""
        n_matches = min(k, self.n_cases_)
        indexes = np.zeros((n_matches, 2), dtype=np.int64)
        distances = np.zeros(n_matches)
        return indexes, distances


class MockDistanceProfileSearch(BaseDistanceProfileSearch):
    """Mock estimator for BaseDistanceProfileSearch."""

    _tags = {
        "capability:multivariate": True,
    }

    def __init__(self, length=10):
        super().__init__(length=length)

    def _fit(self, X, y=None):
        return self

    def compute_distance_profile(self, X):
        """Return squared Euclidean distance profile."""
        n_candidates = self.n_timepoints_ - self.length + 1
        dist_profiles = np.zeros((self.n_cases_, n_candidates))
        for i in range(self.n_cases_):
            for j in range(n_candidates):
                dist_profiles[i, j] = np.sum(
                    (self.X_[i, :, j : j + self.length] - X) ** 2
                )
        return dist_profiles


class MockWholeSeriesSearch(BaseWholeSeriesSearch):
    """Mock estimator for BaseWholeSeriesSearch."""

    def __init__(self):
        super().__init__()

    def _fit(self, X, y=None):
        return self

    def _predict(self, X, k=1, **kwargs):
        """Return dummy predictions."""
        n_matches = min(k, self.n_cases_)
        indexes = np.arange(n_matches, dtype=np.int64)
        distances = np.zeros(n_matches)
        return indexes, distances

    def compute_distance_profile(self, X):
        """Return dummy distance profile."""
        return np.zeros(self.n_cases_)
