"""Mock series transformers useful for testing and debugging."""

__maintainer__ = []
__all__ = [
    "MockSubsequenceSearch",
    "MockMatrixProfile",
]

import numpy as np

from aeon.similarity_search.subsequence_search.base import (
    BaseMatrixProfile,
    BaseSubsequenceSearch,
)


class MockMatrixProfile(BaseMatrixProfile):
    def __init__(self, length=3):
        super().__init__(length=length)

    def compute_matrix_profile(
        self,
        k,
        threshold,
        exclusion_size,
        inverse_distance,
        allow_neighboring_matches,
        X=None,
        X_index=None,
    ):
        """Compute matrix profiles between X_ and X or between all series in X_."""
        return np.zeros((X.shape[1] - self.length + 1, k)), np.zeros(
            (X.shape[1] - self.length + 1, k, 2)
        )

    def compute_distance_profile(self, X):
        """Compute distrance profiles between X_ and X (a series of size length)."""
        return np.zeros(X.shape[1] - self.length + 1)


class MockSubsequenceSearch(BaseSubsequenceSearch):
    """MockSeriesTransformer to set tags."""

    def __init__(self, length=3):
        super().__init__(length=length)

    def _fit(self, X, y=None):
        return self

    def _find_motifs(
        self,
        X,
        k=1,
        threshold=np.inf,
        inverse_distance=False,
        X_index=None,
        allow_neighboring_matches=False,
        exclusion_factor=2.0,
    ):
        return [[0, 0]], self.X_[0][0:1]  # TODO: update after logic is implemented

    def _find_neighbors(
        self,
        X,
        k=1,
        threshold=np.inf,
        inverse_distance=False,
        X_index=None,
        allow_neighboring_matches=False,
        exclusion_factor=2.0,
    ):
        return [[0, 0]], self.X_[0][0:1]
