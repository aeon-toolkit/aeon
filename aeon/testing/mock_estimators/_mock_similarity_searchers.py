"""Mock series transformers useful for testing and debugging."""

__maintainer__ = ["baraline"]
__all__ = [
    "MockSeriesSimilaritySearch",
]

from aeon.similarity_search.series._base import BaseSeriesSimilaritySearch


class MockSeriesSimilaritySearch(BaseSeriesSimilaritySearch):
    """Mock estimator for BaseMatrixProfile."""

    def __init__(
        self,
        length=3,
        normalize=False,
        n_jobs=1,
    ):
        self.length = length
        self.normalize = normalize
        super().__init__(n_jobs=n_jobs)

    def _fit(self, X, y=None):
        return self

    def predict(self, X):
        """Compute matrix profiles between X_ and X or between all series in X_."""
        X = self._pre_predict(X)
        return [0], [0.1]
