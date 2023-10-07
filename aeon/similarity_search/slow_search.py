"""First similarity search."""
import numpy as np

from aeon.similarity_search.base import BaseSimiliaritySearch


class SlowSearch(BaseSimiliaritySearch):
    """First similarity search."""

    def __init__(self):
        super(SlowSearch, self).__init__()

    def _fit(self, X, y):
        return self

    def _predict(self, q):
        index = 0
        min_d = np.Inf
        l2 = len(q)
        for i in range(0, len(self._X) - l2 - 1):
            d = self.distance_function(q, self._X[i : i + l2 - 1])
            if d < min_d:
                index = i
                min_d = d
        return [index]
