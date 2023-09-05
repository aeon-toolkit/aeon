# -*- coding: utf-8 -*-
"""BaseSimilaritySearch."""
from abc import ABC, abstractmethod

import numpy as np

from aeon.base import BaseEstimator
from aeon.distances import get_distance_function


class BaseSimiliaritySearch(BaseEstimator, ABC):
    """BaseSimilaritySearch."""

    _tags = {
        "capability:multivariate": False,
        "capability:missing_values": False,
        "X_inner_mtype": "numpyflat",
    }

    def __init__(self, distance="euclidean", n_nearest=1, normalise=False):
        self.distance = distance
        self.n_nearest = n_nearest
        self.normalise = normalise

    def fit(self, X, y=None):
        """For now, assume X is 1-D numpy.

        Do we put normalising X here? If there are multiple queries, then it makes
        sense. to be decided. Do we even want to call it X?
        """
        if not isinstance(X, np.ndarray) or X.ndim != 1:
            raise TypeError("Error, only supports 1D numpy atm.")
        # Get distance function
        self.distance_function = get_distance_function(self.distance)
        self._n_nearest = self.n_nearest
        if self.normalise:
            # normalise here
            X = X
        self._X = X
        self._fit(X, y)
        return self

    def predict(self, q):
        """Predict: find the self._n_nearest subseries in self._X to q.

        As determined by self.distance_function.

        What to return?
        """
        if not isinstance(q, np.ndarray) or q.ndim != 1:
            raise TypeError("Error, only supports 1D numpy atm.")
        if len(q >= len(self._X)):
            raise TypeError("Error, q must be shorter than X.")
        return self._predict(q)

    @abstractmethod
    def _fit(self, X, y):
        ...

    @abstractmethod
    def _predict(self, X):
        ...
