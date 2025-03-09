"""Density clustering for time series data."""

__maintainer__ = []
__all__ = ["DensityPeakClusterer"]

import numpy as np  # noqa

from aeon.distances import get_distance_function, pairwise_distance  # noqa


class DensityPeakClusterer:
    """Density Peak Clusterer.

    Clusters time series data using a density-based approach that estimates local
    densities and identifies peaks as cluster centers.
    """

    def __init__(
        self,
        rho=None,
        cutoff_distance=None,
        distance="dtw",
        n_jobs=1,
    ):
        self.rho = rho
        self.cutoff_distance = cutoff_distance
        self.distance = distance
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X : array-like
            Time series data to cluster.
        y : array-like, optional
            Labels for the data (unused in clustering).

        Returns
        -------
        self : object
            The fitted clusterer.
        """
        self._fit(X)
        self.is_fitted = True
        return self
