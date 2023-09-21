# -*- coding: utf-8 -*-
"""BaseSimilaritySearch."""

__author__ = ["baraline"]

from abc import ABC, abstractmethod

import numpy as np

from aeon.base import BaseEstimator
from aeon.similarity_search.distance_profiles import (
    naive_euclidean_profile,
    normalized_naive_euclidean_profile,
)
from aeon.utils.numba.general import sliding_mean_std_one_series


class BaseSimiliaritySearch(BaseEstimator, ABC):
    """BaseSimilaritySearch.

    Attributes
    ----------
    distance : str, optional
        Name of the distance function to use. The default is "euclidean".
    normalize : bool, optional
        Wheter the distance function should be z-normalized
    store_distance_profile : bool, optional
        Wheter to store the computed distance profile in the attribute
        "_distance_profile" after calling the predict method.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:missing_values": False,
    }

    def __init__(
        self, distance="euclidean", normalize=False, store_distance_profile=False
    ):
        self.distance = distance
        self.normalize = normalize
        self.store_distance_profile = store_distance_profile

    def _get_distance_profile_function(self):
        dist_profile = DISTANCE_PROFILE_DICT.get(self.distance)
        if dist_profile is None:
            raise ValueError(f"Unknown distrance profile function {dist_profile}")
        return dist_profile[self.normalize]

    def _store_mean_std_from_inputs(self, Q_length):
        n_samples, n_channels, X_length = self._X.shape
        search_space_size = n_samples * (X_length - Q_length + 1)

        means = np.zeros((n_samples, n_channels, search_space_size))
        stds = np.zeros((n_samples, n_channels, search_space_size))

        for i in range(n_samples):
            _mean, _std = sliding_mean_std_one_series(self._X[i], Q_length, 1)
            stds[i] = _std
            means[i] = _mean

        self._X_means = means
        self._X_stds = stds

    def fit(self, X, y=None):
        """
        Fit method: store the input data and get the distance profile function.

        Parameters
        ----------
        X : array, shape (n_samples, n_channels, n_timestamps)
            Input array to used as database for the similarity search
        y : TYPE, optional
            Not used.

        Raises
        ------
        TypeError
            If the input X array is not 3D raise an error.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # For now force (n_samples, n_channels, n_timestamps), we could convert 2D
        #  (n_channels, n_timestamps) to 3D with a warning
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            raise TypeError(
                "Error, only supports 3D numpy of shape"
                "(n_samples, n_channels, n_timestamps)."
            )

        # Get distance function
        self.distance_profile_function = self._get_distance_profile_function()
        self._normalized = self.distance_profile_function.__name__.startswith(
            "normalized"
        )

        self._X = X
        self._fit(X, y)
        return self

    def predict(self, Q):
        """
        Predict method: Check the shape of Q and call _predict to perform the search.

        If the distance profile function is normalized, it stores the mean and stds
        from Q and _X.

        Parameters
        ----------
        Q :  array, shape (n_channels, q_length)
            Input query used for similarity search.

        Raises
        ------
        TypeError
            If the input Q array is not 2D raise an error.

        Returns
        -------
        array
            An array containing the indexes of the matches between Q and _X.
            The decision of wheter a candidate of size q_length from _X is matched with
            Q depends on the subclasses that implent the _predict method
            (e.g. top-k, threshold, ...).

        """
        if not isinstance(Q, np.ndarray) or Q.ndim != 2:
            raise TypeError(
                "Error, only supports 2D numpy atm. If Q is univariate"
                " do Q.reshape(1,-1)."
            )

        if Q.shape[-1] >= self._X.shape[-1]:
            raise TypeError("Error, Q must be shorter than X.")

        if self._normalized:
            self._Q_mean = np.mean(Q, axis=-1)
            self._Q_std = np.std(Q, axis=-1)
            self._store_mean_std_from_inputs(Q.shape[-1])

        return self._predict(Q)

    @abstractmethod
    def _fit(self, X, y):
        ...

    @abstractmethod
    def _predict(self, X):
        ...


"""
Dictionary structure :
    1st lvl key : distance function used
        2nd lvl key : boolean indicating wheter distance is normalized
"""
DISTANCE_PROFILE_DICT = {
    "euclidean": {
        True: normalized_naive_euclidean_profile,
        False: naive_euclidean_profile,
    }
}
