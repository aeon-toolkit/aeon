"""Base class for similarity search."""

__author__ = ["baraline"]

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import final

import numpy as np

from aeon.base import BaseEstimator
from aeon.similarity_search.distance_profiles import (
    naive_euclidean_profile,
    normalized_naive_euclidean_profile,
)
from aeon.utils.numba.general import sliding_mean_std_one_series


class BaseSimiliaritySearch(BaseEstimator, ABC):
    """BaseSimilaritySearch.

    Parameters
    ----------
    distance : str, default ="euclidean"
        Name of the distance function to use.
    normalize : bool, default = False
        Whether the distance function should be z-normalized.
    store_distance_profile : bool, default = False.
        Whether to store the computed distance profile in the attribute
        "_distance_profile" after calling the predict method.

    Attributes
    ----------
    _X : array, shape (n_instances, n_channels, n_timestamps)
        The input time series stored during the fit method.
    distance_profile_function : function
        The function used to compute the distance profile affected
        during the fit method based on the distance and normalize
        parameters.
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
        super(BaseSimiliaritySearch, self).__init__()

    def _get_distance_profile_function(self):
        dist_profile = DISTANCE_PROFILE_DICT.get(self.distance)
        if dist_profile is None:
            raise ValueError(
                f"Unknown or unsupported distance profile function {dist_profile}"
            )
        return dist_profile[self.normalize]

    def _store_mean_std_from_inputs(self, q_length):
        n_instances, n_channels, X_length = self._X.shape
        search_space_size = X_length - q_length + 1

        means = np.zeros((n_instances, n_channels, search_space_size))
        stds = np.zeros((n_instances, n_channels, search_space_size))

        for i in range(n_instances):
            _mean, _std = sliding_mean_std_one_series(self._X[i], q_length, 1)
            stds[i] = _std
            means[i] = _mean

        self._X_means = means
        self._X_stds = stds

    @final
    def fit(self, X, y=None):
        """
        Fit method: store the input data and get the distance profile function.

        Parameters
        ----------
        X : array, shape (n_instances, n_channels, n_timestamps)
            Input array to used as database for the similarity search
        y : optional
            Not used.

        Raises
        ------
        TypeError
            If the input X array is not 3D raise an error.

        Returns
        -------
        self

        """
        # For now force (n_instances, n_channels, n_timestamps), we could convert 2D
        #  (n_channels, n_timestamps) to 3D with a warning
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            raise TypeError(
                "Error, only supports 3D numpy of shape"
                "(n_instances, n_channels, n_timestamps)."
            )

        # Get distance function
        self.distance_profile_function = self._get_distance_profile_function()

        self._X = X.astype(float)
        self._fit(X, y)
        return self

    @final
    def predict(self, q, q_index=None, exclusion_factor=2.0):
        """
        Predict method: Check the shape of q and call _predict to perform the search.

        If the distance profile function is normalized, it stores the mean and stds
        from q and _X.

        Parameters
        ----------
        q :  array, shape (n_channels, q_length)
            Input query used for similarity search.
        q_index : Iterable, default=None
            An Interable (tuple, list, array) used to specify the index of Q if it is
            extracted from the input data X given during the fit method.
            Given the tuple (id_sample, id_timestamp), the similarity search will define
            an exclusion zone around the q_index in order to avoid matching q with
            itself. If None, it is considered that the query is not extracted from X.
        exclusion_factor : float, default=2.
            The factor to apply to the query length to define the exclusion zone. The
            exclusion zone is define from id_timestamp - q_length//exclusion_factor to
            id_timestamp + q_length//exclusion_factor

        Raises
        ------
        TypeError
            If the input q array is not 2D raise an error.
        ValueError
            If the length of the query is greater

        Returns
        -------
        array
            An array containing the indexes of the matches between q and _X.
            The decision of wheter a candidate of size q_length from _X is matched with
            Q depends on the subclasses that implent the _predict method
            (e.g. top-k, threshold, ...).

        """
        if not isinstance(q, np.ndarray) or q.ndim != 2:
            raise TypeError(
                "Error, only supports 2D numpy atm. If q is univariate"
                " do q.reshape(1,-1)."
            )

        q_dim, q_length = q.shape
        if q_length >= self._X.shape[-1]:
            raise ValueError(
                "The length of the query should be inferior or equal to the length of"
                "data (X) provided during fit, but got {} for q and {} for X".format(
                    q_length, self._X.shape[-1]
                )
            )

        if q_dim != self._X.shape[1]:
            raise ValueError(
                "The number of feature should be the same for the query q and the data"
                "(X) provided during fit, but got {} for q and {} for X".format(
                    q_dim, self._X.shape[1]
                )
            )

        n_instances, _, n_timestamps = self._X.shape
        mask = np.ones((n_instances, q_dim, n_timestamps), dtype=bool)

        if q_index is not None:
            if isinstance(q_index, Iterable):
                if len(q_index) != 2:
                    raise ValueError(
                        "The q_index should contain an interable of size 2 such as"
                        "(id_sample, id_timestamp), but got an iterable of"
                        "size {}".format(len(q_index))
                    )
            else:
                raise TypeError(
                    "If not None, the q_index parameter should be an iterable, here"
                    " q_index is of type {}".format(type(q_index))
                )

            if exclusion_factor <= 0:
                raise ValueError(
                    "The value of exclusion_factor should be superior to 0, but got"
                    "{}".format(len(exclusion_factor))
                )

            i_instance, i_timestamp = q_index
            profile_length = n_timestamps - (q_length - 1)
            exclusion_LB = max(0, int(i_timestamp - q_length // exclusion_factor))
            exclusion_UB = min(
                profile_length, int(i_timestamp + q_length // exclusion_factor)
            )
            mask[i_instance, :, exclusion_LB:exclusion_UB] = False

        if self.normalize:
            self._q_means = np.mean(q, axis=-1)
            self._q_stds = np.std(q, axis=-1)
            self._store_mean_std_from_inputs(q_length)

        return self._predict(q.astype(float), mask)

    @abstractmethod
    def _fit(self, X, y):
        ...

    @abstractmethod
    def _predict(self, q):
        ...


# Dictionary structure :
#     1st lvl key : distance function used
#         2nd lvl key : boolean indicating wheter distance is normalized
DISTANCE_PROFILE_DICT = {
    "euclidean": {
        True: normalized_naive_euclidean_profile,
        False: naive_euclidean_profile,
    }
}
