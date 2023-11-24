"""Base class for similarity search."""

__author__ = ["baraline"]

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import final

import numpy as np
from numba.core.registry import CPUDispatcher

from aeon.base import BaseEstimator
from aeon.distances import get_distance_function
from aeon.similarity_search.distance_profiles import (
    naive_distance_profile,
    normalized_naive_distance_profile,
)
from aeon.utils.numba.general import sliding_mean_std_one_series


class BaseSimiliaritySearch(BaseEstimator, ABC):
    """
    BaseSimilaritySearch.

    The following distance functions are available from the
    aeon distance module :
    =============== ========================================
    distance        Distance Function
    =============== ========================================
    'dtw'           distance.dtw_distance
    'shape_dtw'     distance.shape_dtw_distance
    'ddtw'          distance.ddtw_distance
    'wdtw'          distance.wdtw_distance
    'wddtw'         distance.wddtw_distance
    'adtw'          distance.adtw_distance
    'erp'           distance.erp_distance
    'edr'           distance.edr_distance
    'msm'           distance.msm_distance
    'twe'           distance.twe_distance
    'lcss'          distance.lcss_distance
    'euclidean'     distance.euclidean_distance
    'squared'       distance.squared_distance
    =============== ========================================

    And the the following speed ups are available for
    similarity search module:
    =============== =============== ===============
    speed_up        distance        normalize
    =============== =============== ===============

    =============== =============== ===============

    Parameters
    ----------
    distance : str, default ="euclidean"
        Name of the distance function to use. The distance function
        must be one of the distance avaialble in the aeon distance module.
        This can also be a numba njit function used to compute the
        distance between two 1D vectors.
    normalize : bool, default = False
        Whether the distance function should be z-normalized.
    store_distance_profile : bool, default = False.
        Whether to store the computed distance profile in the attribute
        "_distance_profile" after calling the predict method.
    speed_up : str, default = None
        Which speed up technique to use with for the selected distance
        function.

    Attributes
    ----------
    _X : array, shape (n_instances, n_channels, n_timestamps)
        The input time series stored during the fit method.
    distance_profile_function : function
        The function used to compute the distance profile affected
        during the fit method based on the distance and normalize
        parameters.

    Notes
    -----
    For now, the multivariate case is only treated as independent.
    Distances are computed for each channel independently and then
    summed together.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:missing_values": False,
    }

    def __init__(
        self,
        distance="euclidean",
        normalize=False,
        store_distance_profile=False,
        speed_up=None,
    ):
        self.distance = distance
        self.normalize = normalize
        self.store_distance_profile = store_distance_profile
        self.speed_up = speed_up
        super(BaseSimiliaritySearch, self).__init__()

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
        q_dim, q_length = self._check_query_format(q)
        n_instances, _, n_timestamps = self._X.shape
        mask = self._apply_q_index_mask(
            q_index, q_dim, q_length, exclusion_factor=exclusion_factor
        )

        if self.normalize:
            self._q_means = np.mean(q, axis=-1)
            self._q_stds = np.std(q, axis=-1)
            self._store_mean_std_from_inputs(q_length)

        return self._predict(self._call_distance_profile(q, mask))

    def _apply_q_index_mask(self, q_index, q_dim, q_length, exclusion_factor=2.0):
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

        return mask

    def _check_query_format(self, q):
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
        return q_dim, q_length

    def _get_distance_profile_function(self):
        if isinstance(self.distance, str):
            if self.speed_up is None:
                self.distance_function_ = get_distance_function(self.distance)
            else:
                speed_up_profile = (
                    SPEED_UP_DICT.get(self.distance)
                    .get(self.normalize)
                    .get(self.speed_up)
                )
                if speed_up_profile is None:
                    raise ValueError(
                        f"Unknown or unsupported speed up {self.speed_up} for"
                        f"{self.distance} distance function with"
                    )
                return speed_up_profile
        else:
            if isinstance(self.distance, CPUDispatcher):
                self.distance_function_ = self.distance
            else:
                raise ValueError(
                    "If distance argument is not a string, it is expected to be a"
                    f"numba function (CPUDispatcher), but got {type(self.distance)}."
                )
        if self.normalize:
            return normalized_naive_distance_profile
        else:
            return naive_distance_profile

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

    def _call_distance_profile(self, q, mask):
        if self.normalize:
            distance_profile = self.distance_profile_function(
                self._X,
                q,
                mask,
                self._X_means,
                self._X_stds,
                self._q_means,
                self._q_stds,
                self.distance_function_,
            )
        else:
            distance_profile = self.distance_profile_function(
                self._X,
                q,
                mask,
                self.distance_function_,
            )
        # For now, deal with the multidimensional case as "dependent", so we sum.
        distance_profile = distance_profile.sum(axis=1)
        return distance_profile

    @abstractmethod
    def _fit(self, X, y):
        ...

    @abstractmethod
    def _predict(self, distance_profile):
        ...


# Dictionary structure :
#     1st lvl key : distance function used
#     2nd lvl key : boolean indicating whether distance is normalized
#     3rd lvl key : spzeed up name
SPEED_UP_DICT = {}
