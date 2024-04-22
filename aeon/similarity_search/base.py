"""Base class for similarity search."""

__maintainer__ = []

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
from aeon.similarity_search.distance_profiles.euclidean_distance_profile import (
    euclidean_distance_profile,
    normalized_euclidean_distance_profile,
)
from aeon.similarity_search.distance_profiles.squared_distance_profile import (
    normalized_squared_distance_profile,
    squared_distance_profile,
)
from aeon.utils.numba.general import sliding_mean_std_one_series


class BaseSimiliaritySearch(BaseEstimator, ABC):
    """
    BaseSimilaritySearch.

    Parameters
    ----------
    distance : str, default="euclidean"
        Name of the distance function to use. A list of valid strings can be found in
        the documentation for :func:`aeon.distances.get_distance_function`.
        If a callable is passed it must either be a python function or numba function
        with nopython=True, that takes two 1d numpy arrays as input and returns a float.
    distance_args : dict, default=None
        Optional keyword arguments for the distance function.
    normalize : bool, default=False
        Whether the distance function should be z-normalized.
    store_distance_profile : bool, default=False.
        Whether to store the computed distance profile in the attribute
        "_distance_profile" after calling the predict method.
    speed_up : str, default=None
        Which speed up technique to use with for the selected distance
        function.

    Attributes
    ----------
    _X : array, shape (n_cases, n_channels, n_timepoints)
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
        distance_args=None,
        normalize=False,
        store_distance_profile=False,
        speed_up=None,
    ):
        self.distance = distance
        self.distance_args = distance_args
        self.normalize = normalize
        self.store_distance_profile = store_distance_profile
        self.speed_up = speed_up
        super().__init__()

    @final
    def fit(self, X, y=None):
        """
        Fit method: store the input data and get the distance profile function.

        Parameters
        ----------
        X : array, shape (n_cases, n_channels, n_timepoints)
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
                "Error, only supports 3D numpy of shape "
                "(n_cases, n_channels, n_timepoints)."
            )
        # Get distance function
        self.distance_profile_function = self._get_distance_profile_function()

        self._X = X.astype(float)
        self._fit(X, y)
        return self

    @final
    def predict(
        self, q, q_index=None, exclusion_factor=2.0, apply_exclusion_to_result=False
    ):
        """
        Predict method: Check the shape of q and call _predict to perform the search.

        If the distance profile function is normalized, it stores the mean and stds
        from q and _X.

        Parameters
        ----------
        q :  array, shape (n_channels, query_length)
            Input query used for similarity search.
        q_index : Iterable
            An Interable (tuple, list, array) of length two used to specify the index of
            the query q if it was extracted from the input data X given during the fit
            method. Given the tuple (id_sample, id_timestamp), the similarity search
            will define an exclusion zone around the q_index in order to avoid matching
            q with itself. If None, it is considered that the query is not extracted
            from X.
        exclusion_factor : float, default=2.
            The factor to apply to the query length to define the exclusion zone. The
            exclusion zone is define from
            :math:`id_timestamp - query_length//exclusion_factor` to
            :math:`id_timestamp + query_length//exclusion_factor`. This also applies to
            the matching conditions defined by child classes. For example, with
            TopKSimilaritySearch, the k best matches are also subject to the exclusion
            zone, but with :math:`id_timestamp` the index of one of the k matches.
        apply_exclusion_to_result: bool, default=False
            Wheter to apply the exclusion factor to the output of the similarity search.
            This means that two matches of the query from the same sample must be at
            least spaced by +/- :math:`query_length//exclusion_factor`.
            This can avoid pathological matching where, for example if we extract the
            best two matches, there is a high chance that if the best match is located
            at :math:`id_timestamp`, the second best match will be located at
            :math:`id_timestamp` +/- 1, as they both share all their values except one.

        Raises
        ------
        TypeError
            If the input q array is not 2D raise an error.
        ValueError
            If the length of the query is greater

        Returns
        -------
        array, shape (n_matches, 2)
            An array containing the indexes of the matches between q and _X.
            The decision of wheter a candidate of size query_length from _X is matched
            with Q depends on the subclasses that implent the _predict method
            (e.g. top-k, threshold, ...). The first index for each match is the sample
            id, the second is the timestamp id.

        """
        query_dim, query_length = self._check_query_format(q)
        n_cases, _, n_timepoints = self._X.shape
        mask = self._apply_q_index_mask(
            q_index, query_dim, query_length, exclusion_factor=exclusion_factor
        )

        if self.normalize:
            self._q_means = np.mean(q, axis=-1)
            self._q_stds = np.std(q, axis=-1)
            self._store_mean_std_from_inputs(query_length)

        if apply_exclusion_to_result:
            exclusion_size = query_length // exclusion_factor
        else:
            exclusion_size = None
        return self._predict(
            self._call_distance_profile(q, mask),
            exclusion_size=exclusion_size,
        )

    def _apply_q_index_mask(
        self, q_index, query_dim, query_length, exclusion_factor=2.0
    ):
        """
        Initiliaze the mask indicating the candidates to be evaluated in the search.

        Parameters
        ----------
        q_index : Iterable
            An Interable (tuple, list, array) of length two used to specify the index of
            the query q if it was extracted from the input data X given during the fit
            method. Given the tuple (id_sample, id_timestamp), the similarity search
            will define an exclusion zone around the q_index in order to avoid matching
            q with itself. If None, it is considered that the query is not extracted
            from X.
        query_dim : int
            Number of channels of the queries.
        query_length : int
            Length of the queries.
        exclusion_factor : float, optional
            The exclusion factor is used to prevent candidates close or equal to the
            query sample point to be returned as best matches. It is used to define a
            region between :math:`id_timestamp - query_length//exclusion_factor` and
            :math:`id_timestamp + query_length//exclusion_factor` which cannot be used
            in the search. The default is 2.0.

        Raises
        ------
        ValueError
            If the length of the q_index iterable is not two, will raise a ValueError.
        TypeError
            If q_index is not an iterable, will raise a TypeError.

        Returns
        -------
        mask : array, shape=(n_cases, n_timepoints - query_length + 1)
            Boolean array which indicates the candidates that should be evaluated in the
            similarity search.

        """
        n_cases, _, n_timepoints = self._X.shape
        mask = np.ones((n_cases, n_timepoints - query_length + 1), dtype=bool)

        if q_index is not None:
            if isinstance(q_index, Iterable):
                if len(q_index) != 2:
                    raise ValueError(
                        "The q_index should contain an interable of size 2 such as "
                        "(id_sample, id_timestamp), but got an iterable of "
                        "size {}".format(len(q_index))
                    )
            else:
                raise TypeError(
                    "If not None, the q_index parameter should be an iterable, here "
                    "q_index is of type {}".format(type(q_index))
                )

            if exclusion_factor <= 0:
                raise ValueError(
                    "The value of exclusion_factor should be superior to 0, but got "
                    "{}".format(len(exclusion_factor))
                )

            i_instance, i_timestamp = q_index
            profile_length = n_timepoints - query_length + 1
            exclusion_LB = max(0, int(i_timestamp - query_length // exclusion_factor))
            exclusion_UB = min(
                profile_length, int(i_timestamp + query_length // exclusion_factor)
            )
            mask[i_instance, exclusion_LB:exclusion_UB] = False

        return mask

    def _check_query_format(self, q):
        if not isinstance(q, np.ndarray) or q.ndim != 2:
            raise TypeError(
                "Error, only supports 2D numpy for now. If the query q is univariate "
                "do q.reshape(1,-1)."
            )

        query_dim, query_length = q.shape
        if query_length >= self._X.shape[-1]:
            raise ValueError(
                "The length of the query should be inferior or equal to the length of "
                "data (X) provided during fit, but got {} for q and {} for X".format(
                    query_length, self._X.shape[-1]
                )
            )

        if query_dim != self._X.shape[1]:
            raise ValueError(
                "The number of feature should be the same for the query q and the data "
                "(X) provided during fit, but got {} for q and {} for X".format(
                    query_dim, self._X.shape[1]
                )
            )
        return query_dim, query_length

    def _get_distance_profile_function(self):
        """
        Given distance and speed_up parameters, return the distance profile function.

        Raises
        ------
        ValueError
            If the distance parameter given at initialization is not a string nor a
            numba function or a callable, or if the speedup parameter is unknow or
            unsupported, raisea ValueError.

        Returns
        -------
        function
            The distance profile function matching the distance argument.

        """
        if isinstance(self.distance, str):
            if self.speed_up is None:
                self.distance_function_ = get_distance_function(self.distance)
            else:
                speed_up_profile = (
                    _SIM_SEARCH_SPEED_UP_DICT.get(self.distance)
                    .get(self.normalize)
                    .get(self.speed_up)
                )
                if speed_up_profile is None:
                    raise ValueError(
                        f"Unknown or unsupported speed up {self.speed_up} for "
                        f"{self.distance} distance function with"
                    )
                return speed_up_profile
        else:
            if isinstance(self.distance, CPUDispatcher) or callable(self.distance):
                self.distance_function_ = self.distance
            else:
                raise ValueError(
                    "If distance argument is not a string, it is expected to be either "
                    "a callable or a numba function (CPUDispatcher), but got "
                    f"{type(self.distance)}."
                )
        if self.normalize:
            return normalized_naive_distance_profile
        else:
            return naive_distance_profile

    def _store_mean_std_from_inputs(self, query_length):
        """
        Store the mean and std of each subsequence of size query_length in _X.

        Parameters
        ----------
        query_length : int
            Length of the query.

        Returns
        -------
        None.

        """
        n_cases, n_channels, n_timepoints = self._X.shape
        search_space_size = n_timepoints - query_length + 1

        means = np.zeros((n_cases, n_channels, search_space_size))
        stds = np.zeros((n_cases, n_channels, search_space_size))

        for i in range(n_cases):
            _mean, _std = sliding_mean_std_one_series(self._X[i], query_length, 1)
            stds[i] = _std
            means[i] = _mean

        self._X_means = means
        self._X_stds = stds

    def _call_distance_profile(self, q, mask):
        """
        Obtain the distance profile function and call it with the query and the mask.

        Parameters
        ----------
        q :  array, shape (n_channels, query_length)
            Input query used for similarity search.
         mask : array, shape=(n_cases, n_timepoints - query_length + 1)
             Boolean array which indicates the candidates that should be evaluated in
             the similarity search.

        Returns
        -------
        distance_profile : array, shape=(n_cases, n_timepoints - query_length + 1)
            The distance profiles between the input time series and the query.

        """
        if self.speed_up is None:
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
                    distance_args=self.distance_args,
                )
            else:
                distance_profile = self.distance_profile_function(
                    self._X,
                    q,
                    mask,
                    self.distance_function_,
                    distance_args=self.distance_args,
                )
        else:
            if self.normalize:
                distance_profile = self.distance_profile_function(
                    self._X,
                    q,
                    mask,
                    self._X_means,
                    self._X_stds,
                    self._q_means,
                    self._q_stds,
                )
            else:
                distance_profile = self.distance_profile_function(self._X, q, mask)
        # For now, deal with the multidimensional case as "dependent", so we sum.
        distance_profile = distance_profile.sum(axis=1)
        return distance_profile

    @abstractmethod
    def _fit(self, X, y): ...

    @abstractmethod
    def _predict(self, distance_profile, exclusion_size=None): ...


_SIM_SEARCH_SPEED_UP_DICT = {
    "euclidean": {
        True: {"ConvolveDotProduct": normalized_euclidean_distance_profile},
        False: {"ConvolveDotProduct": euclidean_distance_profile},
    },
    "squared": {
        True: {"ConvolveDotProduct": normalized_squared_distance_profile},
        False: {"ConvolveDotProduct": squared_distance_profile},
    },
}


def get_speedup_function_names():
    """
    Get available speedup for similarity search in aeon.

    The returned structure is a dictionnary that contains the names of all avaialble
    speedups for normalized and non-normalized distance functions.

    Returns
    -------
    dict
        The available speedups name that can be used as parameters in similarity search
        classes.

    """
    speedups = {}
    for dist_name in _SIM_SEARCH_SPEED_UP_DICT.keys():
        for normalize in _SIM_SEARCH_SPEED_UP_DICT[dist_name].keys():
            speedups_names = list(
                _SIM_SEARCH_SPEED_UP_DICT[dist_name][normalize].keys()
            )
            if normalize:
                speedups.update({f"normalized {dist_name}": speedups_names})
            else:
                speedups.update({f"{dist_name}": speedups_names})
    return speedups
