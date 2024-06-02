"""Base class for similarity search."""

__maintainer__ = ["baraline"]

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import final

import numpy as np
from numba import get_num_threads, set_num_threads
from numba.core.registry import CPUDispatcher
from numba.typed import List

from aeon.distances import get_distance_function
from aeon.similarity_search.base import BaseSimiliaritySearch
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


class BaseQuerySearch(BaseSimiliaritySearch, ABC):
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
    speed_up : str, default='fastest'
        Which speed up technique to use with for the selected distance
        function. By default, the fastest algorithm is used. A list of available
        algorithm for each distance can be obtained by calling the
        `get_speedup_function_names` function of the child classes.
    n_jobs : int, default=1
        Number of parallel jobs to use.
    store_distance_profiles : bool, default=False.
        Whether to store the computed distance profiles in the attribute
        "distance_profiles_" after calling the predict method.

    Attributes
    ----------
    X_ : array, shape (n_cases, n_channels, n_timepoints)
        The input time series stored during the fit method. This is the
        database we search in when given a query.
    distance_profile_function : function
        The function used to compute the distance profile. This is determined
        during the fit method based on the distance and normalize
        parameters.

    Notes
    -----
    For now, the multivariate case is only treated as independent.
    Distances are computed for each channel independently and then
    summed together.
    """

    def __init__(
        self,
        distance="euclidean",
        distance_args=None,
        normalize=False,
        speed_up="fastest",
        n_jobs=1,
        store_distance_profiles=False,
    ):
        self.store_distance_profiles = store_distance_profiles
        self._previous_query_length = -1
        self.axis = 1

        super().__init__(
            distance=distance,
            distance_args=distance_args,
            normalize=normalize,
            speed_up=speed_up,
            n_jobs=n_jobs,
        )

    def _fit(self, X, y=None):
        """
        Fetch the distance function to be used in query search.

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
        # Get distance function
        self.distance_profile_function_ = self._get_distance_profile_function()

        return self

    @final
    def predict(
        self,
        X,
        axis=1,
        X_index=None,
        exclusion_factor=2.0,
        apply_exclusion_to_result=False,
    ):
        """
        Predict method: Check the shape of X and call _predict to perform the search.

        If the distance profile function is normalized, it stores the mean and stds
        from X and X_, with X_ the training data.

        Parameters
        ----------
        X :  array, shape (n_channels, query_length)
            Input query used for similarity search.
        axis: int
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point. i.e. the
            shape of the data is ``(n_timepoints,n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels,n_timepoints)``.
        X_index : Iterable
            An Interable (tuple, list, array) of length two used to specify the index of
            the query X if it was extracted from the input data X given during the fit
            method. Given the tuple (id_sample, id_timestamp), the similarity search
            will define an exclusion zone around the X_index in order to avoid matching
            X with itself. If None, it is considered that the query is not extracted
            from X_.
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
            If the input X array is not 2D raise an error.
        ValueError
            If the length of the query is greater

        Returns
        -------
        array, shape (n_matches, 2)
            An array containing the indexes of the matches between X and X_.
            The decision of wheter a candidate of size query_length from X_ is matched
            with X depends on the subclasses that implent the _predict method
            (e.g. top-k, threshold, ...). The first index for each match is the sample
            id, the second is the timestamp id.

        """
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)

        query_dim, query_length = self._check_query_format(X, axis)

        mask = self._init_X_index_mask(
            X_index,
            query_dim,
            query_length,
            exclusion_factor=exclusion_factor,
        )

        if self.normalize:
            self.query_means_ = np.mean(X, axis=-1)
            self.query_stds_ = np.std(X, axis=-1)
            if self._previous_query_length != query_length:
                self._store_mean_std_from_inputs(query_length)

        if apply_exclusion_to_result:
            exclusion_size = query_length // exclusion_factor
        else:
            exclusion_size = None

        self._previous_query_length = query_length

        X_preds = self._predict(
            self._call_distance_profile(X, mask),
            exclusion_size=exclusion_size,
        )
        set_num_threads(prev_threads)
        return X_preds

    def _init_X_index_mask(
        self, X_index, query_dim, query_length, exclusion_factor=2.0
    ):
        """
        Initiliaze the mask indicating the candidates to be evaluated in the search.

        Parameters
        ----------
        X_index : Iterable
            An Interable (tuple, list, array) of length two used to specify the index of
            the query X if it was extracted from the input data X given during the fit
            method. Given the tuple (id_sample, id_timestamp), the similarity search
            will define an exclusion zone around the X_index in order to avoid matching
            X with itself. If None, it is considered that the query is not extracted
            from X_ (the training data).
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
        if self.metadata_["unequal_length"]:
            mask = np.ones(
                (self.n_cases_, self.min_timepoints_ - query_length + 1),
                dtype=bool,
            )
        else:
            mask = List(
                [
                    np.ones(self.X_[i].shape[1] - query_length + 1, dtype=bool)
                    for i in range(self.n_cases_)
                ]
            )
        if X_index is not None:
            if isinstance(X_index, Iterable):
                if len(X_index) != 2:
                    raise ValueError(
                        "The X_index should contain an interable of size 2 such as "
                        "(id_sample, id_timestamp), but got an iterable of "
                        "size {}".format(len(X_index))
                    )
            else:
                raise TypeError(
                    "If not None, the X_index parameter should be an iterable, here "
                    "X_index is of type {}".format(type(X_index))
                )

            if exclusion_factor <= 0:
                raise ValueError(
                    "The value of exclusion_factor should be superior to 0, but got "
                    "{}".format(len(exclusion_factor))
                )

            i_instance, i_timestamp = X_index
            profile_length = self.X_[i_instance].shape[1] - query_length + 1
            exclusion_LB = max(0, int(i_timestamp - query_length // exclusion_factor))
            exclusion_UB = min(
                profile_length,
                int(i_timestamp + query_length // exclusion_factor),
            )
            mask[i_instance][exclusion_LB:exclusion_UB] = False

        return mask

    def _check_query_format(self, X, axis):
        if axis not in [0, 1]:
            raise ValueError("The axis argument is expected to be either 1 or 0")
        if self.axis != axis:
            X = X.T
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise TypeError(
                "Error, only supports 2D numpy for now. If the query X is univariate "
                "do X.reshape(1,-1)."
            )

        query_dim, query_length = X.shape
        if query_length >= self.min_timepoints_:
            raise ValueError(
                "The length of the query should be inferior or equal to the length of "
                "data (X_) provided during fit, but got {} for X and {} for X_".format(
                    query_length, self.min_timepoints_
                )
            )

        if query_dim != self.n_channels_:
            raise ValueError(
                "The number of feature should be the same for the query X and the data "
                "(X_) provided during fit, but got {} for X and {} for X_".format(
                    query_dim, self.n_channels_
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
            distance_dict = _SIM_SEARCH_SPEED_UP_DICT.get(self.distance)
            if self.speed_up is None or distance_dict is None:
                self.distance_function_ = get_distance_function(self.distance)
            else:
                speed_up_profile = distance_dict.get(self.normalize).get(self.speed_up)

                if speed_up_profile is None:
                    raise ValueError(
                        f"Unknown or unsupported speed up {self.speed_up} for "
                        f"{self.distance} distance function with"
                    )
                self.speed_up_ = self.speed_up
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
        self.speed_up_ = None
        if self.normalize:
            return normalized_naive_distance_profile
        else:
            return naive_distance_profile

    def _call_distance_profile(self, X, mask):
        """
        Obtain the distance profile function and call it with the query and the mask.

        Parameters
        ----------
        X :  array, shape (n_channels, query_length)
            Input query used for similarity search.
         mask : array, shape=(n_cases, n_timepoints - query_length + 1)
             Boolean array which indicates the candidates that should be evaluated in
             the similarity search.

        Returns
        -------
        distance_profiles : array, shape=(n_cases, n_timepoints - query_length + 1)
            The distance profiles between the input time series and the query.

        """
        if self.speed_up_ is None:
            if self.normalize:
                distance_profiles = self.distance_profile_function_(
                    self.X_,
                    X,
                    mask,
                    self.X_means_,
                    self.X_stds_,
                    self.query_means_,
                    self.query_stds_,
                    self.distance_function_,
                    distance_args=self.distance_args,
                )
            else:
                distance_profiles = self.distance_profile_function_(
                    self.X_,
                    X,
                    mask,
                    self.distance_function_,
                    distance_args=self.distance_args,
                )
        else:
            if self.normalize:
                distance_profiles = self.distance_profile_function_(
                    self.X_,
                    X,
                    mask,
                    self.X_means_,
                    self.X_stds_,
                    self.query_means_,
                    self.query_stds_,
                )
            else:
                distance_profiles = self.distance_profile_function_(self.X_, X, mask)
        # For now, deal with the multidimensional case as "dependent", so we sum.
        if self.metadata_["unequal_length"]:
            distance_profiles = List(
                [distance_profiles[i].sum(axis=0) for i in range(self.n_cases_)]
            )
        else:
            distance_profiles = distance_profiles.sum(axis=1)
        return distance_profiles

    @classmethod
    def get_speedup_function_names(self):
        """
        Get available speedup for similarity search in aeon.

        The returned structure is a dictionnary that contains the names of all
        avaialble speedups for normalized and non-normalized distance functions.

        Returns
        -------
        dict
            The available speedups name that can be used as parameters in
            similarity search classes.

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

    @abstractmethod
    def _predict(self, distance_profile, exclusion_size=None): ...


_SIM_SEARCH_SPEED_UP_DICT = {
    "euclidean": {
        True: {
            "fastest": normalized_euclidean_distance_profile,
            "Mueen": normalized_euclidean_distance_profile,
        },
        False: {
            "fastest": euclidean_distance_profile,
            "Mueen": euclidean_distance_profile,
        },
    },
    "squared": {
        True: {
            "fastest": normalized_squared_distance_profile,
            "Mueen": normalized_squared_distance_profile,
        },
        False: {
            "fastest": squared_distance_profile,
            "Mueen": squared_distance_profile,
        },
    },
}
