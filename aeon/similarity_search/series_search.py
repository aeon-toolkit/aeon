"""Base class for series search."""

__maintainer__ = ["baraline"]

from typing import Union, final

import numpy as np
from numba import get_num_threads, set_num_threads

from aeon.similarity_search.base import BaseSimilaritySearch
from aeon.similarity_search.matrix_profiles.stomp import (
    stomp_euclidean_matrix_profile,
    stomp_normalised_euclidean_matrix_profile,
    stomp_normalised_squared_matrix_profile,
    stomp_squared_matrix_profile,
)
from aeon.utils.numba.general import sliding_mean_std_one_series


class SeriesSearch(BaseSimilaritySearch):
    """
    Series search estimator.

    The series search estimator will return a set of matches for each subsequence of
    size L in a time series given during predict. The matching of each subsequence will
    be made against all subsequences of size L inside the time series given during fit,
    which will represent the search space.

    Depending on the ``k`` and/or ``threshold`` parameters, which condition what is
    considered a valid match during the search, the number of matches will vary. If ``k``
    is used, at most ``k`` matches (the ``k`` best) will be returned. If ``threshold`` is
    used and ``k`` is set to ``np.inf``, all candidates whose distance to the query is
    less than or equal to ``threshold`` will be returned. If both are used, the ``k`` best
    matches to the query with a distance less than ``threshold`` will be returned.

    Parameters
    ----------
    k : int, default=1
        The number of best matches to return during predict for each subsequence.
    threshold : float, default=np.inf
        The maximum allowable distance for a valid match.
    distance : str, default="euclidean"
        Name of the distance function to use. A list of valid strings can be found in
        the documentation for :func:`aeon.distances.get_distance_function`.
        If a callable is passed, it must either be a Python function or a Numba function
        with ``nopython=True`` that takes two 1D NumPy arrays as input and returns a float.
    distance_args : dict, default=None
        Optional keyword arguments for the distance function.
    normalise : bool, default=False
        Whether the distance function should be z-normalised.
    speed_up : str, default='fastest'
        Which speed-up technique to use for the selected distance
        function. By default, the fastest algorithm is used. A list of available
        algorithms for each distance can be obtained by calling the
        ``get_speedup_function_names`` function.
    inverse_distance : bool, default=False
        If ``True``, the matching will be based on the inverse of the distance, meaning
        the worst matches to the query will be returned instead of the best ones.
    n_jobs : int, default=1
        Number of parallel jobs to use.

    Attributes
    ----------
    X_ : np.ndarray, shape (n_cases, n_channels, n_timepoints)
        The input time series stored during the ``fit`` method. This is the
        database we search in when given a query.
    distance_profile_function : function
        The function used to compute the distance profile. This is determined
        during the ``fit`` method based on the ``distance`` and ``normalise``
        parameters.

    Notes
    -----
    For now, the multivariate case is only treated as independent.
    Distances are computed for each channel independently and then
    summed together.
    """

    def __init__(
        self,
        k: int = 1,
        threshold: float = np.inf,
        distance: str = "euclidean",
        distance_args: Union[None, dict] = None,
        inverse_distance: bool = False,
        normalise: bool = False,
        speed_up: str = "fastest",
        n_jobs: int = 1,
    ):
        self.k = k
        self.threshold = threshold
        self._previous_query_length = -1
        self.axis = 1

        super().__init__(
            distance=distance,
            distance_args=distance_args,
            inverse_distance=inverse_distance,
            normalise=normalise,
            speed_up=speed_up,
            n_jobs=n_jobs,
        )

    def _fit(self, X, y=None):
        """
        Check input format and store it to be used as a search space during prediction.

        Parameters
        ----------
        X : np.ndarray, shape (n_cases, n_channels, n_timepoints)
            Input array to be used as the database for the similarity search.
        y : optional
            Not used.

        Raises
        ------
        TypeError
            If the input ``X`` array is not 3D, an error is raised.

        Returns
        -------
        self

        """
        self.X_ = X
        self.matrix_profile_function_ = self._get_series_method_function()
        return self

    @final
    def predict(
        self,
        X: np.ndarray,
        length: int,
        axis: int = 1,
        X_index=None,
        exclusion_factor=2.0,
        apply_exclusion_to_result=False,
    ):
        """
        Predict method: Checks the shape of X and calls `_predict` to perform the search.

        If the distance profile function is normalized, it stores the mean and standard
        deviations from `X` and `X_`, where `X_` is the training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_channels, series_length)
            Input time series used for the search.
        length : int
            The length parameter that will be used to extract queries from `X`.
        axis : int
            The time point axis of the input series if it is 2D. If ``axis == 0``, it is
            assumed each column is a time series and each row is a time point, i.e., the
            shape of the data is ``(n_timepoints, n_channels)``. If ``axis == 1``, the time
            series are in rows, i.e., the shape of the data is ``(n_channels, n_timepoints)``.
        X_index : int, optional
            An integer indicating whether `X` was extracted from the dataset given during
            the `fit` method. If so, this integer should be the sample ID.
            The search will define an exclusion zone for the queries extracted from `X`
            to avoid self-matching. If `None`, it is assumed that the query is not extracted
            from `X_`.
        exclusion_factor : float, default=2.0
            The factor used to define the exclusion zone relative to `query_length`. The
            exclusion zone is defined from
            ``id_timestamp - query_length // exclusion_factor`` to
            ``id_timestamp + query_length // exclusion_factor``.
            This also applies to the matching conditions defined by child classes.
            For example, in `TopKSimilaritySearch`, the `k` best matches are also subject
            to the exclusion zone, but with ``id_timestamp`` being the index of one of
            the `k` matches.
        apply_exclusion_to_result : bool, default=False
            Whether to apply the exclusion factor to the similarity search output.
            This ensures that two matches of the query from the same sample must be at
            least ``query_length // exclusion_factor`` apart.
            This prevents pathological matching cases where, for example, if we extract
            the top two matches, the second best match is likely to be located at
            ``id_timestamp ± 1``, as they share all but one value.

        Raises
        ------
        TypeError
            If the input `X` array is not 2D, an error is raised.
        ValueError
            If the length of the query is greater than `series_length`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - The first array, of shape ``(series_length - length + 1, n_matches)``,
              contains the distances between all queries of size `length` and their best
              matches in `X_`.
            - The second array, of shape ``(series_length - length + 1, n_matches, 2)``,
              contains the indexes of these matches as ``(id_sample, id_timepoint)``.
              The corresponding match can be retrieved as:
              ``X_[id_sample, :, id_timepoint : id_timepoint + length]``.

        """
        self._check_is_fitted()
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        series_dim, series_length = self._check_series_format(X, length, axis)

        mask = self._init_X_index_mask(
            None if X_index is None else [X_index, 0],
            length,
            exclusion_factor=exclusion_factor,
        )

        if self.normalise:
            _mean, _std = sliding_mean_std_one_series(X, length, 1)
            self.T_means_ = _mean
            self.T_stds_ = _std
            if self._previous_query_length != length:
                self._store_mean_std_from_inputs(length)

        if apply_exclusion_to_result:
            exclusion_size = length // exclusion_factor
        else:
            exclusion_size = None

        self._previous_query_length = length

        X_preds = self._predict(
            X,
            length,
            mask,
            exclusion_size,
            X_index,
            exclusion_factor,
            apply_exclusion_to_result,
        )
        set_num_threads(prev_threads)
        return X_preds

    def _predict(
        self,
        X,
        length,
        mask,
        exclusion_size,
        X_index,
        exclusion_factor,
        apply_exclusion_to_result,
    ):
        """
        Private `_predict` method for `SeriesSearch`.

        This method calculates the matrix profile for a given time series dataset by
        comparing all possible subsequences of a specified length against a reference
        time series. It handles exclusion zones to prevent nearby matches from being
        selected and supports normalization.

        Parameters
        ----------
        X : np.ndarray, shape (n_channels, series_length)
            Input time series used for the search.
        length : int
            The length parameter used to extract queries from `X`.
        axis : int
            The time point axis of the input series if it is 2D.
            - If ``axis == 0``, each column is a time series and each row is a time point,
              i.e., the shape of the data is ``(n_timepoints, n_channels)``.
            - If ``axis == 1``, the time series are in rows, i.e., the shape is
              ``(n_channels, n_timepoints)``.
        mask : np.ndarray, shape (n_cases, n_timepoints - length + 1)
            Boolean mask of the distance profiles, indicating where the distance
            should be computed. Initially, it is the mask for the first query of size
            `length` in `X`. This mask will be updated during the algorithm.
        exclusion_size : int, optional
            The size of the exclusion zone used to prevent returning consecutive
            top `k` candidates that are too close to each other (e.g., `i` and `i+1`).
            It defines a region between
            ``id_timestamp - exclusion_size`` and ``id_timestamp + exclusion_size``,
            which cannot be returned as the best match if ``id_timestamp`` was
            already selected. If `None`, no exclusion zone is used.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - The first array, of shape ``(series_length - length + 1, n_matches)``,
              contains the distances between all queries of size `length` and their
              best matches in `X_`.
            - The second array, of shape ``(series_length - length + 1, n_matches, 2)``,
              contains the indexes of these matches as ``(id_sample, id_timepoint)``.
              The corresponding match can be retrieved as:
              ``X_[id_sample, :, id_timepoint : id_timepoint + length]``.

        """
        if self.normalise:
            return self.matrix_profile_function_(
                self.X_,
                X,
                length,
                self.X_means_,
                self.X_stds_,
                self.T_means_,
                self.T_stds_,
                mask,
                k=self.k,
                threshold=self.threshold,
                inverse_distance=self.inverse_distance,
                exclusion_size=exclusion_size,
            )
        else:
            return self.matrix_profile_function_(
                self.X_,
                X,
                length,
                mask,
                k=self.k,
                threshold=self.threshold,
                inverse_distance=self.inverse_distance,
                exclusion_size=exclusion_size,
            )

    def _check_series_format(self, X, length, axis):
        if axis not in [0, 1]:
            raise ValueError("The axis argument is expected to be either 1 or 0")
        if self.axis != axis:
            X = X.T
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise TypeError(
                "Error, only supports 2D numpy for now. If the series X is univariate "
                "do X = X[np.newaxis, :]."
            )

        series_dim, series_length = X.shape
        if series_length < length:
            raise ValueError(
                "The length of the series should be superior or equal to the length "
                "parameter given during predict, but got {} < {}".format(
                    series_length, length
                )
            )

        if series_dim != self.n_channels_:
            raise ValueError(
                "The number of feature should be the same for the series X and the data"
                " (X_) provided during fit, but got {} for X and {} for X_".format(
                    series_dim, self.n_channels_
                )
            )
        return series_dim, series_length

    def _get_series_method_function(self):
        """
        Given `distance` and `speed_up` parameters, return the corresponding series method function.

        Raises
        ------
        ValueError
            - If the `distance` parameter provided at initialization is neither a string,
              a Numba function, nor a callable.
            - If the `speed_up` parameter is unknown or unsupported.

        Returns
        -------
        function
            The series method function that corresponds to the given `distance` argument.

        """
        if isinstance(self.distance, str):
            distance_dict = _SERIES_SEARCH_SPEED_UP_DICT.get(self.distance)
            if distance_dict is None:
                raise NotImplementedError(
                    f"No distance profile have been implemented for {self.distance}."
                )
            else:
                speed_up_series_method = distance_dict.get(self.normalise).get(
                    self.speed_up
                )

                if speed_up_series_method is None:
                    raise ValueError(
                        f"Unknown or unsupported speed up {self.speed_up} for "
                        f"{self.distance} distance function with"
                    )
                self.speed_up_ = self.speed_up
                return speed_up_series_method
        else:
            raise ValueError(
                f"Expected distance argument to be str but got {type(self.distance)}"
            )

    @classmethod
    def get_speedup_function_names(self):
        """
        Get available speedup options for series search in Aeon.

        The returned structure is a dictionary containing the names of all
        available speedup techniques for both normalized and non-normalized
        distance functions.

        Returns
        -------
        dict
            A dictionary of available speedup options that can be used as
            parameters in similarity search classes.

        """
        speedups = {}
        for dist_name in _SERIES_SEARCH_SPEED_UP_DICT.keys():
            for normalise in _SERIES_SEARCH_SPEED_UP_DICT[dist_name].keys():
                speedups_names = list(
                    _SERIES_SEARCH_SPEED_UP_DICT[dist_name][normalise].keys()
                )
                if normalise:
                    speedups.update({f"normalised {dist_name}": speedups_names})
                else:
                    speedups.update({f"{dist_name}": speedups_names})
        return speedups


_SERIES_SEARCH_SPEED_UP_DICT = {
    "euclidean": {
        True: {
            "fastest": stomp_normalised_euclidean_matrix_profile,
            "STOMP": stomp_normalised_euclidean_matrix_profile,
        },
        False: {
            "fastest": stomp_euclidean_matrix_profile,
            "STOMP": stomp_euclidean_matrix_profile,
        },
    },
    "squared": {
        True: {
            "fastest": stomp_normalised_squared_matrix_profile,
            "STOMP": stomp_normalised_squared_matrix_profile,
        },
        False: {
            "fastest": stomp_squared_matrix_profile,
            "STOMP": stomp_squared_matrix_profile,
        },
    },
}
