"""Base class for series search."""

__maintainer__ = ["baraline"]

from typing import Union, final

import numpy as np
from numba import get_num_threads, set_num_threads
from numba.core.registry import CPUDispatcher

from aeon.distances import get_distance_function
from aeon.similarity_search.base import BaseSimilaritySearch
from aeon.similarity_search.series_methods import naive_series_search


class SeriesSearch(BaseSimilaritySearch):
    """
    Series search estimator.

    The series search estimator will return a set of matches for each subsequence of
    size L in a time series given during predict. The matching of each subsequence will
    be made against all subsequence of size L inside the time series given during fit,
    which will represent the search space.

    Depending on the `k` and/or `threshold` parameters, which condition what is
    considered a valid match during the search, the number of matches will vary. If `k`
    is used, at most `k` matches (the `k` best) will be returned, if `threshold` is used
    and `k` is set to `np.inf`, all the candidates which distance to the query is
    inferior or equal to `threshold` will be returned. If both are used, the `k` best
    matches to the query with distance inferior to `threshold` will be returned.


    Parameters
    ----------
    k : int, default=1
        The number of best matches to return during predict for each subsequence.
    threshold : float, default=np.inf
        The number of best matches to return during predict for each subsequence.
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
        `get_speedup_function_names` function.
    inverse_distance : bool, default=False
        If True, the matching will be made on the inverse of the distance, and thus, the
        worst matches to the query will be returned instead of the best ones.
    n_jobs : int, default=1
        Number of parallel jobs to use.

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
        k: int = 1,
        threshold: float = np.inf,
        distance: str = "euclidean",
        distance_args: Union[None, dict] = None,
        inverse_distance: bool = False,
        normalize: bool = False,
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
            normalize=normalize,
            speed_up=speed_up,
            n_jobs=n_jobs,
        )

    def _fit(self, X, y=None):
        """
        Check input format and store it to be used as search space during predict.

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
        self.X_ = X
        self.series_method_function_ = self._get_series_method_function()
        return self

    @final
    def predict(
        self,
        X,
        length,
        axis=1,
        X_index=None,
        exclusion_factor=2.0,
        apply_exclusion_to_result=False,
    ):
        """
        Predict function.

        Returns
        -------
        Tuple(np.ndarray, 2D array of shape (series_length - L + 1, n_matches), np.ndarray, 3D array of shape (series_length - L + 1, n_matches, 2)) # noqa: E501

        """
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)

        # TODO ...

        set_num_threads(prev_threads)
        return 0

    def _get_series_method_function(self):
        """
        Given distance and speed_up parameters, return the series method function.

        Raises
        ------
        ValueError
            If the distance parameter given at initialization is not a string nor a
            numba function or a callable, or if the speedup parameter is unknow or
            unsupported, raisea ValueError.

        Returns
        -------
        function
            The series method function matching the distance argument.

        """
        # TODO : test for correctness
        if isinstance(self.distance, str):
            distance_dict = _SERIES_SEARCH_SPEED_UP_DICT.get(self.distance)
            if self.speed_up is None or distance_dict is None:
                self.distance_function_ = get_distance_function(self.distance)
            else:
                speed_up_series_method = distance_dict.get(self.normalize).get(
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
            if isinstance(self.distance, CPUDispatcher) or callable(self.distance):
                self.distance_function_ = self.distance

            else:
                raise ValueError(
                    "If distance argument is not a string, it is expected to be either "
                    "a callable or a numba function (CPUDispatcher), but got "
                    f"{type(self.distance)}."
                )
        self.speed_up_ = None
        return naive_series_search

    @classmethod
    def get_speedup_function_names(self):
        """
        Get available speedup for series search in aeon.

        The returned structure is a dictionnary that contains the names of all
        avaialble speedups for normalized and non-normalized distance functions.

        Returns
        -------
        dict
            The available speedups name that can be used as parameters in
            similarity search classes.

        """
        speedups = {}
        for dist_name in _SERIES_SEARCH_SPEED_UP_DICT.keys():
            for normalize in _SERIES_SEARCH_SPEED_UP_DICT[dist_name].keys():
                speedups_names = list(
                    _SERIES_SEARCH_SPEED_UP_DICT[dist_name][normalize].keys()
                )
                if normalize:
                    speedups.update({f"normalized {dist_name}": speedups_names})
                else:
                    speedups.update({f"{dist_name}": speedups_names})
        return speedups


_SERIES_SEARCH_SPEED_UP_DICT = {}
