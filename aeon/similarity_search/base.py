"""Base class for similarity search."""

__maintainer__ = ["baraline"]

from abc import abstractmethod
from collections.abc import Iterable
from typing import Optional, final

import numpy as np
from numba import get_num_threads, set_num_threads
from numba.typed import List

from aeon.base import BaseCollectionEstimator
from aeon.utils.numba.general import sliding_mean_std_one_series


class BaseSimilaritySearch(BaseCollectionEstimator):
    """
    Base class for similarity search applications.

    Parameters
    ----------
    distance : str, default="euclidean"
        Name of the distance function to use. A list of valid strings can be found in
        the documentation for :func:`aeon.distances.get_distance_function`.
        If a callable is passed it must either be a python function or numba function
        with nopython=True, that takes two 1d numpy arrays as input and returns a float.
    distance_args : dict, default=None
        Optional keyword arguments for the distance function.
    inverse_distance : bool, default=False
        If True, the matching will be made on the inverse of the distance, and thus, the
        worst matches to the query will be returned instead of the best ones.
    normalise : bool, default=False
        Whether the distance function should be z-normalised.
    speed_up : str, default='fastest'
        Which speed up technique to use with for the selected distance
        function. By default, the fastest algorithm is used. A list of available
        algorithm for each distance can be obtained by calling the
        `get_speedup_function_names` function of the child classes.
    n_jobs : int, default=1
        Number of parallel jobs to use.

    Attributes
    ----------
    X_ : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input time series stored during the fit method.

    Notes
    -----
    For now, the multivariate case is only treated as independent.
    Distances are computed for each channel independently and then
    summed together.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:multithreading": True,
        "fit_is_empty": False,
        "X_inner_type": ["np-list", "numpy3D"],
    }

    @abstractmethod
    def __init__(
        self,
        distance: str = "euclidean",
        distance_args: Optional[dict] = None,
        inverse_distance: bool = False,
        normalise: bool = False,
        speed_up: str = "fastest",
        n_jobs: int = 1,
    ):
        self.distance = distance
        self.distance_args = distance_args
        self.inverse_distance = inverse_distance
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.speed_up = speed_up
        super().__init__()

    @final
    def fit(self, X: np.ndarray, y=None):
        """
        Fit method: data preprocessing and storage.

        Parameters
        ----------
        X : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
            Input array to be used as database for the similarity search
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
        prev_threads = get_num_threads()
        X = self._preprocess_collection(X)
        # Store minimum number of n_timepoints for unequal length collections
        self.min_timepoints_ = min([X[i].shape[-1] for i in range(len(X))])
        self.n_channels_ = X[0].shape[0]
        self.n_cases_ = len(X)
        if self.metadata_["unequal_length"]:
            X = List(X)
        set_num_threads(self._n_jobs)
        self._fit(X, y)
        set_num_threads(prev_threads)
        self.is_fitted = True
        return self

    def _store_mean_std_from_inputs(self, query_length: int) -> None:
        """
        Store the mean and std of each subsequence of size query_length in X_.

        Parameters
        ----------
        query_length : int
            Length of the query.

        Returns
        -------
        None

        """
        means = []
        stds = []

        for i in range(len(self.X_)):
            _mean, _std = sliding_mean_std_one_series(self.X_[i], query_length, 1)

            stds.append(_std)
            means.append(_mean)

        self.X_means_ = List(means)
        self.X_stds_ = List(stds)

    def _init_X_index_mask(
        self,
        X_index: Optional[Iterable[int]],
        query_length: int,
        exclusion_factor: Optional[float] = 2.0,
    ) -> np.ndarray:
        """
        Initiliaze the mask indicating the candidates to be evaluated in the search.

        Parameters
        ----------
        X_index : Iterable
            Any Iterable (tuple, list, array) of length two used to specify the index of
            the query X if it was extracted from the input data X given during the fit
            method. Given the tuple (id_sample, id_timestamp), the similarity search
            will define an exclusion zone around the X_index in order to avoid matching
            X with itself. If None, it is considered that the query is not extracted
            from X_ (the training data).
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
        mask : np.ndarray, 2D array of shape (n_cases, n_timepoints - query_length + 1)
            Boolean array which indicates the candidates that should be evaluated in the
            similarity search.

        """
        if self.metadata_["unequal_length"]:
            mask = List(
                [
                    np.ones(self.X_[i].shape[1] - query_length + 1, dtype=bool)
                    for i in range(self.n_cases_)
                ]
            )
        else:
            mask = np.ones(
                (self.n_cases_, self.min_timepoints_ - query_length + 1),
                dtype=bool,
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

    @abstractmethod
    def _fit(self, X, y=None): ...

    @abstractmethod
    def get_speedup_function_names(self):
        """Return a dictionnary containing the name of the speedup functions."""
        ...
