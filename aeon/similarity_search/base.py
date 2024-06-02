"""Base class for similarity search."""

__maintainer__ = ["baraline"]

from abc import ABC, abstractmethod
from typing import final

from numba import get_num_threads, set_num_threads
from numba.typed import List

from aeon.base import BaseCollectionEstimator
from aeon.utils.numba.general import sliding_mean_std_one_series


class BaseSimiliaritySearch(BaseCollectionEstimator, ABC):
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
    store_distance_profiles : bool, default=False.
        Whether to store the computed distance profiles in the attribute
        "distance_profiles_" after calling the predict method.
    speed_up : str, default='fastest'
        Which speed up technique to use with for the selected distance
        function. By default, the fastest algorithm is used. A list of available
        algorithm for each distance can be obtained by calling the
        `get_speedup_function_names` function of the child classes.
    n_jobs : int, default=1
        Number of parallel jobs to use.


    Attributes
    ----------
    X_ : array, shape (n_cases, n_channels, n_timepoints)
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
        "X_inner_type": ["np-list", "numpy3D"],
    }

    def __init__(
        self,
        distance="euclidean",
        distance_args=None,
        normalize=False,
        speed_up="fastest",
        n_jobs=1,
    ):
        self.distance = distance
        self.distance_args = distance_args
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.speed_up = speed_up
        super().__init__()

    @final
    def fit(self, X, y=None):
        """
        Fit method: data preprocessing and storage.

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
        prev_threads = get_num_threads()
        X = self._preprocess_collection(X)
        # Store minimum number of n_timepoints for unequal length collections
        self.min_timepoints_ = min([X[i].shape[-1] for i in range(len(X))])
        self.n_channels_ = X[0].shape[0]
        self.n_cases_ = len(X)
        if self.metadata_["unequal_length"]:
            X = List(X)
        set_num_threads(self._n_jobs)
        self.X_ = X
        self._fit(X, y)
        set_num_threads(prev_threads)
        return self

    def _store_mean_std_from_inputs(self, query_length):
        """
        Store the mean and std of each subsequence of size query_length in X_.

        Parameters
        ----------
        query_length : int
            Length of the query.

        Returns
        -------
        None.

        """
        means = []
        stds = []

        for i in range(len(self.X_)):
            _mean, _std = sliding_mean_std_one_series(self.X_[i], query_length, 1)

            stds.append(_std)
            means.append(_mean)

        self.X_means_ = List(means)
        self.X_stds_ = List(stds)

    @abstractmethod
    def _fit(self, X, y=None): ...

    @abstractmethod
    def get_speedup_function_names(self):
        """Return a dictionnary containing the name of the speedup functions."""
        ...
