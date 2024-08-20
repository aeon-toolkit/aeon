"""Base class for similarity search."""

__maintainer__ = ["baraline"]

from abc import ABC, abstractmethod
from typing import Optional, final

import numpy as np
from numba import get_num_threads, set_num_threads
from numba.typed import List

from aeon.base import BaseCollectionEstimator


class BaseSimilaritySearch(BaseCollectionEstimator, ABC):
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
    normalize : bool, default=False
        Whether the distance function should be z-normalized.
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
        "X_inner_type": ["np-list", "numpy3D"],
    }

    def __init__(
        self,
        distance: str = "euclidean",
        distance_args: Optional[dict] = None,
        inverse_distance: bool = False,
        normalize: bool = False,
        speed_up: str = "fastest",
        channel_dependency: str = "independent",
        n_jobs: int = 1,
    ):
        self.distance = distance
        self.distance_args = distance_args
        self.inverse_distance = inverse_distance
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.speed_up = speed_up
        self.channel_dependency = channel_dependency.lower()
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
        return self

    @abstractmethod
    def _fit(self, X, y=None): ...

    @abstractmethod
    def get_speedup_function_names(self):
        """Return a dictionnary containing the name of the speedup functions."""
        ...

    def _compute_distances(self, X, query):
        """
        Compute distances between X and a query based on channel_dependency mode.

        Parameters
        ----------
        X : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
            The dataset to compare against the query.
        query : np.ndarray, 2D array of shape (n_channels, n_timepoints)
            The query time series.

        Returns
        -------
        distances : np.ndarray
            Array of distances between the query and each case in X.
        """
        distances = np.zeros(self.n_cases_)

        for i in range(self.n_cases_):
            if self.channel_dependency == "independent":
                # Compute distance for each channel independently and sum them
                for c in range(self.n_channels_):
                    distances[i] += self.distance_function(
                        X[i][c], query[c], **self.distance_args
                    )
            elif self.channel_dependency == "dependent":
                # Compute distance considering all channels together
                distances[i] = self.distance_function(X[i], query, **self.distance_args)
            else:
                raise ValueError(
                    "Invalid value for channel_dependency. Choose 'independent' or 'dependent'."
                )

        if self.inverse_distance:
            distances = -distances

        return distances
