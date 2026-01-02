"""Implementation of whole series NN with brute force."""

__maintainer__ = ["baraline"]
__all__ = ["BruteForce"]

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from aeon.similarity_search.whole_series._base import BaseWholeSeriesSearch
from aeon.similarity_search.whole_series._commons import (
    _extract_top_k_from_dist_profile,
    _inverse_distance_profile,
)
from aeon.utils.numba.general import z_normalise_series_2d, z_normalise_series_3d
from aeon.utils.validation import check_n_jobs


class BruteForce(BaseWholeSeriesSearch):
    """
    Brute force whole series nearest neighbor search.

    This estimator finds nearest neighbors among complete time series in a collection
    using exhaustive pairwise squared Euclidean distance computation. All series must
    have the same length.

    Parameters
    ----------
    normalize : bool, default=False
        Whether the series should be z-normalized before distance computation.
        This results in scale-independent matching, useful when you want to find
        similar shapes regardless of their amplitude.
    n_jobs : int, default=1
        Number of parallel threads to use for distance computation.

    Attributes
    ----------
    X_ : np.ndarray of shape (n_cases, n_channels, n_timepoints)
        The fitted collection of time series.
    X_normalized_ : np.ndarray
        The normalized collection (if ``normalize=True``).
    n_cases_ : int
        Number of time series in the fitted collection.
    n_channels_ : int
        Number of channels in the fitted time series.
    n_timepoints_ : int
        Number of timepoints in each fitted time series.


    Examples
    --------
    >>> import numpy as np
    >>> from aeon.similarity_search.whole_series import BruteForce
    >>> X_fit = np.random.rand(10, 1, 50)
    >>> query = np.random.rand(1, 50)
    >>> searcher = BruteForce(normalize=True)
    >>> searcher.fit(X_fit)
    >>> indexes, distances = searcher.predict(query, k=3)
    """

    _tags = {
        "capability:unequal_length": False,
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        normalize: bool = False,
        n_jobs: int = 1,
    ):
        self.normalize = normalize
        self.n_jobs = n_jobs
        super().__init__()

    def _fit(
        self,
        X: np.ndarray,
        y=None,
    ):
        """
        Store the collection of series for later search.

        Parameters
        ----------
        X : np.ndarray, shape=(n_cases, n_channels, n_timepoints)
            Collection of time series to search within.
        y : ignored

        Returns
        -------
        self
        """
        prev_threads = get_num_threads()
        self._n_jobs = check_n_jobs(self.n_jobs)
        set_num_threads(self._n_jobs)

        self.n_timepoints_ = X.shape[2]
        if self.normalize:
            self.X_normalized_ = z_normalise_series_3d(X)
        else:
            self.X_normalized_ = X

        set_num_threads(prev_threads)
        return self

    def _predict(
        self,
        X: np.ndarray,
        k: int = 1,
        dist_threshold: float = np.inf,
        inverse_distance: bool = False,
        X_index: int = None,
    ):
        """
        Find nearest neighbor series to X in the fitted collection.

        Parameters
        ----------
        X : np.ndarray, shape=(n_channels, n_timepoints)
            Query series.
        k : int, default=1
            Number of neighbors to return.
        dist_threshold : float, default=np.inf
            Maximum distance threshold for matches.
        inverse_distance : bool, default=False
            If True, return farthest neighbors instead.
        X_index : int, optional
            If X is from the fitted collection, specify its index to exclude.

        Returns
        -------
        indexes : np.ndarray, shape=(n_matches,)
            Indexes of the nearest neighbor series.
        distances : np.ndarray, shape=(n_matches,)
            Distances to the nearest neighbors.
        """
        self._check_query_length(X)

        dist_profile = self.compute_distance_profile(X)

        if inverse_distance:
            dist_profile = _inverse_distance_profile(dist_profile)

        if X_index is not None:
            if X_index < 0 or X_index >= self.n_cases_:
                raise ValueError(
                    f"X_index must be between 0 and {self.n_cases_ - 1}, "
                    f"got {X_index}"
                )
            dist_profile[X_index] = np.inf

        if k == np.inf:
            k = len(dist_profile)
        k = min(k, len(dist_profile))

        return _extract_top_k_from_dist_profile(
            dist_profile,
            k,
            dist_threshold,
            allow_trivial_matches=True,
            exclusion_size=0,
        )

    def compute_distance_profile(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the distance profile of X to all series in the fitted collection.

        Parameters
        ----------
        X : np.ndarray, shape=(n_channels, n_timepoints)
            Query series.

        Returns
        -------
        distance_profile : np.ndarray, shape=(n_cases,)
            Squared Euclidean distance from X to each series in the fitted collection.
        """
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)

        if self.normalize:
            X = z_normalise_series_2d(X)

        distance_profile = _pairwise_squared_distance(self.X_normalized_, X)

        set_num_threads(prev_threads)
        return distance_profile

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the estimator."""
        if parameter_set == "default":
            params = {}
        else:
            raise NotImplementedError(
                f"The parameter set {parameter_set} is not yet implemented"
            )
        return params


@njit(cache=True, fastmath=True, parallel=True)
def _pairwise_squared_distance(X_collection, Q):
    """
    Compute squared Euclidean distance between Q and each series in X_collection.

    Parameters
    ----------
    X_collection : np.ndarray, shape=(n_cases, n_channels, n_timepoints)
        Collection of time series.
    Q : np.ndarray, shape=(n_channels, n_timepoints)
        Query series.

    Returns
    -------
    distances : np.ndarray, shape=(n_cases,)
        Squared Euclidean distance from Q to each series in X_collection.
    """
    n_cases, n_channels, n_timepoints = X_collection.shape
    distances = np.zeros(n_cases)
    for i in prange(n_cases):
        for j in range(n_channels):
            for k in range(n_timepoints):
                distances[i] += (X_collection[i, j, k] - Q[j, k]) ** 2
    return distances
