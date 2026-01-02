"""Implementation of NN with brute force."""

__maintainer__ = ["baraline"]
__all__ = ["BruteForce"]

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from aeon.similarity_search.subsequence._base import BaseSubsequenceSearch
from aeon.similarity_search.subsequence._commons import (
    _inverse_distance_profile,
    extract_top_k_from_dist_profiles_2d,
)
from aeon.utils.numba.general import (
    get_all_subsequences,
    z_normalise_series_2d,
    z_normalise_series_3d,
)
from aeon.utils.validation import check_n_jobs


class BruteForce(BaseSubsequenceSearch):
    """
    Brute force subsequence nearest neighbor search.

    This estimator searches for the k nearest neighbor subsequences across a
    collection of time series using exhaustive pairwise distance computation.
    Given a query subsequence, it computes distance profiles against all series
    in the fitted collection and returns the best matches with their
    ``(case_index, timestamp)`` locations.

    Parameters
    ----------
    length : int
        The length of the subsequences to use for the search. The query provided
        to ``predict`` must have exactly this many timepoints.
    normalize : bool, default=False
        Whether the subsequences should be z-normalized before distance computation.
        This results in scale-independent matching.
    n_jobs : int, default=1
        Number of parallel threads to use for distance computation.

    Attributes
    ----------
    X_ : np.ndarray of shape (n_cases, n_channels, n_timepoints)
        The fitted collection of time series.
    X_subs_ : list of np.ndarray
        Precomputed subsequences for each series in the collection.
    n_cases_ : int
        Number of time series in the fitted collection.
    n_channels_ : int
        Number of channels in the fitted time series.
    n_timepoints_ : int
        Number of timepoints in each fitted time series.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.similarity_search.subsequence import BruteForce
    >>> X_fit = np.random.rand(5, 1, 100)
    >>> query = np.random.rand(1, 20)
    >>> searcher = BruteForce(length=20, normalize=False)
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
        length: int,
        normalize: bool | None = False,
        n_jobs: int | None = 1,
    ):
        self.normalize = normalize
        self.n_jobs = n_jobs
        super().__init__(length)

    def _fit(
        self,
        X: np.ndarray,
        y=None,
    ):
        """
        Fit the BruteForce estimator on a collection of time series.

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

        # Extract subsequences from each series in the collection
        n_cases = X.shape[0]
        self.X_subs_ = []
        for i in range(n_cases):
            subs = get_all_subsequences(X[i], self.length, 1)
            if self.normalize:
                subs = z_normalise_series_3d(subs)
            self.X_subs_.append(subs)

        set_num_threads(prev_threads)
        return self

    def _predict(
        self,
        X: np.ndarray,
        k: int = 1,
        dist_threshold: float = np.inf,
        allow_trivial_matches: bool = False,
        exclusion_factor: float = 0.5,
        inverse_distance: bool = False,
        X_index: tuple = None,
    ):
        """
        Find nearest neighbor subsequences to X in the fitted collection.

        Parameters
        ----------
        X : np.ndarray, shape=(n_channels, length)
            Query subsequence.
        k : int, default=1
            Number of neighbors to return.
        dist_threshold : float, default=np.inf
            Maximum distance threshold for matches.
        allow_trivial_matches : bool, default=False
            Whether to allow neighboring matches within the same series.
        exclusion_factor : float, default=0.5
            Factor of query length for exclusion zone size.
        inverse_distance : bool, default=False
            If True, return farthest neighbors instead.
        X_index : tuple (i_case, i_timepoint), optional
            If X is from the fitted collection, specify its location.

        Returns
        -------
        indexes : np.ndarray, shape=(n_matches, 2)
            The (i_case, i_timepoint) indexes of the best matches.
        distances : np.ndarray, shape=(n_matches,)
            The distances of the best matches.
        """
        self._check_query_length(X)

        dist_profiles = self.compute_distance_profile(X)

        if inverse_distance:
            for i in range(len(dist_profiles)):
                dist_profiles[i] = _inverse_distance_profile(dist_profiles[i])

        exclusion_size = int(self.length * exclusion_factor)

        if X_index is not None:
            i_case, i_timepoint = X_index
            if i_case < 0 or i_case >= self.n_cases_:
                raise ValueError(
                    f"X_index case {i_case} is out of bounds for collection "
                    f"with {self.n_cases_} cases."
                )
            _max_timestamp = self.n_timepoints_ - self.length
            ub = min(i_timepoint + exclusion_size, _max_timestamp)
            lb = max(0, i_timepoint - exclusion_size)
            dist_profiles[i_case, lb:ub] = np.inf

        return extract_top_k_from_dist_profiles_2d(
            dist_profiles,
            k,
            dist_threshold,
            allow_trivial_matches,
            exclusion_size,
        )

    def compute_distance_profile(self, X: np.ndarray):
        """
        Compute the distance profile of X to all subsequences in X_.

        Parameters
        ----------
        X : np.ndarray, 2D array of shape (n_channels, length)
            The query to use to compute the distance profiles.

        Returns
        -------
        distance_profiles : np.ndarray, 2D array of shape (n_cases, n_candidates)
            The distance profile of X to all subsequences in all series of X_.
            The ``n_candidates`` value is equal to ``n_timepoints - length + 1``.
        """
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)

        if self.normalize:
            X = z_normalise_series_2d(X)

        n_cases = len(self.X_subs_)
        n_candidates = self.n_timepoints_ - self.length + 1
        distance_profiles = np.zeros((n_cases, n_candidates))

        for i in range(n_cases):
            distance_profiles[i] = _naive_squared_distance_profile(self.X_subs_[i], X)

        set_num_threads(prev_threads)
        return distance_profiles

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
        """
        if parameter_set == "default":
            params = {"length": 20}
        else:
            raise NotImplementedError(
                f"The parameter set {parameter_set} is not yet implemented"
            )
        return params


@njit(cache=True, fastmath=True, parallel=True)
def _naive_squared_distance_profile(
    X_subs,
    Q,
):
    """
    Compute a squared euclidean distance profile.

    Parameters
    ----------
    X_subs : array, shape=(n_subsequences, n_channels, length)
        Subsequences of size length of the input time series to search in.
    Q : array, shape=(n_channels, query_length)
        Query used during the search.

    Returns
    -------
    out : np.ndarray, 1D array of shape (n_subsequences,)
        The distance between the query and all candidates in X.
    """
    n_subs, n_channels, length = X_subs.shape
    dist_profile = np.zeros(n_subs)
    for i in prange(n_subs):
        for j in range(n_channels):
            for k in range(length):
                dist_profile[i] += (X_subs[i, j, k] - Q[j, k]) ** 2
    return dist_profile
