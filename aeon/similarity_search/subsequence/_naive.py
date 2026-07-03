"""Implementation of NN with naive pairwise subsequence search."""

__maintainer__ = ["baraline"]
__all__ = ["NaiveSubsequenceSearch"]

import numpy as np

from aeon.similarity_search._commons import _pairwise_squared_distance
from aeon.similarity_search.subsequence._base import BaseDistanceProfileSearch
from aeon.utils.decorators.numba_threading import numba_thread_handler
from aeon.utils.numba.general import (
    get_all_subsequences,
    z_normalise_series_2d,
    z_normalise_series_3d,
)
from aeon.utils.validation import check_n_jobs


class NaiveSubsequenceSearch(BaseDistanceProfileSearch):
    """
    Naive subsequence nearest neighbor search.

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
    X_subs_ : np.ndarray of shape (n_cases, n_candidates, n_channels, length)
        Precomputed subsequences for each series in the collection, where
        ``n_candidates`` equals ``n_timepoints - length + 1``.
    n_cases_ : int
        Number of time series in the fitted collection.
    n_channels_ : int
        Number of channels in the fitted time series.
    n_timepoints_ : int
        Number of timepoints in each fitted time series.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.similarity_search.subsequence import NaiveSubsequenceSearch
    >>> X_fit = np.random.rand(5, 1, 100)
    >>> query = np.random.rand(1, 20)
    >>> searcher = NaiveSubsequenceSearch(length=20, normalize=False)
    >>> searcher.fit(X_fit)
    NaiveSubsequenceSearch(length=20)
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
        Fit the NaiveSubsequenceSearch estimator on a collection of time series.

        Parameters
        ----------
        X : np.ndarray, shape=(n_cases, n_channels, n_timepoints)
            Collection of time series to search within.
        y : ignored

        Returns
        -------
        self
        """
        self._n_jobs = check_n_jobs(self.n_jobs)

        # Extract subsequences from each series in the collection. Since the
        # collection is equal-length, every case yields the same number of
        # candidate subsequences, so they can be stacked into a single 4D array
        # (n_cases, n_candidates, n_channels, length) and processed by one
        # parallel region in ``compute_distance_profile``.
        n_cases = X.shape[0]
        subs_list = []
        for i in range(n_cases):
            subs = get_all_subsequences(X[i], self.length, 1)
            if self.normalize:
                subs = z_normalise_series_3d(subs)
            subs_list.append(subs)
        # Stack into (n_cases, n_candidates, n_channels, length).
        self.X_subs_ = np.asarray(subs_list)

        return self

    @numba_thread_handler
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
        if self.normalize:
            X = z_normalise_series_2d(X)

        n_cases, n_candidates, n_channels, length = self.X_subs_.shape

        # Flatten the (n_cases, n_candidates) subsequences into a single
        # (n_cases * n_candidates, n_channels, length) array so the shared
        # parallel kernel runs a single parallel region over all candidates
        # instead of one region per case.
        flat_subs = self.X_subs_.reshape(n_cases * n_candidates, n_channels, length)
        distance_profiles = _pairwise_squared_distance(flat_subs, X)
        return distance_profiles.reshape(n_cases, n_candidates)

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
