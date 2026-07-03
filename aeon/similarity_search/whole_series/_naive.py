"""Implementation of whole series NN with naive pairwise search."""

__maintainer__ = ["baraline"]
__all__ = ["NaiveSeriesSearch"]

import numpy as np

from aeon.similarity_search._commons import _pairwise_squared_distance
from aeon.similarity_search.subsequence._commons import (
    _extract_top_k_from_dist_profile,
)
from aeon.similarity_search.whole_series._base import BaseWholeSeriesSearch
from aeon.utils.decorators.numba_threading import numba_thread_handler
from aeon.utils.numba.general import (
    AEON_NUMBA_STD_THRESHOLD,
    z_normalise_series_2d,
    z_normalise_series_3d,
)
from aeon.utils.validation import check_n_jobs


class NaiveSeriesSearch(BaseWholeSeriesSearch):
    """
    Naive whole series nearest neighbor search.

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
        The collection used for distance computation: the z-normalized collection
        when ``normalize=True``, otherwise the raw fitted collection (as stored by
        the base class).
    n_cases_ : int
        Number of time series in the fitted collection.
    n_channels_ : int
        Number of channels in the fitted time series.
    n_timepoints_ : int
        Number of timepoints in each fitted time series.

    Notes
    -----
    In addition to ``k`` and ``axis``, ``predict`` accepts the following search
    options as keyword arguments:

    - ``dist_threshold`` : float, default=``np.inf``
        Maximum (post-transformation) distance for a series to be returned as a
        match. Series farther than this are discarded, so fewer than ``k`` matches
        may be returned.
    - ``inverse_distance`` : bool, default=False
        If True, rank by inverse distance so the farthest series are returned
        instead of the nearest.
    - ``X_index`` : int or None, default=None
        If the query ``X`` is itself a member of the fitted collection, its case
        index, so that it is excluded from its own neighbor search. Must be in
        ``[0, n_cases_ - 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.similarity_search.whole_series import NaiveSeriesSearch
    >>> X_fit = np.random.rand(10, 1, 50)
    >>> query = np.random.rand(1, 50)
    >>> searcher = NaiveSeriesSearch(normalize=True)
    >>> searcher.fit(X_fit)
    NaiveSeriesSearch(normalize=True)
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
        self._n_jobs = check_n_jobs(self.n_jobs)
        if self.normalize:
            # Replace the raw collection (``self.X_``, set by the base ``fit``) with
            # its z-normalized version, which is what search reads: this avoids
            # holding both copies. ``z_normalise_series_3d`` is serial numba, so no
            # thread management is needed here (the parallel path is
            # ``compute_distance_profile``, wrapped by ``@numba_thread_handler``).
            self.X_ = z_normalise_series_3d(X)
        # normalize=False: keep the raw collection stored by the base ``fit`` as-is.
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
            Maximum (post-transformation) distance for a series to be returned as
            a match. Series farther than this are discarded, so fewer than ``k``
            matches may be returned.
        inverse_distance : bool, default=False
            If True, rank by inverse distance so the farthest series are returned
            instead of the nearest.
        X_index : int or None, default=None
            If the query ``X`` is itself a member of the fitted collection, its case
            index, so that it is excluded from its own neighbor search. Must be in
            ``[0, n_cases_ - 1]``.

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
            dist_profile = 1.0 / (dist_profile + AEON_NUMBA_STD_THRESHOLD)

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

        # Reshape 1D profile to 2D (n_cases, 1) for the shared function
        dist_profile_2d = dist_profile.reshape(-1, 1)
        indexes_2d, distances = _extract_top_k_from_dist_profile(
            dist_profile_2d,
            k,
            dist_threshold,
            allow_trivial_matches=True,
            exclusion_size=0,
        )
        # Extract case indexes (column 0) from 2D result
        return indexes_2d[:, 0], distances

    @numba_thread_handler
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
        # ``@numba_thread_handler`` reads ``self.n_jobs``, applies ``check_n_jobs`` and
        # restores the thread count in a try/finally (exception-safe) around the
        # parallel ``_pairwise_squared_distance`` kernel.
        if self.normalize:
            X = z_normalise_series_2d(X)

        return _pairwise_squared_distance(self.X_, X)

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
