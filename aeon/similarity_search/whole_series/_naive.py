"""Implementation of whole series NN with naive pairwise search."""

__maintainer__ = ["baraline"]
__all__ = ["NaiveSeriesSearch"]

from collections.abc import Callable

import numpy as np

from aeon.distances import pairwise_distance
from aeon.similarity_search.subsequence._commons import (
    _extract_top_k_from_dist_profile,
)
from aeon.similarity_search.whole_series._base import BaseWholeSeriesSearch
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
    using exhaustive pairwise distance computation. All series must have the same
    length.

    Parameters
    ----------
    normalize : bool, default=False
        Whether the series should be z-normalized before distance computation.
        This results in scale-independent matching, useful when you want to find
        similar shapes regardless of their amplitude.
    distance : str or callable, default="squared"
        Distance measure between series. A list of valid strings can be found
        in the documentation for :func:`aeon.distances.get_distance_function` or
        through calling :func:`aeon.distances.get_distance_function_names`. If a
        callable is passed it must be a function that takes two 2d numpy arrays of
        shape ``(n_channels, n_timepoints)`` as input and returns a float.
    distance_params : dict, default=None
        Dictionary of distance parameters for the case that ``distance`` is a str.
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
        distance: str | Callable = "squared",
        distance_params: dict | None = None,
        n_jobs: int = 1,
    ):
        self.normalize = normalize
        self.distance = distance
        self.distance_params = distance_params
        self.n_jobs = n_jobs

        self._distance_params = distance_params
        if self._distance_params is None:
            self._distance_params = {}

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
            # thread management is needed here (parallelism is handled by
            # ``pairwise_distance`` in ``compute_distance_profile``).
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
            Distance from X to each series in the fitted collection, according to
            the ``distance`` parameter.
        """
        if self.normalize:
            X = z_normalise_series_2d(X)

        # The query is passed as a (1, n_channels, n_timepoints) collection: a 2D
        # y is interpreted as a collection of univariate series, which would be
        # wrong for multivariate queries.
        return pairwise_distance(
            self.X_,
            X[np.newaxis],
            method=self.distance,
            n_jobs=self._n_jobs,
            **self._distance_params,
        ).reshape(-1)

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
