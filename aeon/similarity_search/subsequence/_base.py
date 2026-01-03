"""Base class for subsequence similarity search."""

__maintainer__ = ["baraline"]
__all__ = [
    "BaseSubsequenceSearch",
    "BaseDistanceProfileSearch",
]

from abc import abstractmethod

import numpy as np

from aeon.similarity_search._base import BaseSimilaritySearch
from aeon.similarity_search.subsequence._commons import (
    _extract_top_k_from_dist_profile,
)
from aeon.utils.numba.general import AEON_NUMBA_STD_THRESHOLD


class BaseSubsequenceSearch(BaseSimilaritySearch):
    """
    Base class for subsequence-based similarity search.

    This class provides the foundation for algorithms that search for similar
    subsequences (patterns of fixed length) within a collection of time series.
    Common applications include finding repeated patterns, motif discovery, and
    anomaly detection based on pattern matching.

    Parameters
    ----------
    length : int
        The length of the subsequences to use for the search. The query provided
        to ``predict`` must have exactly this many timepoints.

    Interface
    ---------
    - **fit(X)**: Takes a 3D collection of shape ``(n_cases, n_channels, n_timepoints)``
      containing the time series database to search within.
    - **predict(X)**: Takes a 2D query subsequence of shape ``(n_channels, length)``
      where ``length`` is specified at initialization. Returns ``(i_case, i_timestamp)``
      pairs indicating matches.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.similarity_search.subsequence import MASS
    >>> X_fit = np.random.rand(10, 1, 100)  # 10 univariate series, 100 timepoints
    >>> query = np.random.rand(1, 20)  # query subsequence of length 20
    >>> searcher = MASS(length=20)
    >>> searcher.fit(X_fit)  # doctest: +SKIP
    >>> indexes, distances = searcher.predict(query, k=3)  # doctest: +SKIP
    >>> # indexes has shape (3, 2) with (case_index, timestamp) pairs
    """

    def __init__(self, length: int):
        self.length = length
        super().__init__()

    def _check_query_length(self, X: np.ndarray):
        """
        Check that query has the expected length.

        Parameters
        ----------
        X : np.ndarray, shape=(n_channels, n_timepoints)
            Query subsequence.

        Raises
        ------
        ValueError
            If query length doesn't match the expected length.
        """
        if X.shape[1] != self.length:
            raise ValueError(
                f"Expected X to have {self.length} timepoints but"
                f" got {X.shape[1]} timepoints."
            )


class BaseDistanceProfileSearch(BaseSubsequenceSearch):
    """
    Base class for distance-profile-based subsequence search.

    This class provides shared logic for search algorithms that compute
    full distance profiles (e.g., MASS, brute force). Algorithms that use
    other approaches should inherit directly from ``BaseSubsequenceSearch``
    instead or create their own base class.

    Subclasses must implement ``compute_distance_profile`` which computes
    distances from a query to all candidate subsequences in the fitted
    collection.

    Parameters
    ----------
    length : int
        The length of the subsequences to use for the search.

    See Also
    --------
    BaseSubsequenceSearch : Parent class for all subsequence search methods.
    MASS : FFT-based distance profile computation.
    BruteForce : Naive pairwise distance computation.
    """

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
                dist_profiles[i] = 1.0 / (dist_profiles[i] + AEON_NUMBA_STD_THRESHOLD)

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

        return _extract_top_k_from_dist_profile(
            dist_profiles,
            k,
            dist_threshold,
            allow_trivial_matches,
            exclusion_size,
        )

    @abstractmethod
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
        ...
