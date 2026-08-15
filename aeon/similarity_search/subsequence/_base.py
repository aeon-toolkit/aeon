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
    Common applications include nearest-neighbor retrieval of subsequences and
    pattern matching within a collection of time series.

    Parameters
    ----------
    length : int
        The length of the subsequences to use for the search. The query provided
        to ``predict`` must have exactly this many timepoints.

    Notes
    -----
    The estimator follows the standard similarity search interface:

    - ``fit(X)`` takes a 3D collection of shape
      ``(n_cases, n_channels, n_timepoints)`` containing the time series database
      to search within.
    - ``predict(X)`` takes a 2D query subsequence of shape ``(n_channels, length)``
      where ``length`` is specified at initialization. It returns
      ``(i_case, i_timestamp)`` pairs indicating matches.

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

    def _validate_fit_params(self):
        """
        Validate the ``length`` parameter against the fitted series length.

        Called by ``fit`` after ``n_timepoints_`` is set. ``length`` must be an
        integer in ``[1, n_timepoints_]``.

        Raises
        ------
        ValueError
            If ``length`` is not an integer or is not in ``[1, n_timepoints_]``.
        """
        if (
            not isinstance(self.length, (int, np.integer))
            or isinstance(self.length, bool)
            or self.length < 1
            or self.length > self.n_timepoints_
        ):
            raise ValueError(
                "length must be an integer between 1 and n_timepoints_ "
                f"({self.n_timepoints_}), got length={self.length!r}."
            )

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
    NaiveSubsequenceSearch : Naive pairwise distance computation.
    """

    def _check_X_index(self, X_index):
        """
        Check that ``X_index`` is a valid ``(i_case, i_timepoint)`` location.

        Parameters
        ----------
        X_index : tuple of (int, int)
            The ``(i_case, i_timepoint)`` location of the query in the fitted
            collection. ``i_case`` must satisfy ``0 <= i_case < n_cases_`` and
            ``i_timepoint`` must satisfy ``0 <= i_timepoint <= n_timepoints_ - length``
            (the index of the last candidate subsequence).

        Raises
        ------
        TypeError
            If ``X_index`` is not a length-2 tuple of integers.
        ValueError
            If ``i_case`` or ``i_timepoint`` is out of bounds.
        """
        if not isinstance(X_index, tuple) or len(X_index) != 2:
            raise TypeError(
                "Expected X_index to be a tuple of two integers "
                f"(i_case, i_timepoint) but got {X_index!r}."
            )
        i_case, i_timepoint = X_index
        for name, value in (("i_case", i_case), ("i_timepoint", i_timepoint)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise TypeError(
                    f"Expected {name} in X_index to be an integer but got "
                    f"{value!r} in X_index={X_index!r}."
                )
        if i_case < 0 or i_case >= self.n_cases_:
            raise ValueError(
                f"X_index case {i_case} is out of bounds for collection "
                f"with {self.n_cases_} cases; expected 0 <= i_case < "
                f"{self.n_cases_}."
            )
        _max_timepoint = self.n_timepoints_ - self.length
        if i_timepoint < 0 or i_timepoint > _max_timepoint:
            raise ValueError(
                f"X_index timepoint {i_timepoint} is out of bounds; expected "
                f"0 <= i_timepoint <= {_max_timepoint} "
                f"(n_timepoints_ - length) in X_index={X_index!r}."
            )

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
        k : int or np.inf, default=1
            Number of neighbors to return. Must be a positive integer, or the
            sentinel ``np.inf`` to return all admissible candidate subsequences.
        dist_threshold : float, default=np.inf
            Maximum distance for a candidate to be returned as a match. Candidates
            with a (post-transformation) distance strictly above this threshold are
            discarded, so fewer than ``k`` matches may be returned.
        allow_trivial_matches : bool, default=False
            Whether to allow neighboring (overlapping) matches within the same
            series. Note the inverted semantics: when ``False``, an exclusion zone
            of ``int(length * exclusion_factor)`` on each side of every returned
            match is applied within the same series, preventing trivial overlapping
            matches; when ``True``, no exclusion zone is applied and the globally
            smallest distances are returned.
        exclusion_factor : float, default=0.5
            Multiplier of the query ``length`` used to size the exclusion zone
            (``int(length * exclusion_factor)``) applied on each side of a match
            when ``allow_trivial_matches`` is ``False``. Ignored when
            ``allow_trivial_matches`` is ``True``.
        inverse_distance : bool, default=False
            If ``True``, rank candidates by inverse distance so the farthest
            subsequences are returned instead of the nearest.
        X_index : tuple of (int, int) or None, default=None
            If the query ``X`` is itself a subsequence of the fitted collection,
            its ``(i_case, i_timepoint)`` location, so that it and its exclusion
            zone are excluded from its own neighbor search. ``i_case`` must be in
            ``[0, n_cases_ - 1]`` and ``i_timepoint`` in
            ``[0, n_timepoints_ - length]``. If ``None`` (default), no
            self-exclusion is applied.

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

        # Number of candidate positions per case, i.e. columns of dist_profiles.
        n_candidates = self.n_timepoints_ - self.length + 1

        if X_index is not None:
            self._check_X_index(X_index)
            i_case, i_timepoint = X_index
            # Inclusive upper bound so the query's own position (and the right edge
            # of the exclusion zone) are masked, matching the exclusion semantics of
            # ``_extract_top_k_from_dist_profile``.
            ub = min(i_timepoint + exclusion_size, n_candidates - 1)
            lb = max(0, i_timepoint - exclusion_size)
            dist_profiles[i_case, lb : ub + 1] = np.inf

        # Support k=np.inf ("return all matches") by clamping to the total number
        # of candidates before the njit top-k call, which allocates an array of
        # shape (k, 2). This mirrors the whole-series estimators' behavior.
        n_total_candidates = self.n_cases_ * n_candidates
        if k == np.inf:
            k = n_total_candidates
        k = min(k, n_total_candidates)

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
