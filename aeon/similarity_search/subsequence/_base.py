"""Base class for subsequence similarity search."""

__maintainer__ = ["baraline"]
__all__ = [
    "BaseSubsequenceSearch",
]

import numpy as np

from aeon.similarity_search._base import BaseSimilaritySearch


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
