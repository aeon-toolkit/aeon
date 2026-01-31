"""Base class for whole series similarity search."""

__maintainer__ = ["baraline"]
__all__ = [
    "BaseWholeSeriesSearch",
]

import numpy as np

from aeon.similarity_search._base import BaseSimilaritySearch


class BaseWholeSeriesSearch(BaseSimilaritySearch):
    """
    Base class for whole series similarity search.

    This class provides the foundation for algorithms that search for similar
    complete time series (comparing entire sequences of equal length) within a
    collection. Common applications include time series retrieval, classification
    via nearest neighbors, and clustering based on similarity.

    Interface
    ---------
    - **fit(X)**: Takes a 3D collection of shape ``(n_cases, n_channels, n_timepoints)``
      containing the time series database to search within.
    - **predict(X)**: Takes a 2D query series of shape ``(n_channels, n_timepoints)``
      where ``n_timepoints`` must match the series length in the fitted collection.
      Returns case indices of the nearest neighbor series.

    Notes
    -----
    Subclasses must implement ``_fit`` and ``_predict``. There is no requirement
    to use distance profiles - algorithms can use any approach (indexing, hashing,
    etc.) as long as they follow the interface.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.similarity_search.whole_series import BruteForce
    >>> X_fit = np.random.rand(10, 1, 100)  # 10 univariate series, 100 timepoints
    >>> query = np.random.rand(1, 100)  # query series of same length
    >>> searcher = BruteForce()
    >>> searcher.fit(X_fit)  # doctest: +SKIP
    >>> indexes, distances = searcher.predict(query, k=3)  # doctest: +SKIP
    >>> # indexes has shape (3,) with case indices of nearest neighbors
    """

    def __init__(self):
        super().__init__()

    def _check_query_length(self, X: np.ndarray):
        """
        Check that query has the expected length.

        Parameters
        ----------
        X : np.ndarray, shape=(n_channels, n_timepoints)
            Query series.

        Raises
        ------
        ValueError
            If query length doesn't match the fitted series length.
        """
        if X.shape[1] != self.n_timepoints_:
            raise ValueError(
                f"Expected X to have {self.n_timepoints_} timepoints but"
                f" got {X.shape[1]} timepoints."
            )
