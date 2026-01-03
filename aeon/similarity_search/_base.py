"""Base class for similarity search."""

__maintainer__ = ["baraline"]
__all__ = [
    "BaseSimilaritySearch",
]


from abc import abstractmethod

import numpy as np
from numba.typed import List

from aeon.base._base_collection import BaseCollectionEstimator


class BaseSimilaritySearch(BaseCollectionEstimator):
    """
    Base class for similarity search applications.

    This class provides the foundation for algorithms that search for similar
    time series or subsequences within a collection. All similarity search estimators
    follow a standard interface:

    - **fit(X)**: Takes a 3D collection of shape ``(n_cases, n_channels, n_timepoints)``
    - **predict(X)**: Takes a 2D query series of shape ``(n_channels, n_timepoints)``

    The module has two specialized subclasses:

    - ``BaseSubsequenceSearch``: For finding similar subsequences within time series
    - ``BaseWholeSeriesSearch``: For finding similar complete time series

    Notes
    -----
    Subclasses must implement ``_fit`` and ``_predict`` methods. The base class
    intentionally does not mandate any particular search algorithm (e.g., distance
    profiles, indexing, hashing) to allow maximum flexibility for different
    algorithmic approaches.

    See Also
    --------
    BaseSubsequenceSearch : Base class for subsequence similarity search.
    BaseWholeSeriesSearch : Base class for whole series similarity search.
    """

    _tags = {
        "requires_y": False,
        "fit_is_empty": False,
        "input_data_type": "Collection",
        "X_inner_type": ["numpy3D"],
    }

    def __init__(self):
        self.axis = 1
        super().__init__()

    def _preprocess_series(self, X, axis, store_metadata=False):
        """
        Preprocess a single input series for predict.

        Parameters
        ----------
        X : np.ndarray
            Input series, can be 1D or 2D array.
        axis : int
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point.
            ``axis==1`` indicates the time series are in rows.
        store_metadata : bool, default=False
            Ignored, exists for API consistency.

        Returns
        -------
        X : np.ndarray, shape = (n_channels, n_timepoints)
            Preprocessed 2D numpy array in axis=1 format.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(X)}")

        if X.ndim == 1:
            X = X[np.newaxis, :]
        elif X.ndim == 2:
            if axis == 0:
                X = X.T
        else:
            raise ValueError(f"Expected 1D or 2D array, got {X.ndim}D")

        return X

    def fit(
        self,
        X: np.ndarray | List,
        y=None,
    ):
        """
        Fit estimator to X.

        Parameters
        ----------
        X : np.ndarray shape (n_cases, n_channels, n_timepoints)
            Input data to store and use as database against the query
            given when calling predict.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        self : a fitted instance of the estimator
        """
        self.reset()
        X = self._preprocess_collection(X)
        self.n_channels_ = self.metadata_["n_channels"]
        self.n_cases_ = self.metadata_["n_cases"]
        # For equal length collections, store the number of timepoints.
        self.n_timepoints_ = (
            X.shape[2] if isinstance(X, np.ndarray) and X.ndim == 3 else None
        )
        self.X_ = X
        self._fit(X, y=y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, k: int = 1, axis: int = 1, **kwargs):
        """
        Find the k nearest neighbors to X in the fitted collection.

        Returns the indexes and distances of the best matches to X. It is possible that
        fewer than k indexes are returned if fewer than k admissible matches exist.

        Parameters
        ----------
        X : np.ndarray, 2D array of shape (n_channels, n_timepoints)
            Query series for which to find the nearest neighbors in the database.
            For subsequence search, ``n_timepoints`` should equal the ``length``
            parameter. For whole series search, ``n_timepoints`` should match the
            series length in the fitted collection.
        k : int, default=1
            Number of best matches to return.
        axis : int, default=1
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point, i.e. the
            shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels, n_timepoints)``.
        **kwargs : dict, optional
            Additional keyword arguments passed to the estimator's ``_predict`` method.

        Returns
        -------
        indexes : np.ndarray
            Indexes of the best matches. The shape and meaning depend on the
            search type:

            - **Subsequence search**: shape ``(n_matches, 2)`` with
              ``(i_case, i_timestamp)`` pairs indicating which series and at
              what position the match was found.
            - **Whole series search**: shape ``(n_matches,)`` with case indices
              indicating which series in the fitted collection are nearest
              neighbors.

        distances : np.ndarray, shape (n_matches,)
            Distances of the matches to the query. Lower values indicate better matches
            (unless ``inverse_distance=True`` is used).
        """
        self._check_is_fitted()
        X = self._preprocess_series(X, axis=axis, store_metadata=False)
        # Check that we have the same number of channel in the series and the fit data.
        self._check_predict_series_format(X)
        indexes, distances = self._predict(X, k, **kwargs)
        return indexes, distances

    @abstractmethod
    def _fit(
        self,
        X: np.ndarray | List,
        y=None,
    ):
        """
        Private fit method to be implemented by the estimators.

        Parameters
        ----------
        X : np.ndarray shape (n_cases, n_channels, n_timepoints)
            Input data to store and use as database against the query given when calling
            predict.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        self : a fitted instance of the estimator

        """
        ...

    @abstractmethod
    def _predict(self, X: np.ndarray, k: int, **kwargs):
        """
        Private predict method to be implemented by subclasses.

        Parameters
        ----------
        X : np.ndarray, 2D array of shape (n_channels, n_timepoints)
            Query series for which to find the nearest neighbors in the database.
        k : int
            Number of best matches to return.
        **kwargs : dict
            Additional keyword arguments specific to the estimator.

        Returns
        -------
        indexes : np.ndarray
            Indexes of the best matches (format depends on search type).
        distances : np.ndarray, shape (n_matches,)
            Distances of the matches to the query.
        """
        ...

    def _check_predict_series_format(self, X):
        """
        Check whether a series X in predict is correctly formatted.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_timepoints)
            A series to be used in predict.
        """
        if self.n_channels_ != X.shape[0]:
            raise ValueError(
                f"Expected X to have {self.n_channels_} channels but"
                f" got {X.shape[0]} channels (shape of X is {X.shape})."
            )
