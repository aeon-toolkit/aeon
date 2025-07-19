"""Base class for similarity search."""

__maintainer__ = ["baraline"]
__all__ = [
    "BaseSimilaritySearch",
]


from abc import abstractmethod
from typing import Union

import numpy as np
from numba.typed import List

from aeon.base._base import BaseAeonEstimator


class BaseSimilaritySearch(BaseAeonEstimator):
    """Base class for similarity search applications."""

    _tags = {
        "requires_y": False,
        "fit_is_empty": False,
        "input_data_type": "Collection",
        "X_inner_type": ["numpy3D"],
    }

    def __init__(self):
        self.axis = 1
        super().__init__()

    def fit(
        self,
        X: Union[np.ndarray, List],
        y=None,
    ):
        """
        Fit estimator to X.

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
        self.reset()
        X = self._preprocess_collection(X)
        self.n_channels_ = self.metadata_["n_channels"]
        self.n_cases_ = self.metadata_["n_cases"]
        self._fit(X, y=y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, k: int = 1, axis: int = 1, **kwargs):
        """
        Predict function.

        Returns the indexes and distances of the best matches to X. It is possible that
        less than k indexes are returned in the case where less than k admissible
        matches exist. An admissible matches is context dependent, it can represent
        the number of series (n_cases) given during fit for whole series search, or
        the number of possible subsequences of size L for subsequences search.

        If you set the length parameter at init, X will need to have ``n_timepoints
        equal == length``.

        Parameters
        ----------
        X : np.ndarray, 2D array of shape = (n_channels, n_timepoints)
            Series for which to find the nearest neighbors in the database.
        k : int, optional
            Number of best matches to return.
        axis: int
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point. i.e. the
            shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels, n_timepoints)``.
        kwargs : dict, optional
            Additional keyword arguments to be passed to the _predict function of the
            estimator.

        Returns
        -------
        indexes : np.ndarray, shape = (k, 2)
            Indexes (i_case, i_timestep) of series or subsequences that are similar to X
            . It is possible that less than k indexes are returned in the case where
            less than k admissible matches exist.
        distances : np.ndarray, shape = (k)
            Distance of the matches to each series. It is possible that less than k
            indexes are returned in the case where less than k admissible matches exist.


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
        X: Union[np.ndarray, List],
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
        Private predict method to be implemented by the estimators.

        Parameters
        ----------
        X : np.ndarray, 2D array of shape = (n_channels, n_timepoints)
            Series for which to find the nearest neighbors in the database.
        k : int, optional
            Number of best matches to return.
        kwargs : dict, optional
            Additional keyword arguments to be passed to the _predict function of the
            estimator.

        Returns
        -------
        indexes : np.ndarray, shape = (k, 2)
            Indexes (i_case, i_timestep) of series or subsequences that are similar to X
            . It is possible that less than k indexes are returned in the case where
            less than k admissible matches exist.
        distances : np.ndarray, shape = (k)
            Distance of the matches to each series. It is possible that less than k
            indexes are returned in the case where less than k admissible matches exist.

        """
        ...

    def _check_predict_series_format(self, X):
        """
        Check whether a series X in predict is correctly formated.

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
