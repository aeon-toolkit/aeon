"""Base similiarity search for series."""

from abc import abstractmethod
from typing import final

import numpy as np

from aeon.base import BaseSeriesEstimator
from aeon.similarity_search._base import BaseSimilaritySearch


class BaseSeriesSimilaritySearch(BaseSeriesEstimator, BaseSimilaritySearch):
    """
    Base class for similarity search applications on single series.

    Such estimators include nearest neighbors on subsequences extracted from a series
    or motif discovery on single series.
    """

    _tags = {
        "input_data_type": "Series",
        "capability:multivariate": True,
    }

    @abstractmethod
    def __init__(self, axis=1):
        super().__init__(axis=axis)

    @final
    def fit(
        self,
        X: np.ndarray,
        y=None,
    ):
        """
        Fit method: data preprocessing and storage.

        Parameters
        ----------
        X : np.ndarray, 2D array of shape (n_channels, n_timepoints)
            Input series to be used for the similarity search operations.
        y : optional
            Not used.

        Raises
        ------
        TypeError
            If the input X array is not 2D raise an error.

        Returns
        -------
        self
        """
        self.reset()
        X = self._preprocess_series(X, self.axis, True)
        self.n_channels_ = self.metadata_["n_channels"]
        timepoint_idx = 1 if self.axis == 1 else 0
        self.n_timepoints_ = X.shape[timepoint_idx]
        self.X_ = X
        self._fit(X, y=y)
        self.is_fitted = True
        return self

    @abstractmethod
    def _fit(
        self,
        X: np.ndarray,
        y=None,
    ): ...

    @final
    def predict(self, X, **kwargs):
        """
        Predict function.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_tiempoints)
            Series to predict on.
        kwargs : dict, optional
            Additional keyword argument as dict or individual keywords args
            to pass to the estimator.

        Returns
        -------
        indexes : np.ndarray, shape = (k)
            Indexes of series in the that are similar to X.
        distances : np.ndarray, shape = (k)
            Distance of the matches to each series

        """
        self._check_is_fitted()
        X = self._preprocess_series(X, self.axis, False)
        self._check_predict_series_format(X)
        indexes, distances = self._predict(X, **kwargs)
        return indexes, distances

    @abstractmethod
    def _predict(self, X, **kwargs): ...

    def _check_predict_series_format(self, X):
        """
        Check wheter a series X is correctly formated regarding series given in fit.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_timepoints)
            A series to be used in predict.

        """
        channel_idx = 0 if self.axis == 1 else 1
        if self.n_channels_ != X.shape[channel_idx]:
            raise ValueError(
                f"Expected X to have {self.n_channels_} channels but"
                f" got {X.shape[channel_idx]} channels."
            )


class BaseSeriesNeighbors(BaseSeriesSimilaritySearch):
    """
    Base class for neighbor search estimators.

    The goal of this base class is to define a fit_predict method to, for example,
    compute self matrix profiles.
    """

    def _check_X_index(self, X_index: int):
        """
        Check wheter a X_index parameter is correctly formated and is admissible.

        Parameters
        ----------
        X_index : int
            Index of a timestamp in X_.

        """
        if X_index is not None:
            if not isinstance(X_index, int):
                raise TypeError("Expected an integer for X_index but got {X_index}")

            max_timepoints = self.n_timepoints_
            if hasattr(self, "length"):
                max_timepoints -= self.length
            if X_index >= max_timepoints or X_index < 0:
                raise ValueError(
                    "The value of X_index cannot exced the number "
                    "of timepoint in series given during fit. Expected a value "
                    f"between [0, {max_timepoints - 1}] but got {X_index}"
                )


class BaseSeriesMotifs(BaseSeriesSimilaritySearch):
    """
    Base class for motif search estimators.

    The goal of this base class is to define a fit_predict method to, for example,
    compute self matrix profiles.
    """

    def fit_predict(self, X, **kwargs):
        """
        Fit and predict on a single series X in order to compute self-motifs.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_tiempoints)
            Series to fit and predict on.
        kwargs : dict, optional
            Additional keyword argument as dict or individual keywords args
            to pass to the estimator during predict.

        Returns
        -------
        indexes : np.ndarray
            Indexes of series in the that are similar to X.
        distances : np.ndarray
            Distance of the matches to each series
        """
        self.fit(X)
        return self.predict(X, is_self_computation=True, **kwargs)
