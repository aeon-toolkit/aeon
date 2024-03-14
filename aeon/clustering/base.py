"""Base class for clustering."""

__maintainer__ = []
__all__ = ["BaseClusterer"]

import time
from abc import ABC, abstractmethod
from typing import final

import numpy as np

from aeon.base import BaseCollectionEstimator
from aeon.utils.validation._dependencies import _check_estimator_deps


class BaseClusterer(BaseCollectionEstimator, ABC):
    """Abstract base class for time series clusterers.

    Parameters
    ----------
    n_clusters : int, default=None
        Number of clusters for model.
    """

    def __init__(self, n_clusters: int = None):
        self.n_clusters = n_clusters
        # required for compatibility with some sklearn interfaces e.g.
        # CalibratedClassifierCV
        self._estimator_type = "clusterer"

        super().__init__()
        _check_estimator_deps(self)

    @final
    def fit(self, X, y=None) -> BaseCollectionEstimator:
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X : 3D np.ndarray (any number of channels, equal length series)
                of shape (n_cases, n_channels, n_timepoints)
            or 2D np.array (univariate, equal length series)
                of shape (n_cases, n_timepoints)
            or list of numpy arrays (any number of channels, unequal length series)
                of shape [n_cases], 2D np.array (n_channels, n_timepoints_i), where
                n_timepoints_i is length of series i
            other types are allowed and converted into one of the above.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        self:
            Fitted estimator.
        """
        self.reset()
        _start_time = int(round(time.time() * 1000))
        X = self._preprocess_collection(X)
        self._fit(X)
        self.fit_time_ = int(round(time.time() * 1000)) - _start_time
        self._is_fitted = True
        return self

    @final
    def predict(self, X, y=None) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. Other types are
            allowed and converted into one of the above.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.array
            shape ``(n_cases)`, index of the cluster each time series in X.
            belongs to.
        """
        self.check_is_fitted()
        X = self._preprocess_collection(X)
        return self._predict(X)

    def fit_predict(self, X, y=None) -> np.ndarray:
        """Compute cluster centers and predict cluster index for each time series.

        Convenience method; equivalent of calling fit(X) followed by predict(X)

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_cases, n_timepoints) or shape
            (n_cases, n_channels, n_timepoints)).
            Time series instances to train clusterer and then have indexes each belong
            to return.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_cases,))
            Index of the cluster each time series in X belongs to.
        """
        self.fit(X)
        return self.predict(X)

    @final
    def predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Default behaviour is to call _predict and set the predicted class probability
        to 1, other class probabilities to 0. Override if better estimates are
        obtainable.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. Other types are
            allowed and converted into one of the above.

        Returns
        -------
        y : 2D array of shape [n_cases, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        self.check_is_fitted()
        X = self._preprocess_collection(X)
        return self._predict_proba(X)

    def score(self, X, y=None) -> float:
        """Score the quality of the clusterer.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_cases, n_timepoints) or shape
            (n_cases, n_channels, n_timepoints)).
            Time series instances to train clusterer and then have indexes each belong
            to return.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        score : float
            Score of the clusterer.
        """
        self.check_is_fitted()
        X = self._preprocess_collection(X)
        return self._score(X, y)

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Default behaviour is to call _predict and set the predicted class probability
        to 1, other class probabilities to 0. Override if better estimates are
        obtainable.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. Other types are
            allowed and converted into one of the above.

        Returns
        -------
        y : 2D array of shape [n_cases, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        preds = self._predict(X)
        n_cases = len(preds)
        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = int(max(preds)) + 1
        dists = np.zeros((X.shape[0], n_clusters))
        for i in range(n_cases):
            dists[i, preds[i]] = 1
        return dists

    @abstractmethod
    def _score(self, X, y=None): ...

    @abstractmethod
    def _predict(self, X, y=None) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_cases, n_timepoints) or shape
            (n_cases,n_channels,n_timepoints)).
            Time series instances to predict their cluster indexes.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_cases,))
            Index of the cluster each time series in X belongs to.
        """
        ...

    @abstractmethod
    def _fit(self, X, y=None):
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_cases, n_timepoints) or shape
            (n_cases,n_channels,n_timepoints)).
            Training time series instances to cluster.

        Returns
        -------
        self:
            Fitted estimator.
        """
        ...
