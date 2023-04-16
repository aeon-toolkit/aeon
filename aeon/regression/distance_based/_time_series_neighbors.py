# -*- coding: utf-8 -*-
"""KNN time series regression.

This class is a KNN regressor which supports time series distance measures.
The class has hardcoded string references to numba based distances in aeon.distances.
It can also be used with callables, or aeon (pairwise transformer) estimators.
"""

__author__ = ["TonyBagnall", "GuiArcencio"]
__all__ = ["KNeighborsTimeSeriesRegressor"]

import numpy as np

from aeon.distances import distance_from_single_to_multiple
from aeon.regression.base import BaseRegressor

WEIGHTS_SUPPORTED = ["uniform", "distance"]


class KNeighborsTimeSeriesRegressor(BaseRegressor):
    """KNN Time Series Regressor.

    An adapted K-Neighbors Regressor for time series data.

    This class is a KNN regressor which supports time series distance measures.
    It has hardcoded string references to numba based distances in aeon.distances,
    and can also be used with callables, or aeon (pairwise transformer) estimators.

    Parameters
    ----------
    n_neighbors : int, set k for knn (default =1)
    weights : string or callable function, optional. default = 'uniform'
        mechanism for weighting a vot
        one of: 'uniform', 'distance', or a callable function
    distance : str or callable, optional. default ='dtw'
        distance measure between time series
        if str, must be one of the following strings:
            'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw',
            'lcss', 'edr', 'erp', 'msm', 'twe'
        this will substitute a hard-coded distance metric from aeon.distances
        When mpdist is used, the subsequence length (parameter m) must be set
            Example: knn_mpdist = KNeighborsTimeSeriesClassifier(
                                metric='mpdist', metric_params={'m':30})
        if callable, must be of signature (X: np.ndarray, X2: np.ndarray) -> np.ndarray
            output must be mxn array if X is array of m Series, X2 of n Series
        can be pairwise panel transformer inheriting from BasePairwiseTransformerPanel
    distance_params : dict, optional. default = None.
        dictionary for metric parameters , in case that distance is a str

    Examples
    --------
    >>> from aeon.datasets import load_unit_test
    >>> from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
    >>> X_train, y_train = load_unit_test(return_X_y=True, split="train")
    >>> X_test, y_test = load_unit_test(return_X_y=True, split="test")
    >>> regressor = KNeighborsTimeSeriesRegressor(distance="euclidean")
    >>> regressor.fit(X_train, y_train)
    KNeighborsTimeSeriesRegressor(...)
    >>> y_pred = regressor.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "X_inner_mtype": ["numpy3D"],
    }

    def __init__(
        self,
        distance="dtw",
        distance_params=None,
        n_neighbors=1,
        weights="uniform",
    ):
        self.distance = distance
        self.distance_params = distance_params
        self.n_neighbors = n_neighbors

        if weights not in WEIGHTS_SUPPORTED:
            raise ValueError(
                f"Unrecognised kNN weights: {weights}. "
                f"Allowed values are: {WEIGHTS_SUPPORTED}. "
            )
        self.weights = weights
        self._distance_params = distance_params
        if distance_params is None:
            self._distance_params = {}

        super(KNeighborsTimeSeriesRegressor, self).__init__()

    def _fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : aeon-compatible Panel data format, with n_samples series
        y : {array-like, sparse matrix}
            Target values of shape = [n_samples]
        """
        self.X_ = X
        self.y_ = y
        return self

    def _predict(self, X):
        """Predict the target values for the provided data.

        Parameters
        ----------
        X : aeon-compatible Panel data format, with n_samples series

        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Target values for each data sample.
        """
        self.check_is_fitted()

        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            idx, weights = self._kneighbors(X[i])
            preds[i] = np.average(self.y_[idx], weights=weights)

        return preds

    def _kneighbors(self, X):
        """Find the K-neighbors of a point.

        Returns indices and weights of each point.

        Parameters
        ----------
        X : aeon-compatible data format, Panel or Series, with n_samples series

        Returns
        -------
        ind : array
            Indices of the nearest points in the population matrix.
        ws : array
            Array representing the weights of each neighbor.
        """
        distances = distance_from_single_to_multiple(
            X, self.X_, self.distance, **self._distance_params
        )
        # Find indices of k nearest neighbors using partitioning:
        # [0..k-1], [k], [k+1..n-1]
        # They might not be ordered within themselves,
        # but it is not necessary and partitioning is
        # O(n) while sorting is O(nlogn)
        closest_idx = np.argpartition(distances, self.n_neighbors)
        closest_idx = closest_idx[: self.n_neighbors]

        if self.weights == "distance":
            ws = distances[closest_idx]
            ws = ws**2

            # Using epsilon ~= 0 to avoid division by zero
            ws = 1 / (ws + np.finfo(float).eps)
        elif self.weights == "uniform":
            ws = np.repeat(1.0, self.n_neighbors)
        else:
            raise Exception(f"Invalid kNN weights: {self.weights}")

        return closest_idx, ws
