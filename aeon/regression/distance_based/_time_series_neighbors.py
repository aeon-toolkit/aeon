"""KNN time series regression.

This class is a KNN regressor which supports time series distance measures.
The class has hardcoded string references to numba based distances in aeon.distances.
It can also be used with callables, or aeon (pairwise transformer) estimators.
"""

__maintainer__ = []
__all__ = ["KNeighborsTimeSeriesRegressor"]

import numpy as np

from aeon.distances import get_distance_function
from aeon.regression.base import BaseRegressor

WEIGHTS_SUPPORTED = ["uniform", "distance"]


class KNeighborsTimeSeriesRegressor(BaseRegressor):
    """
    K-Nearest Neighbour Time Series Regressor.

    An adapted K-Neighbors Regressor for time series data.

    This class is a KNN regressor which supports time series distance measures.
    It has hardcoded string references to numba based distances in aeon.distances,
    and can also be used with callables, or aeon (pairwise transformer) estimators.

    Parameters
    ----------
    n_neighbors : int, default =1
        Set k for knn.
    weights : str or callable function, default = 'uniform'
        Mechanism for weighting a vote.
        one of: 'uniform', 'distance', or a callable function.
    distance : str or callable, default ='dtw'
        Distance measure between time series
        if str, must be one of the following strings:
            'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw',
            'lcss', 'edr', 'erp', 'msm', 'twe'
        this will substitute a hard-coded distance metric from aeon.distances
        When mpdist is used, the subsequence length (parameter m) must be set
            Example: knn_mpdist = KNeighborsTimeSeriesClassifier(
                                metric='mpdist', metric_params={'m':30})
        if callable, must be of signature (X: np.ndarray, X2: np.ndarray) -> np.ndarray
            output must be mxn array if X is array of m Series, X2 of n Series.
    distance_params : dict, default = None
        Dictionary for metric parameters , in case that distance is a str.

    Examples
    --------
    >>> from aeon.datasets import load_covid_3month
    >>> from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
    >>> X_train, y_train = load_covid_3month(split="train")
    >>> X_test, y_test = load_covid_3month(split="test")
    >>> regressor = KNeighborsTimeSeriesRegressor(distance="euclidean")
    >>> regressor.fit(X_train, y_train)
    KNeighborsTimeSeriesRegressor(distance='euclidean')
    >>> y_pred = regressor.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "X_inner_type": ["np-list", "numpy3D"],
        "algorithm_type": "distance",
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
        if self._distance_params is None:
            self._distance_params = {}

        super().__init__()

    def _fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or list of
        shape[n_cases] of 2D arrays shape (n_channels,n_timepoints_i)
                If the series are all equal length, a numpy3D will be passed. If
                unequal, a list of 2D numpy arrays is passed, which may have
                different lengths.
        y : array-like, shape = (n_cases)
            The class labels.
        """
        if isinstance(self.distance, str):
            self.metric_ = get_distance_function(metric=self.distance)

        self.X_ = X
        self.y_ = y
        return self

    def _predict(self, X):
        """Predict the target values for the provided data.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or list of
        shape[n_cases] of 2D arrays shape (n_channels,n_timepoints_i)
                If the series are all equal length, a numpy3D will be passed. If
                unequal, a list of 2D numpy arrays is passed, which may have
                different lengths.

        Returns
        -------
        y : array of shape (n_cases)
            Class labels for each data sample.
        """
        self.check_is_fitted()

        preds = np.empty(len(X))
        for i in range(len(X)):
            idx, weights = self._kneighbors(X[i])
            preds[i] = np.average(self.y_[idx], weights=weights)

        return preds

    def _kneighbors(self, X):
        """Find the K-neighbors of a point.

        Returns indices and weights of each point.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or list of
        shape[n_cases] of 2D arrays shape (n_channels,n_timepoints_i)
                If the series are all equal length, a numpy3D will be passed. If
                unequal, a list of 2D numpy arrays is passed, which may have
                different lengths.

        Returns
        -------
        ind : array
            Indices of the nearest points in the population matrix.
        ws : array
            Array representing the weights of each neighbor.
        """
        distances = np.array(
            [
                self.metric_(X, self.X_[j], **self._distance_params)
                for j in range(len(self.X_))
            ]
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
