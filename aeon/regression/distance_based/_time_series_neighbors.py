"""KNN time series regression.

This class is a KNN regressor which supports time series distance measures.
The class has hardcoded string references to numba based distances in aeon.distances.
It can also be used with callables, or aeon (pairwise transformer) estimators.
"""

from typing import Optional

__maintainer__ = []
__all__ = ["KNeighborsTimeSeriesRegressor"]

from typing import Callable, Union

import numpy as np

from aeon.distances import get_distance_function
from aeon.regression.base import BaseRegressor

WEIGHTS_SUPPORTED = ["uniform", "distance"]


class KNeighborsTimeSeriesRegressor(BaseRegressor):
    """
    K-Nearest Neighbour Time Series Regressor.

    A KNN classifier which supports time series distance measures.
    It determines distance function through string references to numba
    based distances in aeon.distances, and can also be used with callables.

    Parameters
    ----------
    n_neighbors : int, default = 1
        Set k for knn.
    weights : str or callable, default = 'uniform'
        Mechanism for weighting a vote one of: 'uniform', 'distance', or a callable
        function.
    distance : str or callable, default ='dtw'
        Distance measure between time series.
        Distance metric to compute similarity between time series. A list of valid
        strings for metrics can be found in the documentation for
        :func:`aeon.distances.get_distance_function` or through calling
        :func:`aeon.distances.get_distance_function_names`. If a
        callable is passed it must be
        a function that takes two 2d numpy arrays of shape ``(n_channels,
        n_timepoints)`` as input and returns a float.
    distance_params : dict, default = None
        Dictionary for metric parameters for the case that distance is a str.
    n_jobs : int, default = None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        for more details. Parameter for compatibility purposes, still unimplemented.

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
        "capability:multithreading": True,
        "X_inner_type": ["np-list", "numpy3D"],
        "algorithm_type": "distance",
    }

    def __init__(
        self,
        distance: Union[str, Callable] = "dtw",
        distance_params: Optional[dict] = None,
        n_neighbors: int = 1,
        weights: Union[str, Callable] = "uniform",
        n_jobs: int = 1,
    ) -> None:
        self.distance = distance
        self.distance_params = distance_params
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self._distance_params = distance_params
        if self._distance_params is None:
            self._distance_params = {}

        if weights not in WEIGHTS_SUPPORTED:
            raise ValueError(
                f"Unrecognised kNN weights: {weights}. "
                f"Allowed values are: {WEIGHTS_SUPPORTED}. "
            )
        self.weights = weights

        super().__init__()

    def _fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or list of
        shape[n_cases] of 2D arrays shape (n_channels,n_timepoints_i)
        If the series are all equal length, a numpy3D will be passed. If unequal, a list
        of 2D numpy arrays is passed, which may have different lengths.
        y : array-like, shape = (n_cases)
            The output value.
        """
        self.metric_ = get_distance_function(method=self.distance)
        self.X_ = X
        self.y_ = y
        return self

    def _predict(self, X):
        """
        Predict the target values for the provided data.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or list of
        shape[n_cases] of 2D arrays shape (n_channels,n_timepoints_i)
        If the series are all equal length, a numpy3D will be passed. If unequal, a list
        of 2D numpy arrays is passed, which may have different lengths.

        Returns
        -------
        y : array of shape (n_cases)
            Output values for each data sample.
        """
        preds = np.empty(len(X))
        for i in range(len(X)):
            idx, weights = self._kneighbors(X[i])
            preds[i] = np.average(self.y_[idx], weights=weights)

        return preds

    def _kneighbors(self, X):
        """
        Find the K-neighbors of a point.

        Returns indices and weights of each point.

        Parameters
        ----------
        X : np.ndarray
            A single time series instance if shape = (n_channels, n_timepoints)

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
            # Using epsilon ~= 0 to avoid division by zero
            ws = 1 / (ws + np.finfo(float).eps)
        elif self.weights == "uniform":
            ws = np.repeat(1.0, self.n_neighbors)
        else:
            raise Exception(f"Invalid kNN weights: {self.weights}")

        return closest_idx, ws

    @classmethod
    def _get_test_params(
        cls, parameter_set: str = "default"
    ) -> Union[dict, list[dict]]:
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        # non-default distance and algorithm
        params1 = {"distance": "euclidean"}

        return [params1]
