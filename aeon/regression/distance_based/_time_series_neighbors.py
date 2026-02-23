"""KNN time series regression.

This class is a KNN regressor which supports time series distance measures.
The class has hardcoded string references to numba based distances in aeon.distances.
It can also be used with callables, or aeon (pairwise transformer) estimators.
"""

__maintainer__ = []
__all__ = ["KNeighborsTimeSeriesRegressor"]

import numbers
from collections.abc import Callable

import numpy as np

from aeon.distances import pairwise_distance
from aeon.regression.base import BaseRegressor
from aeon.utils.validation import check_n_jobs

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
    n_jobs : int, default=1
        The number of jobs to run in parallel. If -1, then the number of jobs is set
        to the number of CPU cores. If 1, then the function is executed in a single
        thread. If greater than 1, then the function is executed in parallel.

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
        distance: str | Callable = "dtw",
        distance_params: dict | None = None,
        n_neighbors: int = 1,
        weights: str | Callable = "uniform",
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
        self.X_ = X
        self.y_ = y
        self._n_jobs = check_n_jobs(self.n_jobs)
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
            neigh_dist, neigh_ind = self._kneighbors(
                X[i : i + 1],
                n_neighbors=self.n_neighbors,
                return_distance=True,
                query_is_train=False,
            )
            neigh_dist = neigh_dist[0]
            neigh_ind = neigh_ind[0]

            if self.weights == "distance":
                # Using epsilon ~= 0 to avoid division by zero
                weights = 1 / (neigh_dist + np.finfo(float).eps)
            elif self.weights == "uniform":
                weights = np.repeat(1.0, len(neigh_ind))
            else:
                raise Exception(f"Invalid kNN weights: {self.weights}")

            preds[i] = np.average(self.y_[neigh_ind], weights=weights)

        return preds

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Find the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or list of
        shape [n_cases] of 2D arrays shape (n_channels,n_timepoints_i)
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.
        n_neighbors : int, default=None
            Number of neighbors required for each sample. The default is the value
            passed to the constructor.
        return_distance : bool, default=True
            Whether or not to return the distances.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Array representing the distances to points, only present if
            return_distance=True.
        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.
        """
        self._check_is_fitted()

        # Input validation
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif not isinstance(n_neighbors, numbers.Integral):
            raise TypeError(
                f"n_neighbors does not take {type(n_neighbors)} value, enter integer "
                f"value"
            )
        elif n_neighbors <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_neighbors}")

        if not isinstance(return_distance, bool):
            raise TypeError(
                f"return_distance must be a boolean, got {type(return_distance)}"
            )

        # Preprocess X if provided
        query_is_train = X is None
        if query_is_train:
            X = self.X_
        else:
            X = self._preprocess_collection(X, store_metadata=False)
            self._check_shape(X)

        # Validate n_neighbors against data size
        n_samples_fit = len(self.X_)
        if query_is_train:
            if not (n_neighbors < n_samples_fit):
                raise ValueError(
                    "Expected n_neighbors < n_samples_fit, but "
                    f"n_neighbors = {n_neighbors}, n_samples_fit = {n_samples_fit}, "
                    f"n_samples = {len(X)}"
                )
        else:
            if not (n_neighbors <= n_samples_fit):
                raise ValueError(
                    "Expected n_neighbors <= n_samples_fit, but "
                    f"n_neighbors = {n_neighbors}, n_samples_fit = {n_samples_fit}, "
                    f"n_samples = {len(X)}"
                )

        return self._kneighbors(X, n_neighbors, return_distance, query_is_train)

    def _kneighbors(self, X, n_neighbors, return_distance, query_is_train):
        """Find the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or list of
        shape [n_cases] of 2D arrays shape (n_channels,n_timepoints_i)
            The query point or points.
        n_neighbors : int
            Number of neighbors required for each sample.
        return_distance : bool
            Whether or not to return the distances.
        query_is_train : bool
            Whether the query points are from the training set.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Array representing the distances to points, only present if
            return_distance=True.
        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.
        """
        distances = pairwise_distance(
            X,
            None if query_is_train else self.X_,
            method=self.distance,
            n_jobs=self.n_jobs,
            **self._distance_params,
        )

        # If querying the training set, exclude self by setting diag to +inf
        if query_is_train:
            np.fill_diagonal(distances, np.inf)

        k = n_neighbors
        # 1) partial select smallest k
        idx_part = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        # 2) sort those k by (distance, index)
        row_idx = np.arange(distances.shape[0])[:, None]
        part_d = distances[row_idx, idx_part]
        # argsort by distance, then by index for ties (lexsort uses last key as primary)
        order = np.lexsort((idx_part, part_d), axis=1)
        neigh_ind = idx_part[row_idx, order]

        if return_distance:
            neigh_dist = distances[row_idx, neigh_ind]
            return neigh_dist, neigh_ind
        return neigh_ind

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> dict | list[dict]:
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
