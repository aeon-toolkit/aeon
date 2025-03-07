"""KNN time series classification.

A KNN classifier which supports time series distance measures.
The class can take callables or uses string references to utilise the numba based
distances in aeon.distances.
"""

import numbers
from typing import Optional

__maintainer__ = []
__all__ = ["KNeighborsTimeSeriesClassifier"]

from typing import Callable, Union

import numpy as np

from aeon.classification.base import BaseClassifier
from aeon.distances import pairwise_distance
from aeon.utils._threading import threaded

WEIGHTS_SUPPORTED = ["uniform", "distance"]


class KNeighborsTimeSeriesClassifier(BaseClassifier):
    """
    K-Nearest Neighbour Time Series Classifier.

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
    >>> from aeon.datasets import load_unit_test
    >>> from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> classifier = KNeighborsTimeSeriesClassifier(distance="euclidean")
    >>> classifier.fit(X_train, y_train)
    KNeighborsTimeSeriesClassifier(...)
    >>> y_pred = classifier.predict(X_test)
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
        shape [n_cases] of 2D arrays shape (n_channels,n_timepoints_i)
        If the series are all equal length, a numpy3D will be passed. If unequal,
        a list of 2D numpy arrays is passed, which may have different lengths.
        y : array-like, shape = (n_cases)
            The class labels.
        """
        self.X_ = X
        self.classes_, self.y_ = np.unique(y, return_inverse=True)
        return self

    def _predict_proba(self, X):
        """
        Return probability estimates for the provided data.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or list of
        shape[n_cases] of 2D arrays shape (n_channels,n_timepoints_i)
                If the series are all equal length, a numpy3D will be passed. If
                unequal, a list of 2D numpy arrays is passed, which may have
                different lengths.

        Returns
        -------
        p : array of shape = (n_cases, n_classes_)
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        preds = np.zeros((len(X), len(self.classes_)))
        for i in range(len(X)):
            idx, weights = self.kneighbors(X[i])
            for id, w in zip(idx, weights):
                predicted_class = self.y_[id]
                preds[i, predicted_class] += w

            preds[i] = preds[i] / np.sum(preds[i])

        return preds

    def _predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or list of
        shape[n_cases] of 2D arrays shape (n_channels,n_timepoints_i)
        If the series are all equal length, a numpy3D will be passed. If unequal, a list
        of 2D numpy arrays is passed, which may have different lengths.

        Returns
        -------
        y : array of shape (n_cases)
            Class labels for each data sample.
        """
        self._check_is_fitted()

        indexes = self.kneighbors(X, return_distance=False)[:, 0]
        return self.classes_[self.y_[indexes]]

    @threaded
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
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_neighbors}")
        elif not isinstance(n_neighbors, numbers.Integral):
            raise TypeError(
                f"n_neighbors does not take {type(n_neighbors)} value, "
                "enter integer value"
            )

        query_is_train = X is None
        if query_is_train:
            X = self.X_
            n_neighbors += 1
        else:
            X = self._preprocess_collection(X, store_metadata=False)
            self._check_shape(X)

        distances = pairwise_distance(
            X,
            self.X_ if not query_is_train else None,
            method=self.distance,
            n_jobs=self.n_jobs,
            **self._distance_params,
        )

        sample_range = np.arange(distances.shape[0])[:, None]
        neigh_ind = np.argpartition(distances, n_neighbors - 1, axis=1)
        neigh_ind = neigh_ind[:, :n_neighbors]
        neigh_ind = neigh_ind[
            sample_range, np.argsort(distances[sample_range, neigh_ind])
        ]

        if query_is_train:
            neigh_ind = neigh_ind[:, 1:]

        if return_distance:
            if query_is_train:
                neigh_dist = distances[sample_range, neigh_ind]
                return neigh_dist, neigh_ind
            return distances[sample_range, neigh_ind], neigh_ind

        return neigh_ind

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
