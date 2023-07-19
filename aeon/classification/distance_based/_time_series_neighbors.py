# -*- coding: utf-8 -*-
"""KNN time series classification.

A KNN classifier which supports time series distance measures.
The class has hardcoded string references to numba based distances in aeon.distances.
It can also be used with callable distance functions.
"""

__author__ = ["TonyBagnall", "GuiArcencio"]
__all__ = ["KNeighborsTimeSeriesClassifier"]

import numpy as np

from aeon.classification.base import BaseClassifier
from aeon.distances import get_distance_function

WEIGHTS_SUPPORTED = ["uniform", "distance"]


class KNeighborsTimeSeriesClassifier(BaseClassifier):
    """
    KNN Time Series Classifier.

    An adapted K-Neighbors Classifier for time series data.

    This class is a KNN classifier which supports time series distance measures.
    It has hardcoded string references to numba based distances in aeon.distances,
    and can also be used with callables, or aeon (pairwise transformer) estimators.

    Parameters
    ----------
    n_neighbors : int, default =1
        k for knn.
    weights : str or callable, default = 'uniform'
        Mechanism for weighting a vote one of: 'uniform', 'distance', or a callable
        function.
    distance : str or callable, default ='dtw'
        Distance measure between time series.
        if str, must be one of the following strings:
            'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw',
            'lcss', 'edr', 'erp', 'msm', 'twe', 'mpdist'
        this will substitute a hard-coded distance metric from aeon.distances
        When mpdist is used, the subsequence length (parameter m) must be set
            Example: knn_mpdist = KNeighborsTimeSeriesClassifier(
                                distance='mpdist', distance_params={'m':30})
        if callable, must be of signature (X: np.ndarray, X2: np.ndarray) -> np.ndarray
    distance_params : dict, default = None
        Dictionary for metric parameters for the case that distance is a str.
    n_jobs : int, default = None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Parameter for compatibility purposes, still unimplemented.

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
        "X_inner_mtype": ["np-list", "numpy3D"],
        "algorithm_type": "distance",
    }

    def __init__(
        self,
        distance="dtw",
        distance_params=None,
        n_neighbors=1,
        weights="uniform",
        n_jobs=1,
    ):
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

        super(KNeighborsTimeSeriesClassifier, self).__init__()

    def _fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or list of
        shape [n_cases] of 2D arrays shape (n_channels,n_timepoints_i)
        If the series are all equal length, a numpy3D will be passed. If unequal,
        a list of 2D numpy arrays is passed, which may have different lengths.
        y : array-like, shape = (n_cases)
            The class labels.
        """
        if isinstance(self.distance, str):
            self.metric_ = get_distance_function(metric=self.distance)

        self.X_ = X
        self.classes_, self.y_ = np.unique(y, return_inverse=True)
        return self

    def _predict_proba(self, X):
        """Return probability estimates for the provided data.

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
        self.check_is_fitted()

        preds = np.zeros((len(X), len(self.classes_)))
        for i in range(len(X)):
            idx, weights = self._kneighbors(X[i])
            for id, w in zip(idx, weights):
                predicted_class = self.y_[id]
                preds[i, predicted_class] += w

            preds[i] = preds[i] / np.sum(preds[i])

        return preds

    def _predict(self, X):
        """Predict the class labels for the provided data.

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

        preds = np.empty(len(X), dtype=self.classes_.dtype)
        for i in range(len(X)):
            scores = np.zeros(len(self.classes_))
            idx, weights = self._kneighbors(X[i])
            for id, w in zip(idx, weights):
                predicted_class = self.y_[id]
                scores[predicted_class] += w

            preds[i] = self.classes_[np.argmax(scores)]

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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
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
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        # non-default distance and algorithm
        params1 = {"distance": "euclidean"}

        return [params1]
