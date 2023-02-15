# -*- coding: utf-8 -*-
"""KNN time series regression.

This class is a KNN classifier which supports time series distance measures.
The class has hardcoded string references to numba based distances in sktime.distances.
It can also be used with callables, or sktime (pairwise transformer) estimators.
"""

__author__ = ["TonyBagnall", "GuiArcencio"]
__all__ = ["KNeighborsTimeSeriesRegressor"]

import numpy as np

from sktime.distances import distance_factory
from sktime.classification.base import BaseClassifier

WEIGHTS_SUPPORTED = [
    "uniform",
    "distance"
]

class KNeighborsTimeSeriesClassifier(BaseClassifier):
    """KNN Time Series Classifier.

    An adapted K-Neighbors Classifier for time series data.

    This class is a KNN classifier which supports time series distance measures.
    It has hardcoded string references to numba based distances in sktime.distances,
    and can also be used with callables, or sktime (pairwise transformer) estimators.

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
            'lcss', 'edr', 'erp', 'msm'
        this will substitute a hard-coded distance metric from sktime.distances
        When mpdist is used, the subsequence length (parameter m) must be set
            Example: knn_mpdist = KNeighborsTimeSeriesClassifier(
                                metric='mpdist', metric_params={'m':30})
        if callable, must be of signature (X: Panel, X2: Panel) -> np.ndarray
            output must be mxn array if X is Panel of m Series, X2 of n Series
            if distance_mtype is not set, must be able to take
                X, X2 which are pd_multiindex and numpy3D mtype
        can be pairwise panel transformer inheriting from BasePairwiseTransformerPanel
    distance_params : dict, optional. default = None.
        dictionary for metric parameters , in case that distance is a str
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Parameter for compatibility purposes, still unimplemented.

    Examples
    --------
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.regression.distance_based import KNeighborsTimeSeriesClassifier
    >>> X_train, y_train = load_unit_test(return_X_y=True, split="train")
    >>> X_test, y_test = load_unit_test(return_X_y=True, split="test")
    >>> classifier = KNeighborsTimeSeriesClassifier()
    >>> classifier.fit(X_train, y_train)
    KNeighborsTimeSeriesClassifier(...)
    >>> y_pred = classifier.predict(X_test)
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
        n_jobs=1
    ):
        self.distance = distance
        self.distance_params = distance_params
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

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
        X : sktime-compatible Panel data format, with n_samples series
        y : {array-like, sparse matrix}
            Class labels of shape = [n_samples]
        """
        if isinstance(self.distance, str):
            if self.distance_params is None:
                self.metric_ = distance_factory(X[0], X[0], metric=self.distance)
            else:
                self.metric_ = distance_factory(
                    X[0], X[0], metric=self.distance, **self.distance_params
                )

        self.X_ = X
        self.classes_, self.y_ = np.unique(y, return_inverse=True)
        return self

    def _predict_proba(self, X):
        """Return probability estimates for the provided data.

        Parameters
        ----------
        X : sktime-compatible Panel data format, with n_samples series

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        self.check_is_fitted()

        preds = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(X.shape[0]):
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
        X : sktime-compatible Panel data format, with n_samples series

        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        self.check_is_fitted()

        preds = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for i in range(X.shape[0]):
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
        X : sktime-compatible data format, Panel or Series, with n_samples series

        Returns
        -------
        ind : array
            Indices of the nearest points in the population matrix.
        ws : array
            Array representing the weights of each neighbor.
        """
        
        distances = np.array(
            [self.metric_(X, self.X_[j]) for j in range(self.X_.shape[0])]
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
            ws = np.repeat(1., self.n_neighbors)
        else:
            raise Exception(f"Invalid kNN weights: {self.weights}")

        return closest_idx, ws
