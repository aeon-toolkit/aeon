"""Base class for clustering."""

__maintainer__ = []
__all__ = ["BaseClusterer"]

from abc import abstractmethod
from typing import final

import numpy as np
from sklearn.base import ClusterMixin

from aeon.base import BaseCollectionEstimator
from aeon.utils.decorators.method_timer import method_timer


class BaseClusterer(ClusterMixin, BaseCollectionEstimator):
    """Abstract base class for time series clusterers.

    Parameters
    ----------
    n_clusters : int, default=None
        Number of clusters for model.
    """

    _tags = {
        "fit_is_empty": False,
        "capability:predict": True,
    }

    @abstractmethod
    def __init__(self):
        super().__init__()

    @final
    @method_timer("fit_time_millis_", overwrite=False, remove_on_start=True)
    def fit(self, X, y=None) -> BaseCollectionEstimator:
        """Fit time series clusterer to training data.

        Clusters the collection ``X`` and stores the cluster index of each time
        series in the ``labels_`` attribute.

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
        X = self._preprocess_collection(X)
        self._fit(X)
        self.is_fitted = True
        return self

    @final
    def predict(self, X) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Only clusterers with the ``capability:predict`` tag set to True support
        assigning previously unseen cases to clusters. Transductive clusterers
        (tag set to False) can only cluster the collection they are fitted on;
        use ``fit_predict(X)`` or inspect ``labels_`` after ``fit(X)`` instead.

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
        np.array
            shape ``(n_cases,)``, index of the cluster to which each time series in X
            belongs.

        Raises
        ------
        NotImplementedError
            If the ``capability:predict`` tag is False, as the clusterer does not
            support out-of-sample prediction.
        """
        self._check_is_fitted()
        self._check_predict_capability()
        X = self._preprocess_collection(X, store_metadata=False)
        self._check_shape(X)
        return self._predict(X)

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

        Raises
        ------
        NotImplementedError
            If the ``capability:predict`` tag is False, as the clusterer does not
            support out-of-sample prediction.
        """
        self._check_is_fitted()
        self._check_predict_capability()
        X = self._preprocess_collection(X, store_metadata=False)
        self._check_shape(X)
        return self._predict_proba(X)

    @final
    def fit_predict(self, X, y=None) -> np.ndarray:
        """Fit the clusterer to X and return the cluster index of each time series.

        Convenience method; equivalent of calling ``fit(X)`` and returning
        ``labels_``, the cluster indices assigned to the fitted collection. It
        does not call ``predict(X)``, so it is valid for all clusterers,
        including transductive ones that do not support out-of-sample
        prediction (``capability:predict`` tag set to False).

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_cases, n_timepoints) or shape
            (n_cases, n_channels, n_timepoints)).
            Time series instances used to train the clusterer and return their assigned
            cluster indices.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_cases,))
            Index of the cluster each time series in X belongs to.
        """
        self.fit(X, y)
        return self.labels_

    def _check_predict_capability(self):
        """Raise an error if the clusterer cannot predict on unseen data."""
        if not self.get_tag("capability:predict"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support out-of-sample "
                "prediction. Use fit_predict(X) to cluster a collection, or "
                "inspect labels_ after fit(X)."
            )

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Default behaviour is to call _predict and set the predicted class probability
        to 1, other class probabilities to 0. Override if better estimates are
        obtainable.

        Columns are fixed by the fitted clusterer, not by the clusters present in
        ``X``. If ``n_clusters`` is set and both ``labels_`` and the predictions
        are cluster indices in ``[0, n_clusters)``, column ``j`` is cluster ``j``
        and there are ``n_clusters`` columns. Otherwise, such as for negative
        noise labels, column ``j`` is the ``j``-th value of
        ``np.unique(labels_)``.

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

        Raises
        ------
        ValueError
            If the labels are not cluster indices and ``_predict`` returns a label
            not present in ``labels_``.
        """
        preds = np.asarray(self._predict(X))
        n_cases = len(preds)
        n_clusters = getattr(self, "n_clusters", None)
        fitted = np.asarray(getattr(self, "labels_", preds))

        if (
            n_clusters is not None
            and _is_cluster_index(fitted, n_clusters)
            and _is_cluster_index(preds, n_clusters)
        ):
            cols = preds.astype(int)
            n_cols = n_clusters
        else:
            clusters = np.unique(fitted)
            cols = np.searchsorted(clusters, preds)
            unseen = (cols == len(clusters)) | (
                clusters[np.minimum(cols, len(clusters) - 1)] != preds
            )
            if np.any(unseen):
                raise ValueError(
                    f"_predict returned labels {np.unique(preds[unseen])} that are "
                    f"not in labels_ {clusters}, so they have no predict_proba column."
                )
            n_cols = len(clusters)
            if n_clusters is not None:
                n_cols = max(n_cols, n_clusters)

        dists = np.zeros((n_cases, n_cols))
        dists[np.arange(n_cases), cols] = 1
        return dists

    @abstractmethod
    def _predict(self, X) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_cases, n_timepoints) or shape
            (n_cases,n_channels,n_timepoints)).
            Time series instances to predict their cluster indexes.

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


def _is_cluster_index(labels, n_clusters):
    """Check whether all labels are integer values in ``[0, n_clusters)``."""
    if not np.issubdtype(labels.dtype, np.number):
        return False
    return bool(
        np.all(labels >= 0)
        and np.all(labels < n_clusters)
        and np.all(labels == np.floor(labels))
    )
