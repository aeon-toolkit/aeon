"""
Abstract base class for whole-series/collection anomaly detectors.

    class name: BaseCollectionAnomalyDetector

Defining methods:
    fitting                 - fit(self, X, y)
    predicting              - predict(self, X)

Data validation:
    data processing         - _preprocess_collection(self, X, store_metadata=True)
    shape verification      - _check_shape(self, X)

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted
    train input metadata    - metadata_
    resetting state         - reset(self)

Tags:
    default estimator tags  - _tags
    tag retrieval           - get_tag(self, tag_name)
    tag setting             - set_tag(self, tag_name, value)
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["BaseCollectionAnomalyDetector"]

from abc import abstractmethod
from typing import final

import numpy as np
import pandas as pd

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.base import BaseCollectionEstimator


class BaseCollectionAnomalyDetector(BaseCollectionEstimator, BaseAnomalyDetector):
    """
    Abstract base class for collection anomaly detectors.

    The base detector specifies the methods and method signatures that all
    collection anomaly detectors have to implement. Attributes with an underscore
    suffix are set in the method fit.

    Attributes
    ----------
    is_fitted : bool
        True if the estimator has been fitted, False otherwise.
        Unused if ``"fit_is_empty"`` tag is set to True.
    metadata_ : dict
        Dictionary containing metadata about the `fit` input data.
    _tags_dynamic : dict
        Dictionary containing dynamic tag values which have been set at runtime.
    """

    def __init__(self):
        super().__init__()

    @final
    def fit(self, X, y=None):
        """Fit collection anomaly detector to training data.

        Parameters
        ----------
        X : np.ndarray or list
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. Other types are
            allowed and converted into one of the above.

            Different estimators have different capabilities to handle different
            types of input. If ``self.get_tag("capability:multivariate")`` is False,
            they cannot handle multivariate series, so either ``n_channels == 1`` is
            true or X is 2D of shape ``(n_cases, n_timepoints)``. If ``self.get_tag(
            "capability:unequal_length")`` is False, they cannot handle unequal
            length input. In both situations, a ``ValueError`` is raised if X has a
            characteristic that the estimator does not have the capability for is
            passed.
        y : np.ndarray
            1D np.array of int, of shape ``(n_cases)`` - anomaly labels
            (ground truth) for fitting indices corresponding to instance indices in X.

        Returns
        -------
        self : BaseCollectionAnomalyDetector
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        if self.get_tag("fit_is_empty"):
            self.is_fitted = True
            return self

        if self.get_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")

        # reset estimator at the start of fit
        self.reset()

        X = self._preprocess_collection(X)
        if y is not None:
            y = self._check_y(y, self.metadata_["n_cases"])

        self._fit(X, y)

        # this should happen last
        self.is_fitted = True
        return self

    @final
    def predict(self, X):
        """Predicts anomalies for time series in X.

        Parameters
        ----------
        X : np.ndarray or list
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``
            other types are allowed and converted into one of the above.

            Different estimators have different capabilities to handle different
            types of input. If ``self.get_tag("capability:multivariate")`` is False,
            they cannot handle multivariate series, so either ``n_channels == 1`` is
            true or X is 2D of shape ``(n_cases, n_timepoints)``. If ``self.get_tag(
            "capability:unequal_length")`` is False, they cannot handle unequal
            length input. In both situations, a ``ValueError`` is raised if X has a
            characteristic that the estimator does not have the capability for is
            passed.

        Returns
        -------
        predictions : np.ndarray
            1D np.array of float, of shape (n_cases) - predicted anomalies or anomaly
            scores for each time series in X.
            Indices correspond to instance indices in X.
        """
        fit_empty = self.get_tag("fit_is_empty")
        if not fit_empty:
            self._check_is_fitted()

        X = self._preprocess_collection(X, store_metadata=False)
        # Check if X has the correct shape seen during fitting
        self._check_shape(X)

        return self._predict(X)

    @final
    def fit_predict(self, X, y=None, axis=1) -> np.ndarray:
        """Fit time series anomaly detector and find anomalies for X.

        Parameters
        ----------
        X : np.ndarray or list
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. Other types are
            allowed and converted into one of the above.

            Different estimators have different capabilities to handle different
            types of input. If ``self.get_tag("capability:multivariate")`` is False,
            they cannot handle multivariate series, so either ``n_channels == 1`` is
            true or X is 2D of shape ``(n_cases, n_timepoints)``. If ``self.get_tag(
            "capability:unequal_length")`` is False, they cannot handle unequal
            length input. In both situations, a ``ValueError`` is raised if X has a
            characteristic that the estimator does not have the capability for is
            passed.
        y : np.ndarray
            1D np.array of int, of shape ``(n_cases)`` - anomaly labels
            (ground truth) for fitting indices corresponding to instance indices in X.

        Returns
        -------
        predictions : np.ndarray
            1D np.array of float, of shape (n_cases) - predicted anomalies or anomaly
            scores for each time series in X.
            Indices correspond to instance indices in X.
        """
        if self.get_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")

        # reset estimator at the start of fit
        self.reset()

        X = self._preprocess_series(X, axis, store_metadata=True)

        if self.get_tag("fit_is_empty"):
            self.is_fitted = True
            return self._predict(X)

        if y is not None:
            y = self._check_y(y)

        pred = self._fit_predict(X, y)

        # this should happen last
        self.is_fitted = True
        return pred

    def _fit(self, X, y):
        return self

    @abstractmethod
    def _predict(self, X): ...

    def _fit_predict(self, X, y):
        self._fit(X, y)
        return self._predict(X)

    def _check_y(self, y, n_cases):
        """Check y input is valid.

        Must be 1-dimensional and contain only 0s (no anomaly) and 1s (anomaly).
        Must match the number of cases in X.
        """
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError(
                f"y must be a np.array or a pd.Series, but found type: {type(y)}"
            )
        if isinstance(y, np.ndarray) and y.ndim > 1:
            raise TypeError(f"y must be 1-dimensional, found {y.ndim} dimensions")

        if not np.bitwise_or(y == 0, y == 1).all():
            raise ValueError(
                "y input must only contain 0 (not anomalous) or 1 (anomalous) values."
            )

        # Check matching number of labels
        n_labels = y.shape[0]
        if n_cases != n_labels:
            raise ValueError(
                f"Mismatch in number of cases. Found X = {n_cases} and y = {n_labels}"
            )

        if isinstance(y, pd.Series):
            y = pd.Series.to_numpy(y)

        return y
