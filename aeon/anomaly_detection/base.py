from abc import ABC, abstractmethod
from typing import final

import numpy as np

from aeon.base import BaseSeriesEstimator


class BaseAnomalyDetector(BaseSeriesEstimator, ABC):
    """
    Parameters
    ----------
    axis : int, default = 1
        Axis along which to segment if passed a multivariate series (2D input). If axis
        is 0, it is assumed each column is a time series and each row is a
        timepoint. i.e. the shape of the data is ``(n_timepoints,n_channels)``.
        ``axis == 1`` indicates the time series are in rows, i.e. the shape of the data
        is ``(n_channels, n_timepoints)`.

    """

    _tags = {
        "X_inner_type": "np.ndarray",  # One of VALID_INNER_TYPES
        "fit_is_empty": True,
        "requires_y": False,
    }

    def __init__(self, axis=1):
        self.axis = axis

        self._is_fitted = False

        super().__init__(axis=axis)

    @final
    def fit(self, X, y=None, axis=None):
        if self.get_class_tag("fit_is_empty"):
            self._is_fitted = True
            return self

        if self.get_class_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")

        # reset estimator at the start of fit
        self.reset()

        if axis is None:  # If none given, assume it is correct.
            axis = self.axis

        X = self._preprocess_series(X, axis)
        if y is not None:
            y = self._check_y(y)

        self._fit(X=X, y=y)

        # this should happen last
        self._is_fitted = True
        return self

    @final
    def predict(self, X, axis=None) -> np.ndarray:
        self.check_is_fitted()

        if axis is None:
            axis = self.axis

        X = self._preprocess_series(X, axis)

        return self._predict(X)

    def _fit(self, X, y=None):
        return self

    @abstractmethod
    def _predict(self, X) -> np.ndarray: ...
