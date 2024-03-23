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
        pass

    @final
    def predict(self, X, axis=None) -> np.ndarray:
        pass

    def _fit(self, X, y):
        return self

    @abstractmethod
    def _predict(self, X) -> np.ndarray: ...
