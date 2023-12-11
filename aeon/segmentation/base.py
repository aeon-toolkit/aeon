"""Abstract base class for time series segmenters.

Segmenters take a single time series of size n and return a segmentation. The
segmentation can be
in different forms
sparse : a list of integers of length n indicating the segment of each time point
dense : a list of start and end points of segments

"""

__all__ = ["BaseSegmenter"]
__author__ = ["TonyBagnall"]

from abc import ABC, abstractmethod
from typing import final

import numpy as np
import pandas as pd

from aeon.base import BaseEstimator

# allowed types for transformers - Series and Panel
ALLOWED_INPUT_TYPES = [
    "np.ndarray",
    "pd.Series",
    "pd.DataFrame",
]


class BaseSegmenter(BaseEstimator, ABC):
    """Docstring.

    Parameters
    ----------
    n_segments : int, default = 2
    axis : int, default = 0
    """

    _tags = {
        "input_data_type": "Series",
        "output_data_type": "Series",
        "X_inner_type": "np.ndarray",  # One of ALLOWED_INPUT_TYPES
        "capability:unequal_length": False,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "capability:multithreading": False,
        "fit_is_empty": True,
    }

    def __init__(self, n_segments=None, axis=0):
        self.n_segments = n_segments
        self.axis = axis
        self._is_fitted = False
        super(BaseSegmenter, self).__init__()

    @final
    def fit(self, X, y=None, axis=None):
        """Fit time series segmenter from training data."""
        if axis is None:
            axis = self.axis
        self._check_X(X, axis)
        if y is not None:
            self._check_y(y)
        if not isinstance(X, pd.Series):
            if self.axis != axis:
                X = X.T
        self._fit(X=X, y=y)
        self._is_fitted = True
        return self

    def predict(self, X, axis=None):
        """Create segmentation."""
        if axis is None:
            axis = self.axis
        self._check_X(X, axis)
        if not isinstance(X, pd.Series):
            if self.axis != axis:
                X = X.T
        return self._predict(X)

    def fit_predict(self, X, y=None):
        """Fit to data, then predict it."""
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        return self.fit(X, y).predict(X)

    @abstractmethod
    def _fit(self, X, y):
        """Fit time series classifier to training data."""
        ...

    @abstractmethod
    def _predict(self, X) -> np.ndarray:
        """Predict."""
        ...

    def _check_X(self, X, axis):
        # Checks: check valid type
        if not (
            isinstance(X, np.ndarray)
            or isinstance(X, pd.Series)
            or isinstance(X, pd.DataFrame)
        ):
            raise ValueError(" Error in input type =  ", type(X))
        if isinstance(X, np.ndarray):
            # Check valid shape
            if X.ndims > 2:
                raise ValueError(" Should be 1D or 2D")
            # Check axis
            if axis > 1 or axis < 0:
                raise ValueError(" Axis should be 0 or 1")

    def _check_y(self, y):
        if not isinstance(y, np.ndarray):
            raise ValueError("Error in y")
