"""Base class for estimators that fit single (possibly multivariate) time series."""

import numpy as np
import pandas as pd

from aeon.base._base import BaseEstimator
from aeon.utils.validation._dependencies import _check_estimator_deps

# allowed input and internal data types for Segmenters
VALID_INNER_TYPES = [
    "ndarray",
    "Series",
    "DataFrame",
]
VALID_INPUT_TYPES = [pd.DataFrame, pd.Series, np.ndarray]


class BaseSeriesEstimator(BaseEstimator):
    """Base class for estimators that use single (possibly multivariate) time series.

    Provides functions that are common to BaseSegmenter,
    BaseSeriesTransformer for the checking and
    conversion of input to fit, predict and transform, where relevant.

    It also stores the common default tags used by all the subclasses and meta data
    describing the characteristics of time series passed to ``fit``.
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "X_inner_type": "ndarray",
        "capability:multithreading": False,
    }

    def __init__(self, axis=0):
        self.axis = axis
        super(BaseSeriesEstimator, self).__init__()
        _check_estimator_deps(self)

    def _check_input_series(self, X):
        """Check input is a data structure only containing floats."""
        # Checks: check valid type and axis
        if type(X) not in VALID_INPUT_TYPES:
            raise ValueError(
                f" Error in input type should be one onf "
                f" {VALID_INNER_TYPES}, saw {type(X)}"
            )
        if isinstance(X, np.ndarray):
            # Check valid shape
            if X.ndim > 2:
                raise ValueError(" Should be 1D or 2D")
            if not (
                issubclass(X.dtype.type, np.integer)
                or issubclass(X.dtype.type, np.floating)
            ):
                raise ValueError(" array must contain floats or ints")
        elif isinstance(X, pd.Series):
            if not pd.api.types.is_numeric_dtype(X):
                raise ValueError("pd.Series must be numeric")
        else:
            if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
                raise ValueError("pd.DataFrame must be numeric")

    def _check_capabilities(self, X, axis):
        """Check can handle multivariate and missing values."""
        if self.get_tag("capability:multivariate") is False:
            if X.ndim > 1:
                raise ValueError("Multivariate data not supported")

    def _convert_series(self, X, axis):
        inner = self.get_tag("X_inner_type")
        input = type(X).__name__
        if inner != input:
            if inner == "ndarray":
                X = X.to_numpy()
            elif inner == "Series":
                if input == "ndarray":
                    X = pd.Series(X)
            elif inner == "DataFrame":
                X = pd.DataFrame(X)
        if axis > 1 or axis < 0:
            raise ValueError(" Axis should be 0 or 1")
        if self.get_tag("capability:multivariate") and X.ndim == 1:
            X = X.reshape(1, -1)
        else:
            X = X.squeeze()
        if X.ndim > 1:
            if self.axis != axis:
                X = X.T
        return X
