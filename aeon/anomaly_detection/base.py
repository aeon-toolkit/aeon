from abc import ABC, abstractmethod
from typing import final

import numpy as np
import pandas as pd

from aeon.base import BaseSeriesEstimator
from aeon.base._base_series import VALID_INPUT_TYPES


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
            y = self._check_y(y, self.get_class_tag("requires_y"))

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

    def _check_y(self, y: VALID_INPUT_TYPES, requires_y: bool):
        # Remind user if y is not required for this estimator on failure
        req_msg = (
            f"{self.__class__.__name__} does not require a y input."
            if requires_y
            else ""
        )

        new_y = y

        # must be a valid input type, see VALID_INPUT_TYPES in BaseSeriesEstimator
        if isinstance(y, np.ndarray):
            # check valid shape
            if y.ndim > 1:
                raise ValueError(
                    "Error in input type for y: y input as np.ndarray should be 1D."
                    + req_msg
                )

            # check valid dtype
            fail = False
            if issubclass(y.dtype.type, np.integer):
                new_y = y.astype(bool)
                fail = not np.array_equal(y, new_y)
            elif not issubclass(y.dtype.type, bool):
                fail = True

            if fail:
                raise ValueError(
                    "Error in input type for y: y input type must be an integer array "
                    "containing 0 and 1 or a boolean array." + req_msg
                )
        elif isinstance(y, pd.Series):
            if not pd.api.types.is_bool_dtype(y):
                raise ValueError(
                    "Error in input type for y: y input as pd.Series must have a "
                    "boolean dtype." + req_msg
                )

            new_y = y.values
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] > 1:
                raise ValueError(
                    "Error in input type for y: y input as pd.DataFrame should have a "
                    "single column series."
                )

            if not all(pd.api.types.is_numeric_dtype(y[col]) for col in y.columns):
                raise ValueError(
                    "Error in input type for y: y input as pd.DataFrame "
                    "must be numeric"
                )
        else:
            raise ValueError(
                f"Error in input type for y: it should be one of {VALID_INPUT_TYPES}, "
                f"saw {type(y)}"
            )

        return new_y
