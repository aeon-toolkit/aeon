__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["BaseAnomalyDetector"]

from abc import ABC, abstractmethod
from typing import final

import numpy
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

        # If None given, assume it is correct.
        if axis is None:
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
        if not self.get_class_tag("fit_is_empty"):
            self.check_is_fitted()

        # If None given, assume it is correct.
        if axis is None:
            axis = self.axis

        X = self._preprocess_series(X, axis)

        return self._predict(X)

    @final
    def fit_predict(self, X, y=None, axis=None) -> np.ndarray:
        if self.get_class_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")

        # reset estimator at the start of fit
        self.reset()

        # If None given, assume it is correct.
        if axis is None:
            axis = self.axis

        X = self._preprocess_series(X, axis)

        if self.get_class_tag("fit_is_empty"):
            self._is_fitted = True
            return self._predict(X)

        if y is not None:
            y = self._check_y(y)

        pred = self._fit_predict(X, y)

        # this should happen last
        self._is_fitted = True
        return pred

    def _fit(self, X, y):
        return self

    @abstractmethod
    def _predict(self, X) -> np.ndarray:
        ...

    def _fit_predict(self, X, y):
        self._fit(X, y)
        return self._predict(X)

    def _check_y(self, y: VALID_INPUT_TYPES) -> np.ndarray:
        # Remind user if y is not required for this estimator on failure
        req_msg = (
            f"{self.__class__.__name__} does not require a y input."
            if self.get_class_tag("requires_y")
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
            elif not issubclass(y.dtype.type, numpy.bool_):
                fail = True

            if fail:
                raise ValueError(
                    "Error in input type for y: y input type must be an integer array "
                    "containing 0 and 1 or a boolean array." + req_msg
                )
        elif isinstance(y, pd.Series):
            # check series is of boolean dtype
            if not pd.api.types.is_bool_dtype(y):
                raise ValueError(
                    "Error in input type for y: y input as pd.Series must have a "
                    "boolean dtype." + req_msg
                )

            new_y = y.values
        elif isinstance(y, pd.DataFrame):
            # only accept size 1 dataframe
            if y.shape[1] > 1:
                raise ValueError(
                    "Error in input type for y: y input as pd.DataFrame should have a "
                    "single column series."
                )

            # check column is of boolean dtype
            if not all(pd.api.types.is_bool_dtype(y[col]) for col in y.columns):
                raise ValueError(
                    "Error in input type for y: y input as pd.DataFrame must have a "
                    "boolean dtype." + req_msg
                )

            new_y = y.squeeze().values
        else:
            raise ValueError(
                f"Error in input type for y: it should be one of {VALID_INPUT_TYPES}, "
                f"saw {type(y)}"
            )

        new_y = new_y.astype(bool)
        return new_y
