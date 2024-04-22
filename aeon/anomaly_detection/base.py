"""Abstract base class for time series anomaly detectors."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["BaseAnomalyDetector"]

from abc import ABC, abstractmethod
from typing import final

import numpy as np
import pandas as pd

from aeon.base import BaseSeriesEstimator
from aeon.base._base_series import VALID_INPUT_TYPES


class BaseAnomalyDetector(BaseSeriesEstimator, ABC):
    """Base class for anomaly detection algorithms.

    Anomaly detection algorithms are used to identify anomalous subsequences in time
    series data. These algorithms take a series of length m and return a boolean, int or
    float array of length m, where each element indicates whether the corresponding
    subsequence is anomalous or its anomaly score.

    Input and internal data format (where m is the number of time points and d is the
    number of channels):
        Univariate series:
            np.ndarray, shape ``(m,)``, ``(m, 1)`` or ``(1, m)`` depending on axis.
            This is converted to a 2D np.ndarray internally.
            pd.DataFrame, shape ``(m, 1)`` or ``(1, m)`` depending on axis.
            pd.Series, shape ``(m,)``.
        Multivariate series:
            np.ndarray array, shape ``(m, d)`` or ``(d, m)`` depending on axis.
            pd.DataFrame ``(m, d)`` or ``(d, m)`` depending on axis.

    Parameters
    ----------
    axis : int
        The time point axis of the input series if it is 2D. If ``axis==0``, it is
        assumed each column is a time series and each row is a time point. i.e. the
        shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
        the time series are in rows, i.e. the shape of the data is
        ``(n_channels, n_timepoints)``.
        Setting this class variable will convert the input data to the chosen axis.
    """

    _tags = {
        "X_inner_type": "np.ndarray",  # One of VALID_INNER_TYPES
        "fit_is_empty": True,
        "requires_y": False,
    }

    def __init__(self, axis):
        self._is_fitted = False

        super().__init__(axis=axis)

    @final
    def fit(self, X, y=None, axis=1):
        """Fit time series anomaly detector to X.

        If the tag ``fit_is_empty`` is true, this just sets the ``is_fitted`` tag to
        true. Otherwise, it checks ``self`` can handle ``X``, formats ``X`` into
        the structure required by ``self`` then passes ``X`` (and possibly ``y``) to
        ``_fit``.

        Parameters
        ----------
        X: one of aeon.base._base_series.VALID_INPUT_TYPES
            The time series to fit the model to.
            A valid aeon time series data structure. See
            aeon.base._base_series.VALID_INPUT_TYPES for aeon supported types.
        y: one of aeon.base._base_series.VALID_INPUT_TYPES, default=None
            The target values for the time series.
            A valid aeon time series data structure. See
            aeon.base._base_series.VALID_INPUT_TYPES for aeon supported types.
        axis: int
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point. i.e. the
            shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels, n_timepoints)``.

        Returns
        -------
        self: BaseAnomalyDetector
            The fitted estimator, reference to self.
        """
        if self.get_class_tag("fit_is_empty"):
            self._is_fitted = True
            return self

        if self.get_class_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")

        # reset estimator at the start of fit
        self.reset()

        X = self._preprocess_series(X, axis, True)
        if y is not None:
            y = self._check_y(y)

        self._fit(X=X, y=y)

        # this should happen last
        self._is_fitted = True
        return self

    @final
    def predict(self, X, axis=1) -> np.ndarray:
        """Find anomalies in X.

        Parameters
        ----------
        X: one of aeon.base._base_series.VALID_INPUT_TYPES
            The time series to fit the model to.
            A valid aeon time series data structure. See
            aeon.base._base_series.VALID_INPUT_TYPES for aeon supported types.
        axis: int, default=1
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point. i.e. the
            shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels, n_timepoints)``.

        Returns
        -------
        anomalies: np.ndarray
            A boolean, int or float array of length len(X), where each element indicates
            whether the corresponding subsequence is anomalous or its anomaly score.
        """
        fit_empty = self.get_class_tag("fit_is_empty")
        if not fit_empty:
            self.check_is_fitted()

        X = self._preprocess_series(X, axis, fit_empty)

        return self._predict(X)

    @final
    def fit_predict(self, X, y=None, axis=1) -> np.ndarray:
        """Fit time series anomaly detector and find anomalies for X.

        Parameters
        ----------
        X: one of aeon.base._base_series.VALID_INPUT_TYPES
            The time series to fit the model to.
            A valid aeon time series data structure. See
            aeon.base._base_series.VALID_INPUT_TYPES for aeon supported types.
        y: one of aeon.base._base_series.VALID_INPUT_TYPES, default=None
            The target values for the time series.
            A valid aeon time series data structure. See
            aeon.base._base_series.VALID_INPUT_TYPES for aeon supported types.
        axis: int, default=1
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point. i.e. the
            shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels, n_timepoints)``.

        Returns
        -------
        anomalies: np.ndarray
            A boolean, int or float array of length len(X), where each element indicates
            whether the corresponding subsequence is anomalous or its anomaly score.
        """
        if self.get_class_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")

        # reset estimator at the start of fit
        self.reset()

        X = self._preprocess_series(X, axis, True)

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
    def _predict(self, X) -> np.ndarray: ...

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
            elif not issubclass(y.dtype.type, np.bool_):
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
