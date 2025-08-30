"""BaseForecaster class.

A simplified first base class for forecasting models.

"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["BaseForecaster", "DirectForecastingMixin", "IterativeForecastingMixin"]

from abc import abstractmethod
from typing import final

import numpy as np
import pandas as pd

from aeon.base import BaseSeriesEstimator
from aeon.base._base import _clone_estimator
from aeon.utils.data_types import VALID_SERIES_INNER_TYPES


class BaseForecaster(BaseSeriesEstimator):
    """
    Abstract base class for time series forecasters.

    The base forecaster specifies the methods and method signatures that all
    forecasters have to implement. Attributes with an underscore suffix are set in the
    method fit.

    Parameters
    ----------
    horizon : int
        The number of time steps ahead to forecast. If ``horizon`` is one, the
        forecaster will learn to predict one point ahead.
    axis : int
        The axis of time the forecaster uses internally. If ``axis`` is 0, the series
        are internally assumed to be ``(n_timepoints, n_channels)`` and if ``axis`` is
        1, the series are stored as ``(n_channels, n_timepoints)``. This is used to
        convert the input data to the correct shape.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "capability:horizon": True,
        "capability:exogenous": False,
        "fit_is_empty": False,
        "y_inner_type": "np.ndarray",
    }

    def __init__(self, horizon: int, axis: int):
        self.horizon = horizon

        super().__init__(axis)

    @final
    def fit(self, y, exog=None, axis=1):
        """Fit forecaster to series y.

        Fit a forecaster to predict self.horizon steps ahead using y.

        Parameters
        ----------
        y : np.ndarray
            A time series on which to learn a forecaster to predict horizon ahead.
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y.

        Returns
        -------
        self
            Fitted BaseForecaster.
        """
        if self.get_tag("fit_is_empty"):
            self.is_fitted = True
            return self
        y, exog = self._preprocess_forecasting_input(y, exog, axis, True)
        self._fit(y, exog)
        self.is_fitted = True
        return self

    @final
    def predict(self, y, exog=None, axis=1) -> float:
        """Predict the next horizon steps ahead.

        Parameters
        ----------
        y : np.ndarray
            A time series to predict the next horizon value for.
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y.

        Returns
        -------
        float
            single prediction self.horizon steps ahead of y.
        """
        if not self.get_tag("fit_is_empty"):
            self._check_is_fitted()
        y, exog = self._preprocess_forecasting_input(y, exog, axis, False)
        return self._predict(y, exog)

    @final
    def forecast(self, y, exog=None, axis=1) -> float:
        """Forecast the next horizon steps ahead of ``y``.

        By default this is simply fit followed by predict.

        Parameters
        ----------
        y : np.ndarray
            A time series to predict the next horizon value for. Must be of shape
            ``(n_channels, n_timepoints)`` if a multivariate time series.
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y.

        Returns
        -------
        float
            single prediction self.horizon steps ahead of y.
        """
        y, exog = self._preprocess_forecasting_input(y, exog, axis, True)
        y_pred = self._forecast(y, exog)
        self.is_fitted = True
        return y_pred

    def _fit(self, y, exog):
        return self

    @abstractmethod
    def _predict(self, y, exog): ...

    def _forecast(self, y, exog):
        """Forecast values for time series X."""
        self._fit(y, exog)
        return self._predict(y, exog)

    def _preprocess_forecasting_input(self, y, exog, axis, store_meta):
        horizon = self.get_tag("capability:horizon")
        if not horizon and self.horizon > 1:
            raise ValueError(
                f"Horizon is set >1, but {self.__class__.__name__} cannot handle a "
                f"horizon greater than 1"
            )

        exog_tag = self.get_tag("capability:exogenous")
        if not exog_tag and exog is not None:
            raise ValueError(
                f"Exogenous variables passed but {self.__class__.__name__} cannot "
                "handle exogenous variables"
            )
        y = self._preprocess_series(y, axis, store_meta)

        if exog is not None:
            exog = self._convert_y(exog, self.axis)
        return y, exog

    def _convert_y(self, y: VALID_SERIES_INNER_TYPES, axis: int):
        """Convert y to self.get_tag("y_inner_type")."""
        if axis > 1 or axis < 0:
            raise ValueError(f"Input axis should be 0 or 1, saw {axis}")

        inner_type = self.get_tag("y_inner_type")
        if not isinstance(inner_type, list):
            inner_type = [inner_type]
        inner_names = [i.split(".")[-1] for i in inner_type]

        input = type(y).__name__
        if input not in inner_names:
            if inner_names[0] == "ndarray":
                y = y.to_numpy()
            elif inner_names[0] == "DataFrame":
                transpose = False
                if y.ndim == 1 and axis == 1:
                    transpose = True
                y = pd.DataFrame(y)
                if transpose:
                    y = y.T
            else:
                raise ValueError(
                    f"Unsupported inner type {inner_names[0]} derived from {inner_type}"
                )
        if y.ndim > 1 and self.axis != axis:
            y = y.T
        elif y.ndim == 1 and isinstance(y, np.ndarray):
            y = y[np.newaxis, :] if self.axis == 1 else y[:, np.newaxis]
        return y


class DirectForecastingMixin:
    """Mixin class for direct forecasting."""

    @final
    def direct_forecast(self, y, prediction_horizon, exog=None) -> np.ndarray:
        """
        Make ``prediction_horizon`` ahead forecasts using a fit for each horizon.

        This is commonly called the direct strategy. The forecaster is trained to
        predict one ahead, then retrained to fit two ahead etc. Not all forecasters
        are capable of being used with direct forecasting. The ability to
        forecast on horizons greater than 1 is indicated by the tag
        "capability:horizon". If this tag is false this function raises a value
        error. This method cannot be overridden.

        Parameters
        ----------
        y : np.ndarray
            The time series to make forecasts about. Must be of shape
            ``(n_channels, n_timepoints)`` if a multivariate time series.
        prediction_horizon : int
            The number of future time steps to forecast.
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y.

        Returns
        -------
        np.ndarray
            An array of shape `(prediction_horizon,)` containing the forecasts for
            each horizon.

        Raises
        ------
        ValueError
            if ``"capability:horizon`` is False or `prediction_horizon` less than 1.

        Examples
        --------
        >>> from aeon.forecasting import RegressionForecaster
        >>> y = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0])
        >>> f = RegressionForecaster(window=3)
        >>> f.direct_forecast(y,2)
        array([3., 2.])
        """
        horizon = self.get_tag("capability:horizon")
        if not horizon:
            raise ValueError(
                f"{self.__class__.__name__} cannot be used with the direct strategy "
                "because it cannot be trained with a horizon > 1."
            )
        if prediction_horizon < 1:
            raise ValueError(
                "The `prediction_horizon` must be greater than or equal to 1."
            )

        preds = np.zeros(prediction_horizon)
        for i in range(0, prediction_horizon):
            f = _clone_estimator(self)
            f.horizon = i + 1
            preds[i] = f.forecast(y, exog)
        return preds


class IterativeForecastingMixin:
    """Mixin class for iterative forecasting."""

    def iterative_forecast(self, y, prediction_horizon) -> np.ndarray:
        """
        Forecast ``prediction_horizon`` prediction using a single model fit on `y`.

        This function implements the iterative forecasting strategy (also called
        recursive or iterated). This involves a single model fit on ``y`` which is then
        used to make ``prediction_horizon`` ahead forecasts using its own predictions as
        inputs for future forecasts. This is done by taking the prediction at step
        ``i`` and feeding it back into the model to help predict for step ``i+1``.
        The basic contract of `iterative_forecast` is that `fit` is only ever called
        once.

        y : np.ndarray
            The time series to make forecasts about.  Must be of shape
            ``(n_channels, n_timepoints)`` if a multivariate time series.
        prediction_horizon : int
            The number of future time steps to forecast.

        Returns
        -------
        np.ndarray
            An array of shape `(prediction_horizon,)` containing the forecasts for
            each horizon.

        Raises
        ------
        ValueError
            if prediction_horizon` less than 1.

        Examples
        --------
        >>> from aeon.forecasting import RegressionForecaster
        >>> y = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0])
        >>> f = RegressionForecaster(window=3)
        >>> f.iterative_forecast(y,2)
        array([3., 2.])
        """
        if prediction_horizon < 1:
            raise ValueError(
                "The `prediction_horizon` must be greater than or equal to 1."
            )

        preds = np.zeros(prediction_horizon)
        self.fit(y)
        for i in range(0, prediction_horizon):
            preds[i] = self.predict(y)
            y = np.append(y, preds[i])
        return preds


class _SeriesToSeriesForecastingMixin:
    """Mixin class for series-to-series forecasting."""

    def series_to_series_forecast(self, y, prediction_horizon) -> np.ndarray:
        """Unimplemented."""
        pass
