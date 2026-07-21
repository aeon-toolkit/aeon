"""BaseForecaster class.

A simplified first base class for forecasting models.

"""

__maintainer__ = ["TonyBagnall"]
__all__ = [
    "BaseForecaster",
    "DirectForecastingMixin",
    "IterativeForecastingMixin",
    "SeriesToSeriesForecastingMixin",
]

from abc import ABC, abstractmethod
from typing import final

import numpy as np
import pandas as pd

from aeon.base import BaseSeriesEstimator
from aeon.base._base import _clone_estimator
from aeon.utils.data_types import VALID_SERIES_INNER_TYPES
from aeon.utils.decorators.method_timer import method_timer


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
    @method_timer("fit_time_millis_", overwrite=False, remove_on_start=True)
    def fit(self, y, exog=None, axis=1):
        """Fit forecaster to series y.

        Fit a forecaster to predict self.horizon steps ahead using y.

        Parameters
        ----------
        y : np.ndarray
            A time series on which to learn a forecaster to predict horizon ahead.
        exog : np.ndarray, default =None
            Optional target-time exogenous values aligned with ``y``.

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
            Optional exogenous values for the prediction target. For models fitted
            with exogenous variables, this should contain the exogenous values needed
            to make the next prediction.

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
            Optional target-time exogenous values aligned with ``y``.

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

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ) -> np.ndarray:
        """
        Forecast ``prediction_horizon`` steps using a single model fit on ``y``.

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
        exog : np.ndarray or None, default=None
            Target-time exogenous training data aligned with ``y``. If provided,
            ``future_exog`` must also be provided.
        future_exog : np.ndarray or None, default=None
            Target-time future exogenous data aligned with the forecast horizon. If
            provided, ``exog`` must also be provided. These values are passed one row
            at a time to ``predict`` and are not concatenated onto ``exog``.

        Returns
        -------
        np.ndarray
            An array of shape `(prediction_horizon,)` containing the forecasts for
            each horizon.

        Raises
        ------
        ValueError
            If ``prediction_horizon`` is less than 1.
        ValueError
            If only one of ``exog`` and ``future_exog`` is provided.
        ValueError
            If ``exog`` is not aligned with ``y``.
        ValueError
            If ``future_exog`` is not aligned with ``prediction_horizon``.


        Examples
        --------
        >>> from aeon.forecasting import RegressionForecaster
        >>> y = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0])
        >>> f = RegressionForecaster(window=3)
        >>> f.iterative_forecast(y, 2)
        array([3., 2.])
        """
        y, exog, future_exog = self._check_iterative_forecast_inputs(
            y, prediction_horizon, exog, future_exog
        )
        preds = np.zeros(prediction_horizon)
        self.fit(y, exog=exog)
        for i in range(prediction_horizon):
            step_exog = None
            if future_exog is not None:
                step_exog = future_exog[i : i + 1]
            preds[i] = self.predict(y, exog=step_exog)
            y = np.append(y, preds[i])
        return preds

    @staticmethod
    def _check_iterative_forecast_inputs(
        y,
        prediction_horizon,
        exog=None,
        future_exog=None,
    ):
        """Validate inputs for iterative forecasting."""
        if isinstance(prediction_horizon, bool) or not isinstance(
            prediction_horizon, (int, np.integer)
        ):
            raise TypeError(
                "prediction_horizon must be an integer. If you intended to pass "
                "future exogenous values, use future_exog=... and also provide "
                "prediction_horizon."
            )

        if prediction_horizon < 1:
            raise ValueError(
                "The `prediction_horizon` must be greater than or equal to 1."
            )

        if (exog is None) != (future_exog is None):
            raise ValueError(
                "exog and future_exog must be provided together. "
                "exog must be aligned with y, and future_exog must be aligned "
                "with prediction_horizon."
            )

        y = np.asarray(y)

        if y.ndim == 0:
            raise ValueError("y must be at least one-dimensional.")

        if exog is None:
            return y, None, None

        exog = np.asarray(exog)
        future_exog = np.asarray(future_exog)

        if exog.ndim not in (1, 2):
            raise ValueError("exog must be a 1D or 2D array.")

        if future_exog.ndim not in (1, 2):
            raise ValueError("future_exog must be a 1D or 2D array.")

        n_timepoints = y.shape[-1]

        if exog.shape[0] != n_timepoints:
            raise ValueError(
                "exog must contain one row per time point in y. "
                f"Got {exog.shape[0]} rows, expected {n_timepoints}."
            )

        if future_exog.shape[0] != prediction_horizon:
            raise ValueError(
                "future_exog must contain one row per forecast horizon step. "
                f"Got {future_exog.shape[0]} rows, expected {prediction_horizon}."
            )

        exog_n_features = 1 if exog.ndim == 1 else exog.shape[1]
        future_exog_n_features = 1 if future_exog.ndim == 1 else future_exog.shape[1]

        if exog_n_features != future_exog_n_features:
            raise ValueError(
                "exog and future_exog must have the same number of features. "
                f"Got {exog_n_features} and {future_exog_n_features}."
            )

        return y, exog, future_exog


class SeriesToSeriesForecastingMixin(ABC):
    """Mixin class for series-to-series forecasting."""

    @final
    def series_to_series_forecast(self, y, prediction_horizon, exog=None) -> np.ndarray:
        """
        Forecast ``prediction_horizon`` using a series iterative approach.

        This function implements a series-to-series forecasting strategy.
        The forecaster is trained to predict multiple steps ahead in one go,
        returning a series of predictions. This is done by fitting the model
        once and then predicting a series of length `prediction_horizon`.

        y : np.ndarray
            The time series to make forecasts about.  Must be of shape
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
            if prediction_horizon less than 1.

        """
        if prediction_horizon < 1:
            raise ValueError(
                "The `prediction_horizon` must be greater than or equal to 1."
            )

        return self._series_to_series_forecast(y, prediction_horizon, exog)

    @abstractmethod
    def _series_to_series_forecast(self, y, prediction_horizon, exog): ...
