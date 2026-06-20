"""Simple Combination of Univariate Models (SCUM) forecaster."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["SCUM"]

from copy import deepcopy

import numpy as np

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin
from aeon.forecasting.ensembles import EnsembleForecaster
from aeon.forecasting.stats._arima import AutoARIMA
from aeon.forecasting.stats._ces import AutoCES
from aeon.forecasting.stats._dotm import DOTM
from aeon.forecasting.stats._ets import AutoETS


class SCUM(BaseForecaster, IterativeForecastingMixin):
    """Simple Combination of Univariate Models forecaster.

    SCUM is the M4 submission by Petropoulos and Svetunkov [1]_. It combines
    point forecasts from four univariate models by taking the median at each
    horizon: automatic exponential smoothing (ETS), complex exponential
    smoothing (CES), automatic ARIMA, and dynamic optimized theta (DOTM).

    The forecast combination is delegated to
    :class:`aeon.forecasting.ensembles.EnsembleForecaster`, with SCUM defining
    the default component pool and applying the non-negative clipping used in
    the M4 implementation.

    Parameters
    ----------
    season_length : int, default=1
        Seasonal period/frequency passed to seasonal-capable component models.
    forecasters : sequence or None, default=None
        Optional custom forecaster pool. Each entry can be either a forecaster
        instance or a ``(name, forecaster)`` tuple. If ``None``, the SCUM pool
        ``(AutoETS, AutoCES, AutoARIMA, DOTM)`` is used.
    clip_negative : bool, default=True
        If ``True``, replace negative combined forecasts with zero, following
        the M4 implementation for non-negative data.
    dotm_max_length : int or None, default=5000
        Maximum number of most recent observations passed to DOTM. The paper
        applies DOTM only to the last 5000 observations for very long series.
        Set to ``None`` to disable this window.

    Attributes
    ----------
    ensemble_ : EnsembleForecaster
        Fitted median ensemble used to combine the component forecasts.
    forecasters_ : list of tuple
        Fitted ``(name, forecaster)`` pool used by SCUM.
    forecast_ : float
        Stored one-step-ahead combined forecast.

    References
    ----------
    .. [1] Petropoulos, F. and Svetunkov, I. (2020). A simple combination of
       univariate models. International Journal of Forecasting, 36(1), 110-115.
    """

    _tags = {
        "capability:horizon": False,
        "capability:exogenous": False,
    }

    def __init__(
        self,
        season_length=1,
        forecasters=None,
        clip_negative=True,
        dotm_max_length=5000,
    ):
        self.season_length = season_length
        self.forecasters = forecasters
        self.clip_negative = clip_negative
        self.dotm_max_length = dotm_max_length
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        """Fit SCUM and store the one-step-ahead combined forecast."""
        if exog is not None:
            raise NotImplementedError("SCUM does not support exogenous variables.")
        self._validate_parameters()
        y = _as_1d_float(y)
        forecasts, ensemble = self._fit_predict(y, 1)
        self.forecast_ = float(forecasts[0])
        self.ensemble_ = ensemble
        self.forecasters_ = ensemble.forecasters_
        self._y_ = y.copy()
        return self

    def _predict(self, y, exog=None):
        """Predict one step ahead from the supplied context."""
        if exog is not None:
            raise NotImplementedError("SCUM does not support exogenous variables.")
        self._validate_parameters()
        y = _as_1d_float(y)
        forecasts, _ = self._fit_predict(y, 1)
        return float(forecasts[0])

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ):
        """Fit SCUM once and return median-combined multi-step forecasts."""
        if exog is not None or future_exog is not None:
            raise NotImplementedError("SCUM does not support exogenous variables.")
        self._validate_parameters()
        _validate_prediction_horizon(prediction_horizon)

        y = _as_1d_float(y)
        forecasts, ensemble = self._fit_predict(y, int(prediction_horizon))
        self.forecast_ = float(forecasts[0])
        self.ensemble_ = ensemble
        self.forecasters_ = ensemble.forecasters_
        self._y_ = y.copy()
        self.is_fitted = True
        return forecasts

    def _fit_predict(self, y, prediction_horizon):
        """Fit the SCUM ensemble and return combined forecasts."""
        ensemble = EnsembleForecaster(
            forecasters=self._build_forecaster_pool(),
            averaging_method="median",
            iterative_strategy="component",
        )
        forecasts = np.asarray(
            ensemble.iterative_forecast(y, int(prediction_horizon)),
            dtype=np.float64,
        ).reshape(-1)
        if self.clip_negative:
            forecasts = np.maximum(forecasts, 0.0)
        return forecasts, ensemble

    def _build_forecaster_pool(self):
        """Build the default SCUM model pool or return the supplied pool."""
        if self.forecasters is not None:
            return self.forecasters

        forecasters = [
            (
                "ets",
                _FitBeforeIterativeForecaster(
                    AutoETS(seasonal_period=int(self.season_length))
                ),
            ),
            ("ces", AutoCES(season_length=int(self.season_length))),
            ("arima", AutoARIMA()),
            ("dotm", DOTM(season_length=int(self.season_length))),
        ]
        if self.dotm_max_length is not None:
            forecasters[-1] = (
                "dotm",
                _RecentWindowForecaster(
                    forecasters[-1][1], max_length=int(self.dotm_max_length)
                ),
            )
        return forecasters

    def _validate_parameters(self):
        """Validate SCUM configuration."""
        if (
            isinstance(self.season_length, bool)
            or not isinstance(self.season_length, (int, np.integer))
            or self.season_length < 1
        ):
            raise ValueError("season_length must be a positive integer.")
        if self.dotm_max_length is not None and (
            isinstance(self.dotm_max_length, bool)
            or not isinstance(self.dotm_max_length, (int, np.integer))
            or self.dotm_max_length < 1
        ):
            raise ValueError("dotm_max_length must be a positive integer or None.")

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings."""
        return {
            "forecasters": [
                ("a", _ConstantDummyForecaster(1.0)),
                ("b", _ConstantDummyForecaster(2.0)),
                ("c", _ConstantDummyForecaster(3.0)),
                ("d", _ConstantDummyForecaster(4.0)),
            ]
        }


class _FitBeforeIterativeForecaster(BaseForecaster, IterativeForecastingMixin):
    """Fit a wrapped forecaster before calling its iterative forecast."""

    _tags = {
        "capability:horizon": False,
        "capability:exogenous": False,
    }

    def __init__(self, forecaster):
        self.forecaster = forecaster
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        if exog is not None:
            raise NotImplementedError("Fit-before wrapper does not support exog.")
        y = _as_1d_float(y)
        self.forecaster_ = deepcopy(self.forecaster)
        self.forecaster_.fit(y)
        return self

    def _predict(self, y, exog=None):
        if exog is not None:
            raise NotImplementedError("Fit-before wrapper does not support exog.")
        return float(self.forecaster_.predict(_as_1d_float(y)))

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ):
        """Fit the wrapped forecaster, then delegate iterative forecasting."""
        if exog is not None or future_exog is not None:
            raise NotImplementedError("Fit-before wrapper does not support exog.")
        _validate_prediction_horizon(prediction_horizon)
        y = _as_1d_float(y)
        self.forecaster_ = deepcopy(self.forecaster)
        self.forecaster_.fit(y)
        forecasts = self.forecaster_.iterative_forecast(y, int(prediction_horizon))
        self.is_fitted = True
        return _validate_forecast_array(
            forecasts, int(prediction_horizon), self.forecaster_
        )


class _RecentWindowForecaster(BaseForecaster, IterativeForecastingMixin):
    """Limit a wrapped forecaster to the most recent observations."""

    _tags = {
        "capability:horizon": False,
        "capability:exogenous": False,
    }

    def __init__(self, forecaster, max_length=5000):
        self.forecaster = forecaster
        self.max_length = max_length
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        if exog is not None:
            raise NotImplementedError("Recent-window wrapper does not support exog.")
        y = _tail(_as_1d_float(y), self.max_length)
        self.forecaster_ = deepcopy(self.forecaster)
        self.forecaster_.fit(y)
        return self

    def _predict(self, y, exog=None):
        if exog is not None:
            raise NotImplementedError("Recent-window wrapper does not support exog.")
        return float(self.forecaster_.predict(_tail(y, self.max_length)))

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ):
        """Fit the wrapped forecaster on the recent window and forecast."""
        if exog is not None or future_exog is not None:
            raise NotImplementedError("Recent-window wrapper does not support exog.")
        _validate_prediction_horizon(prediction_horizon)
        y = _tail(_as_1d_float(y), self.max_length)
        self.forecaster_ = deepcopy(self.forecaster)
        forecasts = self.forecaster_.iterative_forecast(y, int(prediction_horizon))
        self.is_fitted = True
        return _validate_forecast_array(
            forecasts, int(prediction_horizon), self.forecaster_
        )


class _ConstantDummyForecaster(BaseForecaster, IterativeForecastingMixin):
    """Small deterministic forecaster used only by estimator tests."""

    _tags = {
        "capability:horizon": False,
        "capability:exogenous": False,
        "fit_is_empty": True,
    }

    def __init__(self, value=1.0):
        self.value = value
        super().__init__(horizon=1, axis=1)

    def _predict(self, y, exog=None):
        return float(self.value)

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ):
        """Return a constant forecast for every horizon."""
        _validate_prediction_horizon(prediction_horizon)
        return np.full(int(prediction_horizon), float(self.value))


def _validate_prediction_horizon(prediction_horizon):
    """Validate an iterative forecast horizon."""
    if isinstance(prediction_horizon, bool) or not isinstance(
        prediction_horizon, (int, np.integer)
    ):
        raise TypeError("prediction_horizon must be an integer.")
    if prediction_horizon < 1:
        raise ValueError("prediction_horizon must be greater than or equal to 1.")


def _validate_forecast_array(forecasts, prediction_horizon, model):
    """Return a finite 1D forecast array of the expected length."""
    forecasts = np.asarray(forecasts, dtype=np.float64).reshape(-1)
    if forecasts.shape[0] != prediction_horizon:
        raise ValueError(
            f"{model.__class__.__name__} returned {forecasts.shape[0]} forecasts; "
            f"expected {prediction_horizon}."
        )
    if not np.all(np.isfinite(forecasts)):
        raise ValueError(f"{model.__class__.__name__} returned non-finite forecasts.")
    return forecasts


def _as_1d_float(y):
    """Convert a forecasting target to a finite 1D float array."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.shape[0] < 2:
        raise ValueError("SCUM requires at least two observations.")
    if not np.all(np.isfinite(y)):
        raise ValueError("SCUM requires finite observations.")
    return y


def _tail(y, max_length):
    """Return the most recent ``max_length`` values."""
    y = _as_1d_float(y)
    return y[-int(max_length) :]
