"""Simple Combination of Univariate Models (SCUM) forecaster."""

__maintainer__ = []
__all__ = ["SCUM"]

from copy import deepcopy

import numpy as np

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin
from aeon.forecasting.stats._arima import AutoARIMA
from aeon.forecasting.stats._ets import AutoETS


class SCUM(BaseForecaster, IterativeForecastingMixin):
    """Simple Combination of Univariate Models forecaster.

    SCUM is the M4 submission by Petropoulos and Svetunkov [1]_. It combines
    point forecasts from four univariate models by taking the median at each
    horizon: automatic exponential smoothing (ETS), complex exponential
    smoothing (CES), automatic ARIMA, and dynamic optimized theta (DOTM).

    This implementation is intentionally wired for the forecasting ensemble API
    from PR #3452. While that API is not available on this branch, a private
    compatibility median ensemble is used so the class remains importable and
    testable. Once the ensemble PR is merged, the public ensemble will be used
    automatically.

    Parameters
    ----------
    season_length : int, default=1
        Seasonal period/frequency passed to CES and DOTM when available.
    forecasters : sequence or None, default=None
        Optional custom forecaster pool. Each entry can be either a forecaster
        instance or a ``(name, forecaster)`` tuple. If ``None``, the SCUM pool
        ``(AutoETS, AutoCES, AutoARIMA, DOTM)`` is used. CES and DOTM fall back
        to lightweight dummy forecasters on this branch until their PRs merge.
    clip_negative : bool, default=True
        If ``True``, replace negative combined forecasts with zero, following
        the M4 implementation for non-negative data.
    dotm_max_length : int or None, default=5000
        Maximum number of most recent observations passed to DOTM. The paper
        applies DOTM only to the last 5000 observations for very long series.
        Set to ``None`` to disable this window.
    error_policy : {"raise", "ignore"}, default="raise"
        How to handle individual model failures in the local compatibility
        ensemble. ``"ignore"`` computes the median over successful models and
        raises only if all models fail.

    Attributes
    ----------
    forecasters_ : list of tuple
        Fitted or configured ``(name, forecaster)`` pool used by SCUM.
    ensemble_ : object
        Forecasting ensemble instance. This is the PR #3452 ensemble when
        available, otherwise the private compatibility ensemble.
    forecast_ : float
        Stored one-step-ahead combined forecast.
    component_forecasts_ : dict
        Individual one-step forecasts from each successful component model.

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
        error_policy="raise",
    ):
        self.season_length = season_length
        self.forecasters = forecasters
        self.clip_negative = clip_negative
        self.dotm_max_length = dotm_max_length
        self.error_policy = error_policy
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        """Fit SCUM and store the one-step-ahead combined forecast."""
        if exog is not None:
            raise NotImplementedError("SCUM does not support exogenous variables.")
        self._validate_parameters()
        y = _as_1d_float(y)
        forecasts, component_forecasts, ensemble = self._fit_predict(y, 1)
        self.forecast_ = float(forecasts[0])
        self.component_forecasts_ = {
            name: float(values[0]) for name, values in component_forecasts.items()
        }
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
        forecasts, component_forecasts, ensemble = self._fit_predict(y, 1)
        self.component_forecasts_ = {
            name: float(values[0]) for name, values in component_forecasts.items()
        }
        self.ensemble_ = ensemble
        self.forecasters_ = ensemble.forecasters_
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
        if isinstance(prediction_horizon, bool) or not isinstance(
            prediction_horizon, (int, np.integer)
        ):
            raise TypeError("prediction_horizon must be an integer.")
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be greater than or equal to 1.")

        y = _as_1d_float(y)
        forecasts, component_forecasts, ensemble = self._fit_predict(
            y, int(prediction_horizon)
        )
        self.forecast_ = float(forecasts[0])
        self.component_forecasts_ = component_forecasts
        self.ensemble_ = ensemble
        self.forecasters_ = ensemble.forecasters_
        self._y_ = y.copy()
        self.is_fitted = True
        return forecasts

    def _fit_predict(self, y, prediction_horizon):
        """Fit the SCUM ensemble and return combined/component forecasts."""
        forecasters = self._build_forecaster_pool()
        ensemble = _make_forecasting_ensemble(
            forecasters=forecasters,
            clip_negative=bool(self.clip_negative),
            error_policy=self.error_policy,
        )
        forecasts = _forecast_with_ensemble(ensemble, y, prediction_horizon)
        if self.clip_negative:
            forecasts = np.maximum(forecasts, 0.0)
        component_forecasts = getattr(ensemble, "component_forecasts_", {})
        if not hasattr(ensemble, "forecasters_"):
            ensemble.forecasters_ = forecasters
        return forecasts, component_forecasts, ensemble

    def _build_forecaster_pool(self):
        """Build the default SCUM model pool or normalise the supplied pool."""
        if self.forecasters is not None:
            return _normalise_forecaster_pool(self.forecasters)

        forecasters = [
            ("ets", AutoETS()),
            ("ces", _make_ces_forecaster(self.season_length)),
            ("arima", AutoARIMA()),
            ("dotm", _make_dotm_forecaster(self.season_length)),
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
        if self.error_policy not in ("raise", "ignore"):
            raise ValueError("error_policy must be 'raise' or 'ignore'.")
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


class _MedianForecastingEnsemble:
    """Private compatibility ensemble until PR #3452 is merged."""

    def __init__(self, forecasters, clip_negative=True, error_policy="raise"):
        self.forecasters = forecasters
        self.clip_negative = clip_negative
        self.error_policy = error_policy
        self.forecasters_ = []
        self.component_forecasts_ = {}

    def iterative_forecast(self, y, prediction_horizon):
        """Return median forecast across successful component forecasters."""
        y = _as_1d_float(y)
        forecasts = []
        self.forecasters_ = []
        self.component_forecasts_ = {}
        errors = {}
        for name, forecaster in self.forecasters:
            model = deepcopy(forecaster)
            try:
                pred = _forecast_with_model(model, y, prediction_horizon)
            except Exception as exc:  # noqa: BLE001
                if self.error_policy == "raise":
                    raise RuntimeError(
                        f"SCUM component forecaster {name!r} failed."
                    ) from exc
                errors[name] = exc
                continue
            forecasts.append(pred)
            self.forecasters_.append((name, model))
            self.component_forecasts_[name] = pred

        if not forecasts:
            raise RuntimeError(f"All SCUM component forecasters failed: {errors!r}.")
        combined = np.median(np.vstack(forecasts), axis=0)
        if self.clip_negative:
            combined = np.maximum(combined, 0.0)
        return np.asarray(combined, dtype=np.float64)


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
        forecast = _forecast_with_model(self.forecaster_, _tail(y, self.max_length), 1)
        return float(forecast[0])

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
        y = _tail(_as_1d_float(y), self.max_length)
        model = deepcopy(self.forecaster)
        return _forecast_with_model(model, y, prediction_horizon)


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
        return np.full(int(prediction_horizon), float(self.value))


class _DummyCESForecaster(BaseForecaster, IterativeForecastingMixin):
    """Temporary CES stand-in until PR #3463 lands."""

    _tags = {
        "capability:horizon": False,
        "capability:exogenous": False,
    }

    def __init__(self, season_length=1):
        self.season_length = season_length
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        y = _as_1d_float(y)
        self.forecast_ = float(_seasonal_last_forecast(y, 1, self.season_length)[0])
        return self

    def _predict(self, y, exog=None):
        return float(_seasonal_last_forecast(_as_1d_float(y), 1, self.season_length)[0])

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ):
        """Return a seasonal-naive dummy forecast."""
        return _seasonal_last_forecast(
            _as_1d_float(y), int(prediction_horizon), self.season_length
        )


class _DummyDOTMForecaster(BaseForecaster, IterativeForecastingMixin):
    """Temporary DOTM stand-in until PR #3455 lands."""

    _tags = {
        "capability:horizon": False,
        "capability:exogenous": False,
    }

    def __init__(self, season_length=1):
        self.season_length = season_length
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        y = _as_1d_float(y)
        self.forecast_ = float(_linear_trend_forecast(y, 1)[0])
        return self

    def _predict(self, y, exog=None):
        return float(_linear_trend_forecast(_as_1d_float(y), 1)[0])

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ):
        """Return a linear-trend dummy forecast."""
        return _linear_trend_forecast(_as_1d_float(y), int(prediction_horizon))


def _make_ces_forecaster(season_length):
    """Return real AutoCES when available, otherwise a dummy CES forecaster."""
    try:
        from aeon.forecasting.stats import AutoCES

        return AutoCES(season_length=season_length)
    except ImportError:
        try:
            from aeon.forecasting.stats import CES

            return CES(season_length=season_length)
        except ImportError:
            return _DummyCESForecaster(season_length=season_length)


def _make_dotm_forecaster(season_length):
    """Return real DOTM when available, otherwise a dummy DOTM forecaster."""
    try:
        from aeon.forecasting.stats import DOTM

        return DOTM(season_length=season_length)
    except ImportError:
        return _DummyDOTMForecaster(season_length=season_length)


def _make_forecasting_ensemble(forecasters, clip_negative, error_policy):
    """Create the intended PR #3452 ensemble, falling back to a local shim."""
    ensemble_cls = _get_forecasting_ensemble_class()
    if ensemble_cls is None:
        return _MedianForecastingEnsemble(
            forecasters=forecasters,
            clip_negative=clip_negative,
            error_policy=error_policy,
        )

    for kwargs in (
        {
            "forecasters": forecasters,
            "combination": "median",
            "clip_negative": clip_negative,
        },
        {
            "estimators": forecasters,
            "combination": "median",
            "clip_negative": clip_negative,
        },
        {"forecasters": forecasters, "aggfunc": "median"},
        {"estimators": forecasters, "aggfunc": "median"},
    ):
        try:
            return ensemble_cls(**kwargs)
        except TypeError:
            continue
    return _MedianForecastingEnsemble(
        forecasters=forecasters,
        clip_negative=clip_negative,
        error_policy=error_policy,
    )


def _get_forecasting_ensemble_class():
    """Look up the forecasting ensemble planned in PR #3452."""
    candidates = (
        ("aeon.forecasting.ensemble", "ForecastingEnsemble"),
        ("aeon.forecasting.ensemble", "ForecastingEnsembleForecaster"),
        ("aeon.forecasting.compose", "ForecastingEnsemble"),
        ("aeon.forecasting.compose", "ForecastingEnsembleForecaster"),
    )
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
    return None


def _forecast_with_model(model, y, prediction_horizon):
    """Fit a model and return a 1D forecast array."""
    y = _as_1d_float(y)
    h = int(prediction_horizon)
    if h < 1:
        raise ValueError("prediction_horizon must be greater than or equal to 1.")
    if hasattr(model, "iterative_forecast"):
        pred = model.iterative_forecast(y, h)
    else:
        model.fit(y)
        if h == 1:
            pred = np.asarray([model.predict(y)], dtype=np.float64)
        else:
            raise ValueError(
                f"{model.__class__.__name__} cannot produce multi-step forecasts."
            )
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if pred.shape[0] != h:
        raise ValueError(
            f"{model.__class__.__name__} returned {pred.shape[0]} forecasts; "
            f"expected {h}."
        )
    if not np.all(np.isfinite(pred)):
        raise ValueError(f"{model.__class__.__name__} returned non-finite forecasts.")
    return pred


def _forecast_with_ensemble(ensemble, y, prediction_horizon):
    """Return a 1D forecast array from a forecasting ensemble."""
    y = _as_1d_float(y)
    h = int(prediction_horizon)
    if hasattr(ensemble, "iterative_forecast"):
        pred = ensemble.iterative_forecast(y, h)
    elif hasattr(ensemble, "fit") and hasattr(ensemble, "predict"):
        ensemble.fit(y)
        if h != 1:
            raise ValueError(
                f"{ensemble.__class__.__name__} cannot produce multi-step forecasts."
            )
        pred = np.asarray([ensemble.predict(y)], dtype=np.float64)
    else:
        raise ValueError(
            f"{ensemble.__class__.__name__} does not expose a forecasting method."
        )
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if pred.shape[0] != h:
        raise ValueError(
            f"{ensemble.__class__.__name__} returned {pred.shape[0]} forecasts; "
            f"expected {h}."
        )
    if not np.all(np.isfinite(pred)):
        raise ValueError(
            f"{ensemble.__class__.__name__} returned non-finite forecasts."
        )
    return pred


def _normalise_forecaster_pool(forecasters):
    """Normalise custom forecaster pool to ``(name, forecaster)`` tuples."""
    normalised = []
    for i, item in enumerate(forecasters):
        if isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError("Forecaster tuples must be (name, forecaster).")
            name, forecaster = item
        else:
            name, forecaster = f"model_{i}", item
        if not isinstance(name, str):
            raise ValueError("Forecaster names must be strings.")
        normalised.append((name, forecaster))
    if not normalised:
        raise ValueError("forecasters must contain at least one model.")
    return normalised


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


def _seasonal_last_forecast(y, prediction_horizon, season_length):
    """Return seasonal-naive forecasts."""
    y = _as_1d_float(y)
    h = int(prediction_horizon)
    m = max(1, int(season_length))
    period = y[-m:] if y.shape[0] >= m else y
    return np.asarray([period[i % period.shape[0]] for i in range(h)])


def _linear_trend_forecast(y, prediction_horizon):
    """Return a simple OLS linear-trend forecast."""
    y = _as_1d_float(y)
    h = int(prediction_horizon)
    n = y.shape[0]
    x = np.arange(n, dtype=np.float64)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    slope = (
        0.0 if denom <= 1e-12 else float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    )
    intercept = y_mean - slope * x_mean
    future_x = np.arange(n, n + h, dtype=np.float64)
    return intercept + slope * future_x
