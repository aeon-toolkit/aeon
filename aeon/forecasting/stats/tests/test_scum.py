"""Tests for the SCUM forecaster."""

import numpy as np
import pytest

import aeon.forecasting.stats._scum as scum_module
from aeon.forecasting import NaiveForecaster
from aeon.forecasting.ensembles import EnsembleForecaster
from aeon.forecasting.stats import SCUM
from aeon.forecasting.stats._scum import (
    _as_1d_float,
    _ConstantDummyForecaster,
    _RecentWindowForecaster,
    _validate_forecast_array,
)


@pytest.fixture
def naive_pool():
    """Four deterministic NaiveForecaster components shared across SCUM tests.

    Returns a fresh list on each use so tests never share fitted state.
    """
    return [
        ("last", NaiveForecaster(strategy="last")),
        ("mean", NaiveForecaster(strategy="mean")),
        ("seasonal2", NaiveForecaster(strategy="seasonal_last", seasonal_period=2)),
        ("seasonal3", NaiveForecaster(strategy="seasonal_last", seasonal_period=3)),
    ]


class _VectorForecaster(BaseForecaster, IterativeForecastingMixin):
    """Deterministic vector forecaster used in SCUM tests."""

    _tags = {
        "capability:horizon": False,
        "capability:exogenous": False,
        "fit_is_empty": True,
    }

    def __init__(self, values):
        self.values = values
        super().__init__(horizon=1, axis=1)

    def _predict(self, y, exog=None):
        return float(self.values[0])

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ):
        """Return the configured deterministic vector."""
        values = np.asarray(self.values, dtype=np.float64)
        return values[: int(prediction_horizon)]


class _LengthRecorderForecaster(BaseForecaster, IterativeForecastingMixin):
    """Return the length of the series seen during forecasting."""

    _tags = {
        "capability:horizon": False,
        "capability:exogenous": False,
        "fit_is_empty": True,
    }

    def __init__(self):
        super().__init__(horizon=1, axis=1)

    def _predict(self, y, exog=None):
        return float(np.asarray(y).size)

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ):
        """Return the number of observations supplied to the forecaster."""
        return np.full(int(prediction_horizon), float(np.asarray(y).size))


class _CountingSCUM(SCUM):
    """SCUM test double that counts ensemble fit/build calls."""

    def __init__(self, **kwargs):
        self.fit_predict_calls = 0
        super().__init__(**kwargs)

    def _fit_predict(self, y, prediction_horizon):
        """Count calls to the fit/predict helper."""
        self.fit_predict_calls += 1
        return super()._fit_predict(y, prediction_horizon)


def test_scum_median_combines_component_forecasts_by_horizon():
    """SCUM returns the per-horizon median across component forecasts."""
    y = np.array([1.0, 2.0, 3.0, 4.0])

    pred = SCUM(forecasters=naive_pool, clip_negative=False).iterative_forecast(
        y, prediction_horizon=3
    )

    np.testing.assert_allclose(pred, [2.75, 3.5, 3.5])


def test_scum_clips_negative_combined_forecasts(naive_pool):
    """SCUM clips negative median forecasts to zero by default."""
    y = np.array([-4.0, -3.0, -2.0, -1.0])

    pred = SCUM(forecasters=naive_pool).iterative_forecast(y, prediction_horizon=2)

    np.testing.assert_allclose(pred, [0.0, 0.0])


def test_scum_default_pool_uses_all_four_components(monkeypatch):
    """The default pool uses ETS, CES, ARIMA, and DOTM."""
    monkeypatch.setattr(
        scum_module,
        "AutoETS",
        lambda seasonal_period: NaiveForecaster(strategy="seasonal_last"),
    )
    monkeypatch.setattr(
        scum_module,
        "AutoCES",
        lambda season_length: NaiveForecaster(strategy="mean"),
    )
    monkeypatch.setattr(
        scum_module,
        "AutoARIMA",
        lambda: NaiveForecaster(strategy="last"),
    )
    monkeypatch.setattr(
        scum_module,
        "DOTM",
        lambda season_length: NaiveForecaster(
            strategy="seasonal_last", seasonal_period=2
        ),
    )
    y = np.array([1.0, 2.0, 3.0, 4.0])

    scum = SCUM(season_length=4)
    pred = scum.iterative_forecast(y, prediction_horizon=4)

    assert pred.shape == (4,)
    np.testing.assert_allclose(pred, [3.5, 4.0, 3.5, 4.0])
    assert np.all(pred >= 0.0)
    assert isinstance(scum.ensemble_, EnsembleForecaster)
    assert scum.ensemble_.averaging_method == "median"
    assert {name for name, _ in scum.forecasters_} == {"ets", "ces", "arima", "dotm"}
    assert len(scum.forecasters_) == 4
    assert isinstance(dict(scum.forecasters_)["dotm"], _RecentWindowForecaster)


def test_scum_predict_matches_horizon_one_iterative_forecast(naive_pool):
    """Stored forecast, predict, and horizon-one iterative forecast agree."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    scum = SCUM(forecasters=naive_pool, clip_negative=False).fit(y)

    assert scum.forecast_ == pytest.approx(2.75)
    assert scum.predict(y) == pytest.approx(2.75)
    np.testing.assert_allclose(scum.iterative_forecast(y, 1), [2.75])


def test_scum_predict_uses_fitted_ensemble_not_fit_predict_again():
    """SCUM.predict should use the fitted ensemble instead of refitting."""
    y = np.arange(10, dtype=np.float64)
    forecasters = [
        ("a", _VectorForecaster([1.0])),
        ("b", _VectorForecaster([2.0])),
        ("c", _VectorForecaster([3.0])),
        ("d", _VectorForecaster([4.0])),
    ]
    scum = _CountingSCUM(forecasters=forecasters, clip_negative=False)

    scum.fit(y)
    assert scum.fit_predict_calls == 1

    assert scum.predict(y) == pytest.approx(2.5)
    assert scum.fit_predict_calls == 1


def test_scum_predict_clips_negative_fitted_ensemble_prediction():
    """SCUM.predict should clip negative predictions from the fitted ensemble."""
    y = np.arange(10, dtype=np.float64)
    forecasters = [
        ("a", _VectorForecaster([-4.0])),
        ("b", _VectorForecaster([-3.0])),
        ("c", _VectorForecaster([-2.0])),
        ("d", _VectorForecaster([10.0])),
    ]
    scum = SCUM(forecasters=forecasters).fit(y)

    assert scum.predict(y) == 0.0


def test_scum_accepts_named_custom_pool():
    """SCUM accepts named custom forecaster pools."""
    y = np.arange(20, dtype=np.float64)
    forecasters = [
        ("a", _VectorForecaster([10.0])),
        ("b", _VectorForecaster([10.0])),
        ("c", _VectorForecaster([10.0])),
        ("dotm", _LengthRecorderForecaster()),
    ]

    with pytest.raises(NotImplementedError, match="does not support exogenous"):
        scum.iterative_forecast(y, 1, exog=y, future_exog=np.array([1.0]))


@pytest.mark.parametrize("season_length", [0, -1, True, 1.5])
def test_scum_invalid_season_length_raises(season_length):
    """SCUM validates the seasonal period."""
    with pytest.raises(ValueError, match="season_length"):
        SCUM(season_length=season_length).fit(np.arange(5, dtype=np.float64))


def test_scum_invalid_dotm_max_length_raises():
    """SCUM validates the DOTM window length."""
    with pytest.raises(ValueError, match="dotm_max_length"):
        SCUM(dotm_max_length=0).fit(np.arange(5, dtype=np.float64))


@pytest.mark.parametrize("dotm_max_length", [True, 1.5])
def test_scum_invalid_dotm_max_length_type_raises(dotm_max_length):
    """SCUM validates the DOTM window length type."""
    with pytest.raises(ValueError, match="dotm_max_length"):
        SCUM(dotm_max_length=dotm_max_length).fit(np.arange(5, dtype=np.float64))


def test_scum_rejects_short_or_non_finite_y():
    """SCUM's input guard requires at least two finite observations.

    Tested on the ``_as_1d_float`` guard directly: routing non-finite values
    through ``fit`` would first trip the base estimator's variance check and
    leak a numpy ``RuntimeWarning`` before reaching SCUM's own validation.
    """
    with pytest.raises(ValueError, match="at least two observations"):
        _as_1d_float(np.array([1.0]))

    with pytest.raises(ValueError, match="finite observations"):
        _as_1d_float(np.array([1.0, np.inf]))


def test_recent_window_forecaster_uses_tail_for_fit_and_iterative_forecast():
    """Recent-window wrapper should pass only the final max_length observations."""
    y = np.arange(10, dtype=np.float64)
    wrapper = _RecentWindowForecaster(NaiveForecaster(strategy="mean"), max_length=3)

    wrapper.fit(y)
    assert wrapper.predict(y) == 8.0
    np.testing.assert_allclose(wrapper.iterative_forecast(y, 2), [8.0, 8.0])


def test_recent_window_forecaster_rejects_exog():
    """Recent-window wrapper does not support exogenous variables."""
    y = np.arange(10, dtype=np.float64)
    wrapper = _RecentWindowForecaster(NaiveForecaster(strategy="mean"), max_length=3)

    with pytest.raises(NotImplementedError, match="does not support exog"):
        wrapper.iterative_forecast(y, 1, exog=y, future_exog=np.array([1.0]))


def test_validate_forecast_array_rejects_wrong_length_and_non_finite_values():
    """Forecast validation should enforce length and finite values."""
    model = NaiveForecaster(strategy="last")

    with pytest.raises(ValueError, match="returned 1 forecasts"):
        _validate_forecast_array([1.0], prediction_horizon=2, model=model)

    with pytest.raises(ValueError, match="non-finite forecasts"):
        _validate_forecast_array([1.0, np.nan], prediction_horizon=2, model=model)


def test_scum_get_test_params_pool_runs():
    """SCUM._get_test_params yields a constant-forecaster pool that runs."""
    params = SCUM._get_test_params()
    assert {name for name, _ in params["forecasters"]} == {"a", "b", "c", "d"}

    y = np.arange(5, dtype=np.float64)
    pred = SCUM(**params, clip_negative=False).iterative_forecast(
        y, prediction_horizon=2
    )

    # Median of constant components 1, 2, 3, 4 is 2.5 at every horizon.
    np.testing.assert_allclose(pred, [2.5, 2.5])


def test_constant_dummy_forecaster_predict_and_iterative_forecast():
    """The constant dummy supports both single-step and multi-step paths."""
    f = _ConstantDummyForecaster(3.0)
    y = np.arange(4, dtype=np.float64)

    assert f.fit(y).predict(y) == 3.0
    np.testing.assert_allclose(f.iterative_forecast(y, 3), [3.0, 3.0, 3.0])
