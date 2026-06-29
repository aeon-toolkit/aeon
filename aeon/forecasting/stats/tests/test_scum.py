"""Tests for the SCUM forecaster."""

import numpy as np
import pytest

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin
from aeon.forecasting.ensembles import EnsembleForecaster
from aeon.forecasting.stats import SCUM


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
    y = np.arange(10, dtype=np.float64)
    forecasters = [
        ("a", _VectorForecaster([1.0, 10.0, 100.0])),
        ("b", _VectorForecaster([2.0, 20.0, 200.0])),
        ("c", _VectorForecaster([3.0, 30.0, 300.0])),
        ("d", _VectorForecaster([4.0, 40.0, 400.0])),
    ]

    pred = SCUM(forecasters=forecasters, clip_negative=False).iterative_forecast(
        y, prediction_horizon=3
    )

    np.testing.assert_allclose(pred, [2.5, 25.0, 250.0])


def test_scum_clips_negative_combined_forecasts():
    """SCUM clips negative median forecasts to zero by default."""
    y = np.arange(10, dtype=np.float64)
    forecasters = [
        ("a", _VectorForecaster([-4.0, -1.0])),
        ("b", _VectorForecaster([-3.0, -2.0])),
        ("c", _VectorForecaster([-2.0, -3.0])),
        ("d", _VectorForecaster([10.0, -4.0])),
    ]

    pred = SCUM(forecasters=forecasters).iterative_forecast(y, prediction_horizon=2)

    np.testing.assert_allclose(pred, [0.0, 0.0])


def test_scum_default_pool_uses_all_four_components():
    """The default pool uses ETS, CES, ARIMA, and DOTM."""
    y = np.linspace(1.0, 20.0, 30)

    scum = SCUM(season_length=4)
    pred = scum.iterative_forecast(y, prediction_horizon=4)

    assert pred.shape == (4,)
    assert np.all(np.isfinite(pred))
    assert np.all(pred >= 0.0)
    assert isinstance(scum.ensemble_, EnsembleForecaster)
    assert scum.ensemble_.averaging_method == "median"
    assert {name for name, _ in scum.forecasters_} == {"ets", "ces", "arima", "dotm"}
    assert len(scum.forecasters_) == 4


def test_scum_predict_matches_horizon_one_iterative_forecast():
    """Stored forecast, predict, and horizon-one iterative forecast agree."""
    y = np.arange(10, dtype=np.float64)
    forecasters = [
        ("a", _VectorForecaster([1.0])),
        ("b", _VectorForecaster([2.0])),
        ("c", _VectorForecaster([3.0])),
        ("d", _VectorForecaster([4.0])),
    ]
    scum = SCUM(forecasters=forecasters, clip_negative=False).fit(y)

    assert scum.forecast_ == pytest.approx(2.5)
    assert scum.predict(y) == pytest.approx(2.5)
    np.testing.assert_allclose(scum.iterative_forecast(y, 1), [2.5])


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

    pred = SCUM(
        forecasters=forecasters,
        clip_negative=False,
        dotm_max_length=None,
    ).iterative_forecast(y, 1)

    np.testing.assert_allclose(pred, [10.0])


def test_scum_invalid_dotm_max_length_raises():
    """SCUM validates the DOTM window length."""
    with pytest.raises(ValueError, match="dotm_max_length"):
        SCUM(dotm_max_length=0).fit(np.arange(5, dtype=np.float64))
