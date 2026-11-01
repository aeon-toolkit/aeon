"""Tests for the FourTheta forecaster.

Reference values come from the authors' code (github.com/vangspiliot/AutoTheta)
run in R 4.5.3 with forecast 9.0.2 on the airline series.
"""

import numpy as np
import pytest

from aeon.datasets import load_airline
from aeon.forecasting.stats import FourTheta
from aeon.forecasting.stats._four_theta import (
    _ADD,
    _EXPONENTIAL,
    _LINEAR,
    _MULT,
    _MULTIPLICATIVE_FLOOR,
    _NONE,
    _fit_candidate,
    _fit_ses,
    _project,
    _ses_sse,
)
from aeon.forecasting.utils._seasonality import (
    _seasonal_decompose_numba,
    _seasonal_forecast,
)

AIRLINE = np.asarray(load_airline(), dtype=np.float64).squeeze()

# Theta.models.fit(y / mean(y), fh=3, theta=1.7): three forecasts and in-sample MAE.
R_CANDIDATES = {
    (_NONE, _MULT, _EXPONENTIAL): ([1.54758959, 1.55400611, 1.56044924], 0.09138626),
    (_NONE, _ADD, _LINEAR): ([1.54510236, 1.54900582, 1.55290928], 0.09127768),
    (_NONE, _ADD, _EXPONENTIAL): ([1.54896640, 1.55681209, 1.56473702], 0.09119796),
    (_MULT, _MULT, _EXPONENTIAL): ([1.57563188, 1.53592698, 1.75828312], 0.02715278),
    (_MULT, _ADD, _LINEAR): ([1.57159416, 1.52909303, 1.74713975], 0.02720481),
    (_MULT, _ADD, _EXPONENTIAL): ([1.57676353, 1.53767797, 1.76107376], 0.02720943),
    (_ADD, _ADD, _LINEAR): ([1.55892140, 1.52201252, 1.64702510], 0.04510930),
    (_ADD, _ADD, _EXPONENTIAL): ([1.56300335, 1.53025776, 1.65951621], 0.04548166),
}
# AutoTheta(AirPassengers, fh=4)
R_MODEL = "M M Exp"
R_FORECAST = [443.455181, 433.389284, 497.403726, 485.111109]


@pytest.mark.parametrize("candidate", R_CANDIDATES)
def test_candidates_match_authors_code(candidate):
    """Each of the eight models at a fixed theta gives the authors' forecasts."""
    seasonal, expr, trend = candidate
    expected, expected_mae = R_CANDIDATES[candidate]
    y = AIRLINE / AIRLINE.mean()
    n = y.shape[0]
    factors = np.zeros(1)
    adjusted = y
    if seasonal != _NONE:
        _, factors, adjusted = _seasonal_decompose_numba(
            y, 12, seasonal == _MULT, _MULTIPLICATIVE_FLOOR
        )
    theta = 1.7
    ok, mae, theta, _, _, level, a, b, _ = _fit_candidate(
        y, adjusted, factors, seasonal, expr, trend, theta, theta + 1e-9, 1e-12
    )
    assert ok
    forecast = _project(n, 3, expr, trend, theta, level, a, b)
    if seasonal == _ADD:
        forecast = forecast + _seasonal_forecast(factors, 3, 12, n)
    elif seasonal == _MULT:
        forecast = forecast * _seasonal_forecast(factors, 3, 12, n)
    np.testing.assert_allclose(forecast, expected, rtol=2e-3)
    # Not equality: on A A Exp, R stops at a poor initial level, which only affects
    # the first error when alpha is near one, so the MAE here is lower.
    assert mae <= expected_mae * (1 + 2e-3)


def test_selection_matches_authors_code():
    """The full algorithm selects the same model and forecast as AutoTheta."""
    forecaster = FourTheta(seasonal_period=12)
    forecast = forecaster.iterative_forecast(AIRLINE, 4)
    assert forecaster.model_ == R_MODEL
    assert forecaster.seasonal_
    np.testing.assert_allclose(forecast, R_FORECAST, rtol=2e-3)


def test_ses_initial_level_is_optimal():
    """The closed form initial level beats nearby levels at the same alpha."""
    z = AIRLINE / AIRLINE.mean()
    alpha, l0 = _fit_ses(z)
    sse = _ses_sse(z, alpha)[0]

    def sse_at(start):
        level = start
        total = 0.0
        for value in z:
            error = value - level
            total += error * error
            level += alpha * error
        return total

    assert sse == pytest.approx(sse_at(l0))
    assert sse < sse_at(l0 + 0.01)
    assert sse < sse_at(l0 - 0.01)


def test_fitted_state_is_consistent():
    """Residuals, one step forecast and scaling agree with the fitted state."""
    forecaster = FourTheta(seasonal_period=12).fit(AIRLINE)
    assert forecaster.seasonal_factors_.shape == (12,)
    np.testing.assert_allclose(
        forecaster.residuals_, AIRLINE - forecaster.fitted_values_
    )
    mae = np.mean(np.abs(forecaster.residuals_)) / AIRLINE.mean()
    assert mae == pytest.approx(forecaster.mae_)
    assert forecaster.predict(AIRLINE) == forecaster.forecast_
    first = forecaster.iterative_forecast(AIRLINE, 3)[0]
    assert first == pytest.approx(forecaster.forecast_)


def test_scale_equivariance():
    """Forecasts scale with the series."""
    base = FourTheta(seasonal_period=12).iterative_forecast(AIRLINE, 5)
    scaled = FourTheta(seasonal_period=12).iterative_forecast(AIRLINE * 1000, 5)
    np.testing.assert_allclose(scaled, base * 1000, rtol=1e-6)


def test_short_series_is_not_seasonal():
    """Seasonality needs three cycles, so the seasonal pool is not used."""
    forecaster = FourTheta(seasonal_period=12).fit(AIRLINE[:35])
    assert not forecaster.seasonal_
    assert forecaster.seasonal_factors_ is None
    assert forecaster.model_.startswith("N")


def test_non_positive_series_uses_additive_linear_models():
    """Only additive linear models are possible with non-positive values."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=60).cumsum()
    y = y - y.mean()
    forecaster = FourTheta().fit(y)
    assert forecaster.model_ == "N A Lrl"
    t = np.arange(48.0)
    y = np.sin(2 * np.pi * t / 12) + 0.01 * rng.normal(size=48)
    forecaster = FourTheta(seasonal_period=12).fit(y)
    assert forecaster.model_ == "A A Lrl"


def test_floor_only_for_non_negative_data():
    """Forecasts are floored at zero only if the training data are non-negative."""
    falling = np.arange(30.0, 1.0, -1.0)
    assert FourTheta().iterative_forecast(falling, 40).min() == 0.0
    assert FourTheta().iterative_forecast(falling - 40.0, 3).max() < 0.0
    assert FourTheta().iterative_forecast(falling - 15.0, 40).min() < 0.0


@pytest.mark.parametrize("y", [np.zeros(10), np.ones(10), np.full(10, -2.0)])
def test_constant_series(y):
    """Constant series, including zero mean, forecast the constant."""
    np.testing.assert_allclose(FourTheta().iterative_forecast(y, 3), y[:3], atol=1e-10)


@pytest.mark.parametrize(
    "params, message",
    [
        ({"seasonal_period": 0}, "seasonal_period"),
        ({"seasonal_period": True}, "seasonal_period"),
        ({"theta_bounds": (0.5, 3.0)}, "theta_bounds"),
        ({"theta_bounds": (3.0, 1.0)}, "theta_bounds"),
    ],
)
def test_invalid_parameters(params, message):
    """Invalid parameters raise in fit."""
    with pytest.raises(ValueError, match=message):
        FourTheta(**params).fit(AIRLINE)


def test_invalid_inputs():
    """Short series, bad horizons and exogenous data are rejected."""
    with pytest.raises(ValueError, match="at least 3"):
        FourTheta().fit(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="greater than"):
        FourTheta().iterative_forecast(AIRLINE, 0)
    with pytest.raises(TypeError, match="integer"):
        FourTheta().iterative_forecast(AIRLINE, 1.5)
    with pytest.raises(NotImplementedError, match="exog"):
        FourTheta().iterative_forecast(AIRLINE, 2, exog=AIRLINE)
