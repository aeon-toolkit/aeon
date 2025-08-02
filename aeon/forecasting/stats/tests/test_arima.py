"""Test the ARIMA forecaster."""

import numpy as np
import pytest

from aeon.forecasting.stats._arima import ARIMA

y = np.array(
    [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118], dtype=np.float64
)


def test_arima_zero_orders():
    """Test ARIMA(0,0,0) which should return the mean if constant is used."""
    model = ARIMA(p=0, d=0, q=0, use_constant=True)
    model.fit(y)
    forecast = model.predict(y)
    assert np.isfinite(forecast)
    assert abs(forecast - np.mean(y)) < 10


@pytest.mark.parametrize(
    "p, d, q, use_constant",
    [
        (1, 0, 1, True),  # basic ARIMA
        (2, 1, 1, False),  # no constant
        (1, 2, 1, True),  # higher-order differencing
    ],
)
def test_arima_fit_and_predict_variants(p, d, q, use_constant):
    """Test ARIMA fit and predict for various (p,d,q) and use_constant settings."""
    model = ARIMA(p=p, d=d, q=q, use_constant=use_constant)
    model.fit(y)
    forecast = model.forecast_
    assert isinstance(forecast, float)
    assert np.isfinite(forecast)


def test_arima_iterative_forecast():
    """Test multi-step forecasting using iterative_forecast method."""
    model = ARIMA(p=1, d=1, q=1)
    horizon = 3
    preds = model.iterative_forecast(y, prediction_horizon=horizon)
    assert preds.shape == (horizon,)
    assert np.all(np.isfinite(preds))
    model = ARIMA(p=1, d=0, q=1, use_constant=True)
    preds = model.iterative_forecast(y, prediction_horizon=horizon)
    assert preds.shape == (horizon,)


@pytest.mark.parametrize(
    "y_input, error_match",
    [
        (np.array([1.0, 2.0]), "Series too short for differencing"),
        (np.array([1.0, 2.0, 3.0]), "Series too short for ARMA"),
    ],
)
def test_arima_too_short_series_errors(y_input, error_match):
    """Test errors raised for too short input series."""
    model = ARIMA(p=3, d=2, q=3)
    model.fit(y)
    with pytest.raises(ValueError, match=error_match):
        model._predict(y_input)


def test_forecast_attribute_set():
    """Test that calling _forecast sets the internal forecast_ attribute."""
    model = ARIMA(p=1, d=1, q=1)
    forecast = model._forecast(y)
    assert hasattr(model, "forecast_")
    assert np.isclose(forecast, model.forecast_)


def test_iterative_forecast_with_d2():
    """Test iterative forecast output shape and validity with d=2."""
    model = ARIMA(p=1, d=2, q=1)
    preds = model.iterative_forecast(y, prediction_horizon=5)
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))


@pytest.mark.parametrize(
    "p, d, q, use_constant, expected_forecast",
    [
        (1, 0, 1, False, 118.47506756),  # precomputed from known ARIMA implementation
        (2, 1, 1, False, 209.1099231455),  # precomputed
        (3, 0, 0, True, 137.47368045155),  # precomputed
    ],
)
def test_arima_fixed_paras(p, d, q, use_constant, expected_forecast):
    """Test ARIMA fit/predict accuracy against known forecasts.

    expected values calculated with values fitted by Nelder-Mead:

    1. phi = [0.99497524] theta [0.0691515]
    2. phi = [ 0.02898788 -0.4330671 ] theta [1.26699252]
    3. phi = [ 0.19202414  0.05207654 -0.07367897] theta [], constant 105.970867164

    """
    model = ARIMA(p=p, d=d, q=q, use_constant=use_constant)
    model.fit(y)
    forecast = model.forecast_
    assert isinstance(forecast, float)
    assert np.isfinite(forecast)
    assert np.isclose(forecast, expected_forecast, atol=1e-6)


def test_arima_known_output():
    """Test ARIMA for fixed parameters.

    Test ARMIMA with forecast generated externally.
    """
    model = ARIMA(p=1, d=0, q=1)
    model.fit(y)
    f = model.forecast_
    assert np.isclose(118.47506756, f)
