"""Test the ARIMA forecaster."""

import numpy as np
import pytest

from aeon.forecasting import ARIMA as ARIMAForecaster

y = np.array(
    [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118], dtype=np.float64
)


def test_arima_zero_orders():
    """Test ARIMA(0,0,0) which should return the mean if constant is used."""
    model = ARIMAForecaster(p=0, d=0, q=0, use_constant=True)
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
    model = ARIMAForecaster(p=p, d=d, q=q, use_constant=use_constant)
    model.fit(y)
    forecast = model.forecast_
    assert isinstance(forecast, float)
    assert np.isfinite(forecast)


def test_arima_iterative_forecast():
    """Test multi-step forecasting using iterative_forecast method."""
    model = ARIMAForecaster(p=1, d=1, q=1)
    horizon = 3
    preds = model.iterative_forecast(y, prediction_horizon=horizon)
    assert preds.shape == (horizon,)
    assert np.all(np.isfinite(preds))


def test_predict_too_short_for_d():
    """Test error is raised when input series is too short for differencing."""
    model = ARIMAForecaster(p=1, d=2, q=1)
    model.fit(y)
    with pytest.raises(ValueError, match="Series too short for differencing"):
        model._predict(np.array([1.0, 2.0]))


def test_predict_too_short_for_pq():
    """Test error is raised when input series is too short for AR/MA terms."""
    model = ARIMAForecaster(p=5, d=0, q=0)
    model.fit(y)
    with pytest.raises(ValueError, match="Series too short for ARMA"):
        model._predict(np.array([1.0, 2.0, 3.0]))


def test_forecast_attribute_set():
    """Test that calling _forecast sets the internal forecast_ attribute."""
    model = ARIMAForecaster(p=1, d=1, q=1)
    forecast = model._forecast(y)
    assert hasattr(model, "forecast_")
    assert np.isclose(forecast, model.forecast_)


def test_iterative_forecast_with_d2():
    """Test iterative forecast output shape and validity with d=2."""
    model = ARIMAForecaster(p=1, d=2, q=1)
    preds = model.iterative_forecast(y, prediction_horizon=5)
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))
