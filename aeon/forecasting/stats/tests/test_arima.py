"""Test the ARIMA forecaster."""

import numpy as np
import pytest

from aeon.forecasting.stats._arima import ARIMA, AutoARIMA

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


# todo
# failing due to incorrect expected results (or output)
# see also test_dispatch_loss in forecasting utils

# @pytest.mark.parametrize(
#     "p, d, q, use_constant, expected_forecast",
#     [
#         (1, 0, 1, False, 118.47506756),  # precomputed from known ARIMA implementation
#         (2, 1, 1, False, 138.9587),  # precomputed
#         (3, 0, 0, True, 137.007633),  # precomputed
#     ],
# )
# def test_arima_fixed_paras(p, d, q, use_constant, expected_forecast):
#     """Test ARIMA fit/predict accuracy against known forecasts.
#
#     expected values calculated with values fitted by Nelder-Mead:
#
#     1. phi = [0.99497524] theta [0.0691515]
#     2. phi = [ 0.02898788 -0.4330671 ] theta [1.26699252]
#     3. phi = [ 0.19202414  0.05207654 -0.07367897] theta [], constant 105.970867164
#
#     """
#     model = ARIMA(p=p, d=d, q=q, use_constant=use_constant)
#     model.fit(y)
#     forecast = model.forecast_
#     assert isinstance(forecast, float)
#     assert np.isfinite(forecast)
#     assert np.isclose(forecast, expected_forecast, atol=1e-6)
#
#
# def test_arima_known_output():
#     """Test ARIMA for fixed parameters.
#
#     Test ARMIMA with forecast generated externally.
#     """
#     model = ARIMA(p=1, d=0, q=1)
#     model.fit(y)
#     f = model.forecast_
#     assert np.isclose(118.47506756, f)


def test_autoarima_fit_sets_model_and_orders_within_bounds():
    """Fit should set (p_, d_, q_) within configured maxima and wrap an ARIMA."""
    forecaster = AutoARIMA(max_p=3, max_d=3, max_q=2)
    forecaster.fit(y)

    # wrapped model exists and is ARIMA
    assert forecaster.final_model_ is not None
    assert isinstance(forecaster.final_model_, ARIMA)

    # orders exist and are within bounds
    assert 0 <= forecaster.p_ <= forecaster.max_p
    assert 0 <= forecaster.d_ <= forecaster.max_d
    assert 0 <= forecaster.q_ <= forecaster.max_q


def test_autoarima_predict_returns_finite_float():
    """_predict should return a finite float once fitted."""
    forecaster = AutoARIMA()
    forecaster.fit(y)
    pred = forecaster._predict(y)
    assert isinstance(pred, float)
    assert np.isfinite(pred)


def test_autoarima_forecast_sets_wrapped_and_returns_forecast_float():
    """_forecast should refit, set wrapped forecast_, and return that value."""
    forecaster = AutoARIMA()
    f = forecaster._forecast(y)
    assert isinstance(f, float)
    assert hasattr(forecaster.final_model_, "forecast_")
    assert np.isclose(f, forecaster.final_model_.forecast_)


def test_autoarima_iterative_forecast_shape_and_validity():
    """iterative_forecast should delegate to wrapped model and return valid shape."""
    horizon = 4
    forecaster = AutoARIMA()
    forecaster.fit(y)
    preds = forecaster.iterative_forecast(y, prediction_horizon=horizon)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (horizon,)
    assert np.all(np.isfinite(preds))


def test_autoarima_respects_small_max_orders():
    """With small max orders, ensure discovered orders don’t exceed those limits."""
    forecaster = AutoARIMA(max_p=1, max_d=1, max_q=1)
    forecaster.fit(y)
    assert 0 <= forecaster.p_ <= 1
    assert 0 <= forecaster.d_ <= 1
    assert 0 <= forecaster.q_ <= 1


def test_autoarima_predict_matches_wrapped_predict():
    """_predict should be a thin wrapper around final_model_.predict."""
    forecaster = AutoARIMA()
    forecaster.fit(y)
    a = forecaster._predict(y)
    b = forecaster.final_model_.predict(y)
    # both are floats and close
    assert isinstance(a, float) and isinstance(b, float)
    assert np.isfinite(a) and np.isfinite(b)
    assert np.isclose(a, b)


def test_autoarima_forecast_is_consistent_with_wrapped():
    """_forecast should match the wrapped model's forecast after internal fit."""
    forecaster = AutoARIMA()
    val = forecaster._forecast(y)
    assert np.isclose(val, forecaster.final_model_.forecast_)
