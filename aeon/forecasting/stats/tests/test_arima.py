"""Test the ARIMA forecaster."""

import numpy as np
import pytest

from aeon.forecasting.stats._arima import ARIMA, AutoARIMA

y = np.array(
    [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118], dtype=np.float64
)


class _FitCountingARIMA(ARIMA):
    """ARIMA test double that counts internal fit calls."""

    def __init__(self, p=0, d=0, q=0, use_constant=True, iterations=5):
        self.fit_calls_ = 0
        super().__init__(
            p=p,
            d=d,
            q=q,
            use_constant=use_constant,
            iterations=iterations,
        )

    def _fit(self, y, exog=None):
        self.fit_calls_ += 1
        return super()._fit(y, exog=exog)


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


def test_arima_iterative_forecast_fits_once_per_call():
    """iterative_forecast should fit once before recursive forecasting."""
    model = _FitCountingARIMA()
    preds = model.iterative_forecast(y, prediction_horizon=3)

    assert preds.shape == (3,)
    assert model.fit_calls_ == 1


def test_arima_iterative_forecast_with_future_exog_fits_once():
    """iterative_forecast should use training exog and future exog."""
    horizon = 3
    rng = np.random.RandomState(12)
    exog = rng.randn(len(y), 2)
    future_exog = rng.randn(horizon, 2)
    model = _FitCountingARIMA()

    preds = model.iterative_forecast(
        y,
        prediction_horizon=horizon,
        exog=exog,
        future_exog=future_exog,
    )

    assert preds.shape == (horizon,)
    assert model.fit_calls_ == 1


def test_arima_iterative_forecast_accepts_one_dimensional_exog():
    """iterative_forecast should accept 1D train and future exog arrays."""
    horizon = 3
    train_exog = np.arange(len(y), dtype=float)
    future_exog = np.arange(len(y), len(y) + horizon, dtype=float)
    model = ARIMA(p=1, d=0, q=1)

    preds = model.iterative_forecast(
        y,
        prediction_horizon=horizon,
        exog=train_exog,
        future_exog=future_exog,
    )

    assert preds.shape == (horizon,)
    assert np.all(np.isfinite(preds))


def test_arima_iterative_forecast_rejects_future_only_exog():
    """iterative_forecast should reject future-only exog."""
    horizon = 3
    future_exog = np.random.RandomState(13).randn(horizon, 2)
    model = ARIMA(p=1, d=0, q=1)

    with pytest.raises(ValueError, match="provided together"):
        model.iterative_forecast(y, prediction_horizon=horizon, future_exog=future_exog)


def test_arima_iterative_forecast_rejects_train_only_exog():
    """iterative_forecast should reject training-only exog."""
    horizon = 3
    train_exog = np.random.RandomState(14).randn(len(y), 2)
    model = ARIMA(p=1, d=0, q=1)

    with pytest.raises(ValueError, match="provided together"):
        model.iterative_forecast(y, prediction_horizon=horizon, exog=train_exog)


def test_arima_iterative_forecast_refits_on_repeated_close_series():
    """Repeated iterative_forecast calls should refit even for close series."""
    model = _FitCountingARIMA()
    y_close = y.copy()
    y_close[0] += 1e-12

    model.iterative_forecast(y, prediction_horizon=2)
    model.iterative_forecast(y_close, prediction_horizon=2)

    assert model.fit_calls_ == 2


def test_arima_predict_does_not_refit():
    """Predict should use the fitted ARIMA state without calling _fit."""
    model = _FitCountingARIMA()
    model.fit(y)
    fit_calls = model.fit_calls_

    model.predict(y)

    assert model.fit_calls_ == fit_calls


def test_arima_iterative_forecast_rejects_non_positive_horizon():
    """iterative_forecast should reject horizons below one."""
    model = ARIMA(p=1, d=0, q=0)

    with pytest.raises(ValueError, match="prediction_horizon"):
        model.iterative_forecast(y, prediction_horizon=0)


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


def test_autoarima_respects_max_d_zero():
    """AutoARIMA(max_d=0) must not apply any non-seasonal differencing.

    A pure trend series is non-stationary, so the differencing loop would
    otherwise apply one difference; with max_d=0 the fitted order must stay 0.
    Regression test for #3577.
    """
    y_trend = np.arange(50, dtype=float)
    forecaster = AutoARIMA(max_d=0)
    forecaster.fit(y_trend)
    assert forecaster.d_ == 0


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
    """iterative_forecast should fit and return valid shape."""
    horizon = 4
    forecaster = AutoARIMA()
    preds = forecaster.iterative_forecast(y, prediction_horizon=horizon)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (horizon,)
    assert np.all(np.isfinite(preds))


def test_autoarima_iterative_forecast_with_future_exog():
    """AutoARIMA.iterative_forecast should use training exog and future exog."""
    horizon = 3
    rng = np.random.RandomState(15)
    exog = rng.randn(len(y), 1)
    future_exog = rng.randn(horizon, 1)
    forecaster = AutoARIMA(max_p=1, max_d=1, max_q=1)

    preds = forecaster.iterative_forecast(
        y,
        prediction_horizon=horizon,
        exog=exog,
        future_exog=future_exog,
    )

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


def test_arima_with_exog_basic_fit_predict():
    """Test ARIMA fit and predict with exogenous variables."""
    y_local = np.arange(50, dtype=float)
    exog = np.random.RandomState(42).randn(50, 2)
    model = ARIMA(p=1, d=0, q=1)
    model.fit(y_local, exog=exog)
    pred = model.predict(y_local, exog=exog[-1:].copy())
    assert isinstance(pred, float)
    assert np.isfinite(pred)


def test_arima_exog_shape_mismatch_raises():
    """Test that exogenous shape mismatches raise ValueError."""
    y_local = np.arange(20, dtype=float)
    exog = np.random.RandomState(0).randn(20, 3)
    model = ARIMA(p=1, d=0, q=1)
    with pytest.raises(ValueError, match="same number of rows"):
        model.fit(y_local, exog=np.random.randn(10, 3))
    model.fit(y_local, exog=exog)
    with pytest.raises(ValueError, match="exog must have .* features"):
        model.predict(y_local, exog=np.random.randn(1, 5))


def test_arima_iterative_forecast_with_exog():
    """Test multi-step forecast with training and future exogenous variables."""
    y_local = np.arange(40, dtype=float)
    h = 5
    rng = np.random.RandomState(1)
    exog = rng.randn(len(y_local), 2)
    future_exog = rng.randn(h, 2)
    model = ARIMA(p=1, d=1, q=1)
    preds = model.iterative_forecast(
        y_local,
        prediction_horizon=h,
        exog=exog,
        future_exog=future_exog,
    )
    assert preds.shape == (h,)
    assert np.all(np.isfinite(preds))


def test_arima_no_exog_backward_compatibility():
    """Test ARIMA works normally when no exogenous variables are provided."""
    y_local = np.arange(30, dtype=float)
    model = ARIMA(p=1, d=1, q=1)
    model.fit(y_local)
    pred = model.predict(y_local)
    assert isinstance(pred, float)
    assert np.isfinite(pred)


def test_autoarima_passes_exog_correctly():
    """Test that AutoARIMA successfully passes exog to the inner ARIMA."""
    rng = np.random.RandomState(42)
    y_local = rng.randn(30)
    exog = rng.randn(30, 1)
    model = AutoARIMA(max_p=1, max_q=1, max_d=1)
    model.fit(y_local, exog=exog)
    next_exog = rng.randn(1, 1)
    pred = model.predict(y_local, exog=next_exog)
    assert np.isfinite(pred)
    assert model.final_model_.exog_ is not None
