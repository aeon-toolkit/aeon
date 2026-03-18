"""Tests for SETAR forecaster."""

import numpy as np
import pytest

from aeon.forecasting.machine_learning._setar import SETAR


def test_constant_series_scalar_and_close():
    """Forecasting on a constant series returns a scalar near the constant."""
    y = np.ones(40)
    f = SETAR(lag=10)
    f.fit(y)
    pred = f.predict(y)
    assert np.isscalar(pred), "Prediction should be a scalar for univariate series"
    assert np.isclose(pred, 1.0, atol=1e-6), f"Prediction {pred} not close to 1.0"


def test_linear_series_scalar_and_next_value():
    """Forecasting on an arithmetic progression returns the next value (scalar)."""
    y = np.arange(1.0, 41.0)  # 1..40
    f = SETAR(lag=5)
    f.fit(y)
    pred = f.predict(y)
    assert np.isscalar(pred), "Prediction should be a scalar for univariate series"
    assert np.isclose(pred, 41.0, atol=0.05), f"Prediction {pred} not close to 41.0"


def test_forecast_sets_attribute_and_matches_predict():
    """forecast() returns scalar and sets forecast_ equal to the returned value."""
    y = np.arange(1.0, 31.0)
    f = SETAR(lag=6)
    out = f.forecast(y)  # triggers internal fit + 1-step forecast
    assert np.isscalar(out), "forecast() should return a scalar"
    assert hasattr(f, "forecast_"), "forecast_ should be set by _fit/_forecast"
    assert np.isclose(f.forecast_, out), "forecast_ must equal the returned forecast"
    f2 = SETAR(lag=6)
    f2.fit(y)
    pred = f2.predict(y)
    assert np.isclose(
        pred, out, atol=0.1
    ), "predict() should match forecast() on same y"


def test_too_short_series_for_fallback_raises():
    """Fitting with len(y) <= lag triggers fallback length check and raises."""
    y = np.ones(5)  # deliberately short
    f = SETAR(lag=10)  # fallback path will require len(y) > lag
    with pytest.raises(ValueError):
        f.fit(y)


# helpers for synthetic data


def _simulate_ar1(phi=0.8, a0=0.0, y0=1.0, n=200):
    """Simulate y_t = a0 + phi * y_{t-1} (no noise)."""
    y = np.empty(n)
    y[0] = y0
    for t in range(1, n):
        y[t] = a0 + phi * y[t - 1]
    return y


def _simulate_setar(
    a_low=-0.5, b_low=0.6, a_high=0.4, b_high=0.8, thr=0.0, y0=-2.0, n=300
):
    """Simulate a noiseless 2-regime SETAR(1) with threshold on y_{t-1}."""
    y = np.empty(n)
    y[0] = y0
    for t in range(1, n):
        if y[t - 1] <= thr:
            y[t] = a_low + b_low * y[t - 1]
        else:
            y[t] = a_high + b_high * y[t - 1]
    return y


def test_ar1_decay_next_value_matches_rule():
    """AR(1) decay series predicts next value close to phi * last."""
    y = _simulate_ar1(phi=0.82, a0=0.0, y0=1.7, n=300)
    f = SETAR(lag=3)
    f.fit(y)
    pred = f.predict(y)
    expected = 0.82 * y[-1]
    assert np.isscalar(pred), "Prediction should be a scalar"
    assert np.isclose(pred, expected, atol=1e-2), f"{pred=} not close to {expected=}"


def test_ar1_with_intercept_next_value_matches_rule():
    """AR(1) with intercept predicts next value close to a0 + phi * last."""
    a0, phi = 0.5, 0.9
    y = _simulate_ar1(phi=phi, a0=a0, y0=0.0, n=300)
    f = SETAR(lag=4)
    f.fit(y)
    pred = f.predict(y)
    expected = a0 + phi * y[-1]
    assert np.isclose(pred, expected, atol=2e-2), f"{pred=} not close to {expected=}"


def test_step_change_piecewise_constant_predicts_level():
    """Piecewise constant (0 then 5) predicts near the last level 5."""
    y = np.concatenate([np.zeros(120), np.full(120, 5.0)])
    f = SETAR(lag=6)
    f.fit(y)
    pred = f.predict(y)
    assert np.isclose(pred, 5.0, atol=1e-3), f"Prediction {pred} not close to 5.0"


def test_setar_two_regime_series_predicts_rule_next_step():
    """Synthetic 2-regime SETAR(1) predicts close to the true next value."""
    a_low, b_low = -0.5, 0.6
    a_high, b_high = 0.4, 0.8
    y = _simulate_setar(a_low, b_low, a_high, b_high, thr=0.0, y0=-2.0, n=600)
    f = SETAR(lag=1)  # true process is lag-1
    f.fit(y)
    last = y[-1]
    expected = (a_low + b_low * last) if last <= 0.0 else (a_high + b_high * last)
    pred = f.predict(y)
    assert np.isclose(pred, expected, atol=5e-2), f"{pred=} not close to {expected=}"


def test_predict_depends_only_on_tail_lags():
    """Predictions on series with identical last `current_lag` values are equal."""
    y_full = np.arange(1.0, 501.0)  # 1..500
    f = SETAR(lag=6)
    f.fit(y_full)
    pred_full = f.predict(y_full)
    keep = f.current_lag + 25
    y_tail = y_full[-keep:]
    pred_tail = f.predict(y_tail)
    assert np.isclose(pred_full, pred_tail, atol=1e-9), "Tail-invariance violated"


def test_none_return_from_fit_setar_falls_back_to_linear():
    """If no valid threshold is found for any lag, it falls back and still predicts."""
    # Constant series makes threshold grid degenerate; ensures None path is hit.
    y = np.ones(80)
    f = SETAR(lag=8)
    f.fit(y)
    assert f.model in {"linear", "setar"}  # typically linear here
    pred = f.predict(y)
    assert np.isfinite(pred)
    assert np.isclose(pred, 1.0, atol=1e-6)
