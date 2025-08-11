"""Tests for Theta forecaster."""

import numpy as np
import pytest

from aeon.forecasting.stats import Theta


def test_fit_sets_attributes_and_predict_scalar():
    """Fit sets a_, b_, alpha_, forecast_, and _predict returns a scalar."""
    y = np.array([1.0, 1.2, 1.4, 1.6, 1.8])
    f = Theta(theta=2.0, weight=0.5).fit(y)
    assert hasattr(f, "a_") and isinstance(f.a_, float)
    assert hasattr(f, "b_") and isinstance(f.b_, float)
    assert hasattr(f, "alpha_") and 0.0 < f.alpha_ <= 1.0
    assert hasattr(f, "forecast_") and np.isscalar(f.forecast_)
    # _predict uses stored one-step forecast
    p = f.predict(y)
    assert np.isscalar(p)


def test_constant_series_forecast_is_constant_for_any_weight():
    """Constant series yields constant forecasts for h>1."""
    y = np.full(20, 7.0)
    for w in [0.0, 0.5, 1.0]:
        f = Theta(theta=2.0, weight=w).iterative_forecast(y, prediction_horizon=5)
        assert f.shape == (5,)
        assert np.allclose(f, 7.0)


def test_linear_series_trend_exact_with_weight_one():
    """With pure trend and weight=1, h-step forecast matches linear extrapolation."""
    n = 30
    a, b = 3.5, 0.8
    t = np.arange(n, dtype=float)
    y = a + b * t
    h = 6
    # weight=1: rely purely on trend component
    f = Theta(theta=2.0, weight=1.0).iterative_forecast(y, prediction_horizon=h)
    # Expected trend extrapolation: a + b * (n .. n+h-1)
    expected = a + b * np.arange(n, n + h, dtype=float)
    assert np.allclose(f, expected)


def test_fit_raises_on_too_few_observations():
    """Fit raises ValueError if series has fewer than 3 observations."""
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="at least 3 observations"):
        Theta(theta=2.0, weight=0.5).fit(y)


def test_weight_clipping_matches_extremes():
    """Weights <0 clip to 0; >1 clip to 1 (forecasts match)."""
    y = np.linspace(1.0, 2.0, 15)
    h = 4
    # Baselines at clipped extremes
    f_w0 = Theta(theta=2.0, weight=0.0).iterative_forecast(y, prediction_horizon=h)
    f_w1 = Theta(theta=2.0, weight=1.0).iterative_forecast(y, prediction_horizon=h)
    # Out-of-range weights should match clipped outputs
    f_w_neg = Theta(theta=2.0, weight=-5.0).iterative_forecast(y, prediction_horizon=h)
    f_w_big = Theta(theta=2.0, weight=5.0).iterative_forecast(y, prediction_horizon=h)
    assert np.allclose(f_w0, f_w_neg)
    assert np.allclose(f_w1, f_w_big)


def test_iterative_forecast_shape_and_type_and_squeeze_input():
    """iterative_forecast returns correct shape; input can be (n,1) and squeezed."""
    y = np.arange(10.0).reshape(-1, 1)  # column vector
    h = 3
    f = Theta(theta=2.0, weight=0.5).iterative_forecast(y, prediction_horizon=h)
    assert isinstance(f, np.ndarray)
    assert f.shape == (h,)


def test_tags_contain_horizon_false():
    """_tags has capability:horizon False (per provided implementation)."""
    assert Theta._tags.get("capability:horizon") is False
