"""Tests for ARForecaster."""

import numpy as np
import pytest

from aeon.datasets import load_airline
from aeon.forecasting.stats import ARForecaster

METHODS = ["burg", "yule_walker", "ols"]


def _simulate_ar(phi, n, seed=0):
    rng = np.random.default_rng(seed)
    p = len(phi)
    e = rng.normal(size=n + 100)
    y = np.zeros(n + 100)
    for t in range(p, n + 100):
        y[t] = np.dot(phi, y[t - p : t][::-1]) + e[t]
    return y[100:]


@pytest.mark.parametrize("method", METHODS)
def test_ar_selects_true_order(method):
    """Order selection finds AR(2) on a long simulated series."""
    y = _simulate_ar([0.6, -0.3], 2000)
    f = ARForecaster(method=method).fit(y)
    assert f.p_ == 2
    np.testing.assert_allclose(f.coef_, [0.6, -0.3], atol=0.05)


@pytest.mark.parametrize("method", METHODS)
def test_ar_selects_order_zero_for_white_noise(method):
    """AIC picks order 0 for most white noise series (about 70% asymptotically)."""
    zero = 0
    for seed in range(50):
        y = np.random.default_rng(seed).normal(size=500) + 10.0
        f = ARForecaster(method=method).fit(y)
        if f.p_ == 0:
            zero += 1
            np.testing.assert_allclose(f.forecast_, np.mean(y))
    assert zero >= 30


@pytest.mark.parametrize("method", METHODS)
def test_ar_selection_is_scale_invariant(method):
    """Rescaling the series does not change the selected model."""
    y = _simulate_ar([0.5, 0.2, -0.2], 300, seed=2)
    fits = [ARForecaster(method=method).fit(y * s) for s in (1e-3, 1.0, 1e3)]
    assert len({f.p_ for f in fits}) == 1
    for f, s in zip(fits, (1e-3, 1.0, 1e3)):
        np.testing.assert_allclose(f.coef_, fits[1].coef_, rtol=1e-6)
        np.testing.assert_allclose(f.forecast_ / s, fits[1].forecast_, rtol=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_ar_predict_uses_new_series(method):
    """Predict forecasts from the series passed to it, without refitting."""
    y = np.asarray(load_airline(), dtype=np.float64).squeeze()
    f = ARForecaster(p=3, method=method).fit(y[:100])
    new = y[100:]
    expected = f.intercept_ + f.coef_ @ new[-1:-4:-1]
    np.testing.assert_allclose(f.predict(new), expected)
    assert f.predict(new) != f.forecast_


def test_ar_fixed_order_ols_matches_least_squares():
    """OLS with a fixed order matches a direct least squares fit."""
    y = np.asarray(load_airline(), dtype=np.float64).squeeze()
    f = ARForecaster(p=2, method="ols").fit(y)
    X = np.column_stack([np.ones(len(y) - 2), y[1:-1], y[:-2]])
    beta = np.linalg.lstsq(X, y[2:], rcond=None)[0]
    np.testing.assert_allclose(np.r_[f.intercept_, f.coef_], beta, rtol=1e-8)


def test_ar_horizon_matches_iterative_forecast():
    """A horizon h forecast equals step h of the iterative forecast."""
    y = np.asarray(load_airline(), dtype=np.float64).squeeze()
    path = ARForecaster(p=4).iterative_forecast(y, 5)
    for h in range(1, 6):
        assert ARForecaster(p=4, horizon=h).forecast(y) == pytest.approx(path[h - 1])


def test_ar_iterative_forecast_matches_base_mixin():
    """The single-call override gives the same path as predicting step by step."""
    y = np.asarray(load_airline(), dtype=np.float64).squeeze()
    f = ARForecaster(method="ols")
    path = f.iterative_forecast(y, 6)
    ext = y.copy()
    for i in range(6):
        assert f.predict(ext) == pytest.approx(path[i])
        ext = np.append(ext, path[i])


def test_ar_constant_series():
    """A constant series gives order 0 and the constant as forecast."""
    y = np.full(40, 3.0)
    for method in METHODS:
        f = ARForecaster(method=method).fit(y)
        assert f.p_ == 0
        assert f.forecast_ == pytest.approx(3.0)


def test_ar_max_p_reduced_for_short_series():
    """max_p is reduced when the series is too short for it."""
    y = np.random.default_rng(3).normal(size=12)
    assert ARForecaster(max_p=50, method="ols").fit(y).p_ <= (12 - 2) // 2
    assert ARForecaster(max_p=50).fit(y).p_ <= 11


def test_ar_invalid_parameters():
    """Invalid parameters and inputs raise informative errors."""
    y = np.random.default_rng(4).normal(size=20)
    with pytest.raises(ValueError, match="method"):
        ARForecaster(method="mle").fit(y)
    with pytest.raises(ValueError, match="p must be"):
        ARForecaster(p=-1).fit(y)
    with pytest.raises(ValueError, match="too short"):
        ARForecaster(p=10, method="ols").fit(y)
    f = ARForecaster(p=5).fit(y)
    with pytest.raises(ValueError, match="at least"):
        f.predict(y[:3])
    with pytest.raises(NotImplementedError):
        f.iterative_forecast(y, 2, exog=y)
