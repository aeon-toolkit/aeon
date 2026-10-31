"""Tests for the AR estimation utilities."""

import numpy as np
import pytest

from aeon.datasets import load_airline
from aeon.forecasting.utils._ar import (
    ar_forecast,
    ar_ols,
    autocovariance,
    burg,
    levinson_durbin,
    yule_walker,
)

# Reference values from statsmodels 0.15.0 on log(airline): acovf(fft=False),
# burg, pacf_burg, yule_walker(method="mle") and AutoReg(lags=3, trend="c").
# The Burg innovation variance follows R's ar.burg, not statsmodels.
ACOV = np.array([0.19353041493967493, 0.18457060877899242, 0.17396757598511745])
BURG_COEF = np.array([1.218873875130377, -0.3971203213271053, 0.14732803473080236])
BURG_PACF = np.array([1.0, 0.9703358759884286, -0.22237275155642144, 0.147328034730801])
YW_COEF = np.array([1.0722061711796846, -0.17537263682250223, 0.0542327376684344])
YW_SIGMA2 = 0.017212067464452117
OLS_PARAMS = np.array(
    [0.19484419941643039, 1.2096528247056044, -0.3980740729184295, 0.15473582369512603]
)
OLS_RSS = 1.4624872749368942


@pytest.fixture
def log_airline():
    """Log of the airline series."""
    return np.log(np.asarray(load_airline(), dtype=np.float64).squeeze())


def _simulate_ar(phi, n, seed=0):
    rng = np.random.default_rng(seed)
    p = len(phi)
    e = rng.normal(size=n + 100)
    y = np.zeros(n + 100)
    for t in range(p, n + 100):
        y[t] = np.dot(phi, y[t - p : t][::-1]) + e[t]
    return y[100:]


def test_autocovariance_matches_reference(log_airline):
    """Autocovariance matches statsmodels acovf."""
    np.testing.assert_allclose(autocovariance(log_airline, 2), ACOV, rtol=1e-12)


def test_burg_matches_reference(log_airline):
    """Burg coefficients match statsmodels, variance follows R."""
    coefs, sigma2, pacf = burg(log_airline, 3)
    np.testing.assert_allclose(coefs[3], BURG_COEF, rtol=1e-10)
    np.testing.assert_allclose(sigma2[0], ACOV[0], rtol=1e-12)
    np.testing.assert_allclose(sigma2[1:], sigma2[:-1] * (1 - BURG_PACF[1:] ** 2))
    np.testing.assert_allclose(pacf, BURG_PACF, rtol=1e-10)


def test_yule_walker_matches_reference(log_airline):
    """Yule-Walker matches statsmodels yule_walker(method='mle')."""
    coefs, sigma2, _ = yule_walker(log_airline, 3)
    np.testing.assert_allclose(coefs[3], YW_COEF, rtol=1e-10)
    np.testing.assert_allclose(sigma2[3], YW_SIGMA2, rtol=1e-10)


def test_ar_ols_matches_reference(log_airline):
    """Least squares matches statsmodels AutoReg."""
    coefs, intercepts, rss = ar_ols(log_airline, 3, 3)
    np.testing.assert_allclose(
        np.r_[intercepts[3], coefs[3]], OLS_PARAMS, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(rss[3], OLS_RSS, rtol=1e-10)


@pytest.mark.parametrize("order", [0, 1, 4, 13])
def test_ar_ols_every_order_matches_lstsq(order):
    """Each nested order matches a direct least squares fit on the same rows."""
    y = np.asarray(load_airline(), dtype=np.float64).squeeze()
    start = 13
    coefs, intercepts, rss = ar_ols(y, 13, start)
    X = np.column_stack(
        [np.ones(len(y) - start)]
        + [y[start - j : len(y) - j] for j in range(1, order + 1)]
    )
    beta, res, _, _ = np.linalg.lstsq(X, y[start:], rcond=None)
    np.testing.assert_allclose(
        np.r_[intercepts[order], coefs[order, :order]], beta, rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(rss[order], res[0], rtol=1e-8)


def test_levinson_durbin_lower_orders_are_nested(log_airline):
    """Row k of the coefficient table is the AR(k) Yule-Walker solution."""
    r = autocovariance(log_airline, 5)
    coefs, sigma2, _ = levinson_durbin(r, 5)
    for k in range(1, 6):
        toeplitz = np.array([[r[abs(i - j)] for j in range(k)] for i in range(k)])
        np.testing.assert_allclose(
            coefs[k, :k], np.linalg.solve(toeplitz, r[1 : k + 1]), rtol=1e-10
        )
        np.testing.assert_allclose(sigma2[k], r[0] - coefs[k, :k] @ r[1 : k + 1])


@pytest.mark.parametrize("estimator", [burg, yule_walker])
def test_burg_and_yule_walker_are_stationary(estimator):
    """Partial autocorrelations lie in (-1, 1), even near a unit root."""
    y = np.cumsum(_simulate_ar([0.5], 200))
    _, sigma2, pacf = estimator(y, 10)
    assert np.all(np.abs(pacf[1:]) < 1)
    assert np.all(sigma2 > 0)


def test_estimators_recover_ar2():
    """All estimators recover AR(2) coefficients on a long series."""
    y = _simulate_ar([0.6, -0.3], 5000)
    for coefs in [burg(y, 2)[0], yule_walker(y, 2)[0], ar_ols(y, 2, 2)[0]]:
        np.testing.assert_allclose(coefs[2], [0.6, -0.3], atol=0.05)


def test_estimators_on_constant_series():
    """A constant series gives zero coefficients and variance."""
    y = np.full(30, 5.0)
    for coefs, sigma2 in [burg(y, 3)[:2], yule_walker(y, 3)[:2]]:
        assert np.all(coefs == 0)
        assert np.all(sigma2 == 0)
    coefs, intercepts, rss = ar_ols(y, 3, 3)
    assert np.all(coefs == 0)
    np.testing.assert_allclose(intercepts, 5.0)
    np.testing.assert_allclose(rss, 0.0, atol=1e-20)


def test_ar_forecast_runs_recursion_forward():
    """ar_forecast feeds its own forecasts back in."""
    y = np.array([1.0, 2.0, 3.0])
    f = ar_forecast(y, 0.5, np.array([0.5, 0.25]), 3)
    f1 = 0.5 + 0.5 * 3 + 0.25 * 2
    f2 = 0.5 + 0.5 * f1 + 0.25 * 3
    f3 = 0.5 + 0.5 * f2 + 0.25 * f1
    np.testing.assert_allclose(f, [f1, f2, f3])
    np.testing.assert_allclose(ar_forecast(y, 2.0, np.zeros(0), 2), [2.0, 2.0])
