"""Tests for Seasonal ARIMA (SARIMA) forecaster."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aeon.datasets import load_airline
from aeon.forecasting.stats import ARIMA, AutoARIMA, AutoSARIMA
from aeon.forecasting.stats._sarima import (
    SARIMA,
    _build_model,
    _difference,
    _get_active_lags,
    _get_diff_poly,
    _roots_clear_of_unit_circle,
    _seasonal_strength,
    _undifference_sarima,
)
from aeon.forecasting.utils._extract_paras import _sarima_coefficients, _sarima_lags
from aeon.forecasting.utils._loss_functions import _arima_css_aic, _sarima_fit
from aeon.forecasting.utils._nelder_mead import _sarima_hannan_rissanen


def test_active_lags():
    """Verify active lag indices calculations."""
    ar_lags, ma_lags = _get_active_lags(p=1, q=1, P=1, Q=1, m=4)
    assert_allclose(ar_lags, [1, 4, 5])
    assert_allclose(ma_lags, [1, 4, 5])


def test_differencing_sarima():
    """Test seasonal and non-seasonal differencing and undifferencing."""
    # Test non-seasonal d=1, D=0
    poly = _get_diff_poly(d=1, D=0, s=1)
    assert_allclose(poly, [1.0, -1.0])

    y = np.array([1.0, 2.0, 4.0, 7.0])
    y_diff = _difference(y, poly)
    assert_allclose(y_diff, [1.0, 2.0, 3.0])

    y_undiff = _undifference_sarima(y_diff, y[:1], poly)
    assert_allclose(y_undiff, y)

    # Test seasonal d=0, D=1, s=4
    poly = _get_diff_poly(d=0, D=1, s=4)
    assert_allclose(poly, [1.0, 0.0, 0.0, 0.0, -1.0])

    y = np.array([1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 5.0, 8.0])
    y_diff = _difference(y, poly)
    assert_allclose(y_diff, [1.0, 2.0, 2.0, 4.0])

    y_undiff = _undifference_sarima(y_diff, y[:4], poly)
    assert_allclose(y_undiff, y)


def test_seasonal_naive_special_case():
    """Verify SARIMA(0,0,0)(0,1,0)_m behaves like seasonal naive forecasting."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5, 2.0, 3.0, 4.0, 5.0])
    # D=1, seasonal_period=4
    model = SARIMA(p=0, d=0, q=0, P=0, D=1, Q=0, seasonal_period=4, use_constant=False)
    model.fit(y)

    # Future differenced forecasts are 0, so forecast should repeat
    # values from s periods ago:
    # y[-4] = 2.0, y[-3] = 3.0, y[-2] = 4.0, y[-1] = 5.0
    forecasts = model.iterative_forecast(y, prediction_horizon=4)
    assert_allclose(forecasts, [2.0, 3.0, 4.0, 5.0])


def test_sarima_vs_statsmodels():
    """Verify SARIMA forecasting correctness against statsmodels SARIMAX."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        pytest.skip("statsmodels not installed, skipping comparison tests.")

    np.random.seed(42)
    # Generate some simple seasonal data
    y = np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.05, 100)

    # Fit statsmodels SARIMAX (p=1, d=0, q=1) x (P=1, D=0, Q=0)_4
    sm_model = SARIMAX(y, order=(1, 0, 1), seasonal_order=(1, 0, 0, 4), trend="c")
    sm_fit = sm_model.fit(disp=False)
    sm_pred = sm_fit.forecast(steps=1)[0]

    # Fit aeon SARIMA
    aeon_model = SARIMA(
        p=1, d=0, q=1, P=1, D=0, Q=0, seasonal_period=4, use_constant=True
    )
    aeon_model.fit(y)
    aeon_pred = aeon_model.predict(y)

    # Compare 1-step ahead forecasts
    assert_allclose(aeon_pred, sm_pred, atol=0.1)


def test_sarima_too_short_series_errors():
    """Test errors raised for too short input series."""
    y_input = np.array([1.0, 2.0])
    model = SARIMA(p=3, d=2, q=3)
    model.fit(np.arange(10.0))
    with pytest.raises(ValueError, match="Series too short for differencing"):
        model._predict(y_input)


def test_sarima_series_shorter_than_seasonal_lags_errors():
    """Test fit and predict raise when no residual can be estimated."""
    period = 12
    y = load_airline()
    model = SARIMA(p=1, d=0, q=0, P=1, seasonal_period=period)
    with pytest.raises(ValueError):
        model.fit(y[: period + 1])
    model.fit(y)
    with pytest.raises(ValueError):
        model.predict(y[: period + 1])


@pytest.mark.parametrize(
    "params", [{"seasonal_period": 0}, {"P": -1}, {"d": -1}, {"Q": -1}]
)
def test_sarima_invalid_parameters_error(params):
    """Test fit rejects a non-positive period and negative orders."""
    with pytest.raises(ValueError):
        SARIMA(**params).fit(load_airline())


def test_sarima_rejects_exog():
    """Test exogenous variables are not accepted."""
    y = load_airline()
    with pytest.raises(ValueError):
        SARIMA().fit(y, exog=y)


@pytest.mark.parametrize(
    "p, d, q, use_constant", [(1, 0, 0, True), (1, 1, 1, False), (2, 1, 2, True)]
)
def test_sarima_without_seasonal_terms_matches_arima(p, d, q, use_constant):
    """Test SARIMA with no seasonal terms gives the same fit as ARIMA."""
    y = load_airline()
    arima = ARIMA(p=p, d=d, q=q, use_constant=use_constant).fit(y)
    sarima = SARIMA(p=p, d=d, q=q, use_constant=use_constant).fit(y)
    assert_allclose(sarima._parameters, arima._parameters)
    assert_allclose(sarima.aic_, arima.aic_)
    assert_allclose(sarima.forecast_, arima.forecast_)


def test_sarima_aic_counts_all_observations():
    """Test the AIC uses the ARIMA likelihood over the whole differenced series."""
    y = load_airline()
    model = SARIMA(p=1, q=1, P=1, seasonal_period=12).fit(y)
    assert_allclose(
        model.aic_,
        _arima_css_aic(
            model.residuals_, _sarima_lags(model._model)[2], len(model._parameters)
        ),
    )


@pytest.mark.parametrize(
    "params",
    [
        [1.1, 0.1, 0.1, 0.1],  # phi
        [0.1, 1.1, 0.1, 0.1],  # theta
        [0.1, 0.1, 1.1, 0.1],  # Phi
        [0.1, 0.1, 0.1, 1.1],  # Theta
    ],
)
def test_sarima_fit_rejects_non_stationary_or_non_invertible(params):
    """Test the loss is infinite if any polynomial has a root in the unit circle."""
    p, q, P, Q, m = 1, 1, 1, 1, 4
    ar_lags, ma_lags = _get_active_lags(p, q, P, Q, m)
    model = np.array(
        [0, p, q, P, Q, m, len(ar_lags), len(ma_lags), *ar_lags, *ma_lags],
        dtype=np.int32,
    )
    data = np.random.default_rng(0).normal(size=50)
    assert np.isinf(_sarima_fit(np.array(params), data, model))


def test_sarima_coefficients_expand_multiplicative_polynomials():
    """Test the sparse lag coefficients are the product of the polynomials."""
    phi, theta, Phi, Theta = 0.5, 0.25, 0.4, 0.2
    p, q, P, Q, m = 1, 1, 1, 1, 4
    ar_lags, ma_lags = _get_active_lags(p, q, P, Q, m)
    model = np.array(
        [0, p, q, P, Q, m, len(ar_lags), len(ma_lags), *ar_lags, *ma_lags],
        dtype=np.int32,
    )
    params = np.array([phi, theta, Phi, Theta])
    _, ar_values, ma_values = _sarima_coefficients(params, model)
    assert_allclose(ar_values, [phi, Phi, -phi * Phi])
    assert_allclose(ma_values, [theta, Theta, theta * Theta])


def test_sarima_hannan_rissanen_recovers_seasonal_ar():
    """Test the starting values use the parameter layout of the model."""
    period, phi, Phi = 4, 0.5, -0.4
    rng = np.random.default_rng(0)
    e = rng.normal(size=5000)
    y = np.zeros(len(e))
    for t in range(period + 1, len(e)):
        y[t] = phi * y[t - 1] + Phi * y[t - period]
        y[t] += -phi * Phi * y[t - period - 1] + e[t]
    p, q, P, Q = 1, 0, 1, 0
    ar_lags, ma_lags = _get_active_lags(p, q, P, Q, period)
    model = np.array(
        [0, p, q, P, Q, period, len(ar_lags), len(ma_lags), *ar_lags, *ma_lags],
        dtype=np.int32,
    )
    assert_allclose(_sarima_hannan_rissanen(y, model), [phi, Phi], atol=0.1)


def test_sarima_predict_matches_forecast_on_fitted_series():
    """Test predict on the training series gives the forecast from fit."""
    y = load_airline()
    model = SARIMA(p=0, d=1, q=1, D=1, Q=1, seasonal_period=12).fit(y)
    assert_allclose(model.predict(y), model.forecast_)


def test_sarima_iterative_forecast_uses_fitted_state():
    """Test iterative forecasts continue the fitted model with a single fit."""
    y = load_airline()
    model = SARIMA(p=0, d=1, q=1, D=1, Q=1, seasonal_period=12)
    horizon = 4
    forecasts = model.iterative_forecast(y, horizon)
    assert forecasts.shape == (horizon,)
    assert_allclose(forecasts[0], model.forecast_)
    # the second step is the prediction from y extended by the first, with no refit
    assert_allclose(forecasts[1], model.predict(np.append(y, forecasts[0])))


def test_sarima_airline_model_vs_statsmodels():
    """Verify the seasonal fit on airline is close to the exact likelihood fit."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        pytest.skip("statsmodels not installed, skipping comparison tests.")

    y = load_airline()
    horizon = 4
    sm_fit = SARIMAX(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    model = SARIMA(p=0, d=1, q=1, D=1, Q=1, seasonal_period=12)
    forecasts = model.iterative_forecast(y, horizon)
    assert_allclose(forecasts, sm_fit.forecast(steps=horizon), rtol=0.01)


def test_auto_sarima_non_seasonal_matches_auto_arima():
    """Test that with a period of one the search is that of AutoARIMA."""
    y = load_airline()
    auto_arima = AutoARIMA().fit(y)
    auto_sarima = AutoSARIMA().fit(y)
    assert (auto_sarima.p_, auto_sarima.d_, auto_sarima.q_) == (
        auto_arima.p_,
        auto_arima.d_,
        auto_arima.q_,
    )
    assert auto_sarima.P_ == auto_sarima.D_ == auto_sarima.Q_ == 0
    assert auto_sarima.constant_term_ == auto_arima.constant_term_
    assert_allclose(auto_sarima.fit_aic_, auto_arima.fit_aic_)
    assert_allclose(auto_sarima.forecast_, auto_arima.forecast_)


def test_auto_sarima_selects_seasonal_difference():
    """Test a strongly seasonal series is seasonally differenced."""
    period = 12
    y = load_airline()
    model = AutoSARIMA(seasonal_period=period).fit(y)
    assert model.D_ == 1
    assert model.final_model_.seasonal_period == period
    assert_allclose(model.fit_aic_, model.final_model_.aic_)
    assert_allclose(model.predict(y), model.forecast_)


def test_auto_sarima_improves_on_non_seasonal_search():
    """Test seasonal terms reduce the error on a simulated seasonal series."""
    period, n, horizon = 4, 200, 8
    rng = np.random.default_rng(0)
    e = rng.normal(size=n + horizon)
    y = 10.0 * np.sin(2 * np.pi * np.arange(n + horizon) / period) + e
    train, test = y[:n], y[n:]
    seasonal = AutoSARIMA(seasonal_period=period).iterative_forecast(train, horizon)
    plain = AutoSARIMA().iterative_forecast(train, horizon)
    assert np.mean(np.abs(seasonal - test)) < np.mean(np.abs(plain - test))


@pytest.mark.parametrize(
    "limits",
    [
        {"max_p": 1, "max_q": 1, "max_P": 0, "max_Q": 1},
        {"max_p": 0, "max_q": 0, "max_P": 1, "max_Q": 0},
    ],
)
def test_auto_sarima_respects_limits(limits):
    """Test the selected orders are within the limits."""
    model = AutoSARIMA(seasonal_period=12, **limits).fit(load_airline())
    assert model.p_ <= limits["max_p"]
    assert model.q_ <= limits["max_q"]
    assert model.P_ <= limits["max_P"]
    assert model.Q_ <= limits["max_Q"]


def test_auto_sarima_max_D_zero_skips_seasonal_difference():
    """Test seasonal differencing can be switched off."""
    assert AutoSARIMA(seasonal_period=12, max_D=0).fit(load_airline()).D_ == 0


def test_seasonal_strength_short_series_is_zero():
    """Test a series of at most two seasons cannot be decomposed."""
    period = 12
    assert _seasonal_strength(load_airline()[: 2 * period], period) == 0.0


def test_auto_sarima_short_seasonal_series():
    """Test a series shorter than a season falls back to non-seasonal terms."""
    period = 12
    model = AutoSARIMA(seasonal_period=period).fit(load_airline()[: period - 2])
    assert model.P_ == model.D_ == model.Q_ == 0


def test_seasonal_root_margin_applies_at_seasonal_lags():
    """Test a seasonal root clears 1.01 at the seasonal lag, not at lag one."""
    period = 12
    # root of 1 - Phi B^m has modulus (1 / Phi)^(1 / m) in B, inside 1.01
    Phi = 0.95
    assert 1.01 > (1 / Phi) ** (1 / period)
    model = _build_model(False, 0, 0, 1, 0, period)
    assert not _roots_clear_of_unit_circle(np.array([Phi]), model)
    # the same coefficient on a non-seasonal lag is clear
    model = _build_model(False, 1, 0, 0, 0, period)
    assert _roots_clear_of_unit_circle(np.array([Phi]), model)
