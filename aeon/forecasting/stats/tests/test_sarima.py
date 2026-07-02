"""Tests for Seasonal ARIMA (SARIMA) forecaster."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aeon.forecasting.stats._sarima import (
    SARIMA,
    _difference,
    _get_active_lags,
    _get_diff_poly,
    _undifference_sarima,
)


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
