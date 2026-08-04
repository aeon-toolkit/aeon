"""Tests for autoregressive forecasting."""

import numpy as np
import pytest

from aeon.forecasting.stats import AR


def test_ar_zero_order_with_demean_forecasts_training_mean():
    """AR(0) with demeaning should forecast on the original scale."""
    y = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

    forecaster = AR(ar_order=0, demean=True).fit(y)

    assert forecaster.p_ == 0
    assert np.isclose(forecaster.forecast_, np.mean(y))
    assert np.isclose(forecaster.predict(y), np.mean(y))


def test_ar_zero_order_without_demean_forecasts_intercept():
    """AR(0) without demeaning estimates an intercept."""
    y = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

    forecaster = AR(ar_order=0, demean=False).fit(y)

    assert forecaster.p_ == 0
    assert np.isclose(forecaster.intercept_, np.mean(y), atol=1e-10)
    assert np.isclose(forecaster.forecast_, np.mean(y), atol=1e-10)


def test_ar_one_recovers_noiseless_ar1_rule():
    """AR(1) predicts close to the generating rule on a noiseless process."""
    intercept = 0.5
    phi = 0.8
    y = np.empty(120)
    y[0] = 1.0
    for i in range(1, y.size):
        y[i] = intercept + phi * y[i - 1]

    forecaster = AR(ar_order=1, demean=False).fit(y)
    expected = intercept + phi * y[-1]

    assert np.isclose(forecaster.coef_[0], phi, atol=1e-4)
    assert np.isclose(forecaster.forecast_, expected, atol=1e-4)


def test_ar_order_selection_returns_finite_forecast():
    """Automatic order selection should return a finite one-step forecast."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=80)

    forecaster = AR(ar_order=None, p_max=5, criterion="BIC").fit(y)

    assert 0 <= forecaster.p_ <= 5
    assert np.isfinite(forecaster.forecast_)
    assert forecaster.params_["selection"]["mode"] == "scan"


@pytest.mark.parametrize(
    "kwargs, error_type, message",
    [
        ({"ar_order": -1}, ValueError, "ar_order"),
        ({"ar_order": 1.5}, TypeError, "ar_order"),
        ({"p_max": -1}, TypeError, "p_max"),
        ({"criterion": "HQIC"}, ValueError, "criterion"),
        ({"demean": "yes"}, TypeError, "demean"),
    ],
)
def test_ar_invalid_parameters_raise(kwargs, error_type, message):
    """Invalid constructor parameters raise clear errors."""
    with pytest.raises(error_type, match=message):
        AR(**kwargs).fit(np.arange(20.0))
