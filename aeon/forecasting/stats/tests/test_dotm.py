"""Tests for Dynamic Optimised Theta Model forecaster."""

import numpy as np
import pytest

from aeon.forecasting.stats import DynamicOptimisedThetaForecaster

Y_EXAMPLE = np.array([2.1, 2.4, 2.8, 3.0, 3.6, 4.1, 4.4, 4.9, 5.3, 5.9])


def test_dotm_fit_sets_attributes():
    """Fit should estimate DOTM parameters and fitted-state attributes."""
    forecaster = DynamicOptimisedThetaForecaster().fit(Y_EXAMPLE)

    assert np.isfinite(forecaster.initial_level_)
    assert 0.1 <= forecaster.alpha_ <= 0.99
    assert forecaster.theta_ >= 1.0
    assert forecaster.fitted_values_.shape == Y_EXAMPLE.shape
    assert forecaster.residuals_.shape == Y_EXAMPLE.shape
    assert np.isfinite(forecaster.forecast_)
    assert np.isfinite(forecaster.sse_)
    assert np.isfinite(forecaster.level_)
    assert np.isfinite(forecaster.a_)
    assert np.isfinite(forecaster.b_)
    assert np.isfinite(forecaster.mean_y_)


def test_dotm_iterative_forecast_shape():
    """iterative_forecast should return one forecast for each horizon step."""
    horizon = 5
    pred = DynamicOptimisedThetaForecaster().iterative_forecast(
        Y_EXAMPLE,
        prediction_horizon=horizon,
    )

    assert isinstance(pred, np.ndarray)
    assert pred.shape == (horizon,)
    assert np.all(np.isfinite(pred))


def test_dotm_forecast_matches_iterative_horizon_one():
    """forecast(y) should match iterative_forecast(y, 1)[0]."""
    forecaster = DynamicOptimisedThetaForecaster()
    forecast = forecaster.forecast(Y_EXAMPLE)
    iterative = forecaster.iterative_forecast(Y_EXAMPLE, prediction_horizon=1)[0]

    assert np.isclose(forecast, iterative)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.5, "theta": 2.0},
        {"initial_level": Y_EXAMPLE[0] / 2.0, "alpha": 0.5, "theta": 2.0},
    ],
)
def test_dotm_fixed_parameter_modes(kwargs):
    """Fixed DOTM parameters should be honoured during fitting."""
    forecaster = DynamicOptimisedThetaForecaster(**kwargs).fit(Y_EXAMPLE)

    if "initial_level" in kwargs:
        assert forecaster.initial_level_ == kwargs["initial_level"]
    if "alpha" in kwargs:
        assert forecaster.alpha_ == kwargs["alpha"]
    if "theta" in kwargs:
        assert forecaster.theta_ == kwargs["theta"]
    assert np.isfinite(forecaster.forecast_)


def test_dotm_short_series_raises():
    """DOTM should require at least four observations."""
    with pytest.raises(ValueError, match="at least 4 observations"):
        DynamicOptimisedThetaForecaster().fit(np.array([1.0, 2.0, 3.0]))


def test_dotm_constant_series_returns_finite_forecasts():
    """Constant series should fit and forecast without numerical failures."""
    y = np.full(12, 4.0)
    pred = DynamicOptimisedThetaForecaster().iterative_forecast(
        y,
        prediction_horizon=4,
    )

    assert pred.shape == (4,)
    assert np.all(np.isfinite(pred))


def test_dotm_exog_raises_not_implemented():
    """DOTM iterative forecasting should reject exogenous variables."""
    exog = np.arange(Y_EXAMPLE.shape[0], dtype=float)

    with pytest.raises(NotImplementedError, match="does not support exog"):
        DynamicOptimisedThetaForecaster().iterative_forecast(
            Y_EXAMPLE,
            prediction_horizon=2,
            exog=exog,
        )


def test_dotm_forecast_matches_statsforecast_reference():
    """Forecasts should be close to StatsForecast DynamicOptimizedTheta values."""
    expected = np.array([6.31989318, 6.74828469, 7.18045737, 7.61489322, 8.05073122])

    pred = DynamicOptimisedThetaForecaster().iterative_forecast(
        Y_EXAMPLE,
        prediction_horizon=5,
    )

    np.testing.assert_allclose(pred, expected, rtol=5e-3, atol=5e-3)
