"""Test the regression forecaster."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from aeon.forecasting import RegressionForecaster
from aeon.regression import DummyRegressor


def test_regression_forecaster():
    """Test the regression forecaster.

    Test fit/predict and forecast are equivalent for sklearn and aeon regressors.
    Test invalid window handling.
    """
    y = np.random.rand(100)
    f = RegressionForecaster(window=10)
    p = f.forecast(y)
    f.fit(y)
    p2 = f.predict(y)
    assert p == p2
    f2 = RegressionForecaster(regressor=LinearRegression(), window=10)
    p2 = f2.forecast(y)
    assert p == p2
    f2 = RegressionForecaster(regressor=DummyRegressor(), window=10)
    f2.fit(y)
    f2.predict(y)

    with pytest.raises(ValueError):
        f = RegressionForecaster(window=-1)
        f.fit(y)
    with pytest.raises(ValueError):
        f = RegressionForecaster(window=101)
        f.fit(y)
    f = RegressionForecaster(window=10)
    f.fit(y)
    y_test = np.random.rand(5)
    with pytest.raises(ValueError):
        p = f.predict(y_test)


def test_regression_forecaster_with_exog():
    """Test the regression forecaster with exogenous variables."""
    n_samples = 100
    exog = np.random.rand(n_samples) * 10
    y = 2 * exog + np.random.rand(n_samples) * 0.1
    f = RegressionForecaster(window=10)

    # Test fit and predict with exog
    f.fit(y, exog=exog)
    p1 = f.forecast_
    assert isinstance(p1, float)

    # Test that exog variable have an impact
    exog_zeros = np.zeros(n_samples)
    f.fit(y, exog=exog_zeros)
    p2 = f.forecast_
    assert p1 != p2

    # Test that forecast method works and is equivalent to fit+predict
    y_new = np.arange(50, 150)
    exog_new = np.arange(50, 150) * 2

    # Manual fit + predict
    f.fit(y=y_new, exog=exog_new)
    p_manual = f.predict(y_new, exog=exog_new)

    # forecast() method
    p_forecast = f.forecast(y=y_new, exog=exog_new)
    assert p_manual == pytest.approx(p_forecast)

    # Test with multivariate exog
    exog_m = np.array([exog, exog_zeros])
    p1 = f.forecast(y, exog_m)
    f.fit(y, exog_m)
    p2 = f.predict(y, exog_m)
    assert p1 == p2
    y = np.random.random((1, 100))
    exog = np.random.random((1, 100))
    f._fit(y, exog)


def test_regression_forecaster_with_exog_errors():
    """Test errors in regression forecaster with exogenous variables."""
    y = np.random.rand(100)
    exog_short = np.random.rand(99)
    f = RegressionForecaster(window=10)

    # Test for unequal length series in fit
    with pytest.raises(
        ValueError, match="y and exog must have the same number of time points"
    ):
        f.fit(y, exog=exog_short)
    # Test for fit/predict mismatches in shape

    # If exog in fit, must have them in predict
    exog_train = np.array(np.random.rand(100))
    with pytest.raises(ValueError, match="predict passed no exogenous variables"):
        f.fit(y, exog=exog_train)
        f.predict(y)
    exog_test = np.array([np.random.rand(10), np.random.rand(10)])
    f.fit(y, exog=exog_train)
    with pytest.raises(ValueError, match="Forecaster passed"):
        f.predict(y, exog_test)
    exog_short = np.random.rand(5)
    with pytest.raises(
        ValueError, match="Exogenous variables passed in predict of length"
    ):
        f.predict(y, exog_short)
    with pytest.raises(ValueError, match="predict passed no exogenous variables"):
        f.predict(y)
    with pytest.raises(ValueError, match="must be greater than or equal to 1"):
        f.direct_forecast(y, prediction_horizon=0)
