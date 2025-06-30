"""Test the regression forecaster."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from aeon.forecasting import RegressionForecaster
from aeon.regression import DummyRegressor


def test_regression_forecaster():
    """Test the regression forecaster."""
    y = np.random.rand(100)
    f = RegressionForecaster(window=10)
    f.fit(y)
    p = f.predict(y)
    p2 = f.forecast(y)
    assert p == p2
    f2 = RegressionForecaster(regressor=LinearRegression(), window=10)
    f2.fit(y)
    p2 = f2.predict(y)
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


def test_regression_forecaster_with_exog():
    """Test the regression forecaster with exogenous variables."""
    np.random.seed(0)

    n_samples = 100
    exog = np.random.rand(n_samples) * 10
    y = 2 * exog + np.random.rand(n_samples) * 0.1

    f = RegressionForecaster(window=10)

    # Test fit and predict with exog
    f.fit(y, exog=exog)
    p = f.predict(y, exog=exog)
    assert isinstance(p, float)

    # Test that exog variable has an impact
    exog_zeros = np.zeros(n_samples)
    f.fit(y, exog=exog_zeros)
    p2 = f.predict(y, exog=exog)
    assert p != p2

    # Test that forecast method works and is equivalent to fit+predict
    y_new = np.arange(50, 150)
    exog_new = np.arange(50, 150) * 2

    # Manual fit + predict
    f.fit(y=y_new, exog=exog_new)
    p_manual = f.predict(y_new, exog=exog_new)

    # forecast() method
    p_forecast = f.forecast(y=y_new, exog=exog_new)
    assert p_manual == pytest.approx(p_forecast)


def test_regression_forecaster_with_exog_errors():
    """Test errors in regression forecaster with exogenous variables."""
    y = np.random.rand(100)
    exog_short = np.random.rand(99)
    f = RegressionForecaster(window=10)

    # Test for unequal length series
    with pytest.raises(ValueError, match="must have the same number of time points"):
        f.fit(y, exog=exog_short)

    with pytest.raises(ValueError, match="must have the same number of time points"):
        f.fit(y)
        f.predict(y, exog=exog_short)
