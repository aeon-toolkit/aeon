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
    p = f.predict()
    p2 = f.predict(y)
    assert p == p2
    p3 = f.forecast(y)
    assert p == p3
    f2 = RegressionForecaster(regressor=LinearRegression(), window=10)
    f2.fit(y)
    p2 = f2.predict()
    assert p == p2
    f2 = RegressionForecaster(regressor=DummyRegressor(), window=10)
    f2.fit(y)
    f2.predict()

    with pytest.raises(ValueError):
        f = RegressionForecaster(window=-1)
        f.fit(y)
    with pytest.raises(ValueError):
        f = RegressionForecaster(window=101)
        f.fit(y)
