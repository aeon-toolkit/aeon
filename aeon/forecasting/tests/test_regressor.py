"""Test the regression forecaster."""

from sklearn.linear_model import LinearRegression

from aeon.datasets import load_airline
from aeon.forecasting import RegressionForecaster


def test_regression_forecaster():
    """Test the regression forecaster."""
    y = load_airline()
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
