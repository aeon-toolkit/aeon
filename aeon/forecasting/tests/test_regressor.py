"""Test the regression forecaster."""

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
