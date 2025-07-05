"""Test TVP forecaster."""

import numpy as np

from aeon.forecasting._tvp import TVPForecaster


def test_expected_results():
    """Test aeon TVP Forecaster equivalent to statsmodels."""
    expected = np.array([1.0, 2.0, 3.0])
    tvp = TVPForecaster(window=5, horizon=5, var=0.01, beta_var=0.01)
    p = tvp.forecast(expected)
    p2 = tvp.direct_forecast(expected, prediction_horizon=5)
    p3 = tvp.iterative_forecast(expected, prediction_horizon=5)
    assert p == p2[0]
    assert p == p3[0]
