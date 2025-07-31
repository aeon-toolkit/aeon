"""Theta forecater tests."""

import numpy as np

from aeon.forecasting.stats._theta import Theta


def test_theta():
    """Test the theta forecaster."""
    t = Theta(theta=0.0)  # predict last value
    t2 = Theta(theta=1.0)  # linear model
    t3 = Theta(theta=4.0)  # classical
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    p1 = t.fit(y).predict(y)
    p2 = t2.fit(y).predict(y)
    p3 = t3.fit(y).predict(y)
    assert p1 == p2 == p3  # Not correct!
