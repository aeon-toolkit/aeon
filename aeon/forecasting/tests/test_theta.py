"""Tests for the theta forecaster."""

import numpy as np

from aeon.forecasting import Theta


def test_theta_numba():
    """Test Theta."""
    y = np.array(
        [
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
        ]
    )
    theta = Theta()
    theta.fit(y)
    y_hat = theta.predict(y)
    assert y_hat == 10.0
