"""Test base forecaster."""

import numpy as np

from aeon.forecasting import DummyForecaster


def test_base_forecaster():
    """Test base forecaster functionality."""
    f = DummyForecaster()
    y = np.random.rand(50)
    f.fit(y)
    p1 = f.predict()
    assert p1 == y[-1]
    p2 = f.forecast(y)
    assert p2 == p1
