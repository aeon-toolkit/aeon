"""Test ETS."""

import numpy as np

from aeon.forecasting import ETSForecaster


def test_ets_forecaster():
    """TestETSForecaster."""
    data = np.array(
        [3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12]
    )  # Sample seasonal data
    forecaster = ETSForecaster(alpha=0.5, beta=0.3, gamma=0.4, season_length=4)
    forecaster.fit(data)
    p = forecaster.predict()
    assert np.isclose(p, 15.85174501127)
