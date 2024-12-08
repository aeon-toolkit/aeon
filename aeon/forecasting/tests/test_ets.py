"""Test ETS forecaster."""

import pytest

from aeon.forecasting import ETSForecaster
from aeon.testing.data_generation import make_example_1d_numpy


def test_ets_params():
    """Test ETS forecaster."""
    y = make_example_1d_numpy(n_timepoints=100)
    forecaster = ETSForecaster(error_type=3)
    with pytest.raises(
        ValueError, match="Error must be either additive or " "multiplicative"
    ):
        forecaster.fit(y)
    forecaster = ETSForecaster(seasonality_type=-3)
    forecaster.fit(y)
    assert forecaster._seasonal_period == 1
    forecaster = ETSForecaster(trend_type=None, seasonality_type=0, beta=1.0, gamma=1.0)
    forecaster.fit(y)
    assert forecaster._beta == 0
    assert forecaster._gamma == 0

    forecaster = ETSForecaster(error_type=2, phi=1.0)
    pred = forecaster.forecast(y)
    assert isinstance(pred, float)
