"""Test functions for LinearForecaster."""

import numpy as np
import pytest

from aeon.datasets import load_airline
from aeon.forecasting.deep_learning._ltsf_linear import LinearForecaster
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("horizon,window,epochs", [(1, 10, 2), (2, 12, 3), (5, 15, 2)])
def test_linear_forecaster(horizon, window, epochs):
    """Test LinearForecaster with different parameter combinations."""
    y = load_airline()

    forecaster = LinearForecaster(
        horizon=horizon, window=window, n_epochs=epochs, batch_size=16, verbose=0
    )

    forecaster.fit(y)
    prediction = forecaster.predict(y)

    assert prediction is not None
    if isinstance(prediction, np.ndarray):
        assert not np.isnan(prediction).any()


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize(
    "prediction_horizon,horizon,window",
    [(1, 3, 10), (3, 3, 10), (8, 3, 10), (12, 3, 10)],
)
def test_series_to_series_forecast(prediction_horizon, horizon, window):
    """Test Linear Forecaster for different `prediction_horizon` values."""
    y = load_airline()

    forecaster = LinearForecaster(
        window=window, horizon=horizon, n_epochs=10, batch_size=16
    )

    forecaster.fit(y)
    predictions = forecaster.series_to_series_forecast(
        y=y, prediction_horizon=prediction_horizon
    )

    assert predictions is not None
    assert len(predictions) == prediction_horizon
    assert isinstance(predictions, np.ndarray)
