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
@pytest.mark.parametrize(
    "horizon,window,type", [(1, 10, "linear"), (1, 10, "nlinear"), (1, 10, "dlinear")]
)
def test_linear_forecaster(horizon, window, type):
    """Test LinearForecaster with different parameter combinations."""
    y = load_airline()

    forecaster = LinearForecaster(
        horizon=horizon, window=window, type=type, n_epochs=5, batch_size=16, verbose=0
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
    "prediction_horizon,horizon,window,type",
    [
        (3, 3, 10, "linear"),
        (5, 3, 10, "linear"),
        (8, 3, 10, "nlinear"),
        (12, 3, 10, "dlinear"),
    ],
)
def test_series_to_series_forecast(prediction_horizon, horizon, window, type):
    """Test Linear Forecaster for different `prediction_horizon` and `type` values."""
    y = load_airline()

    forecaster = LinearForecaster(
        window=window, horizon=horizon, type=type, n_epochs=10, batch_size=16
    )

    forecaster.fit(y)
    predictions = forecaster.series_to_series_forecast(
        y=y, prediction_horizon=prediction_horizon
    )

    assert predictions is not None
    assert len(predictions) == prediction_horizon
    assert isinstance(predictions, np.ndarray)
    assert predictions.ndim == 1
