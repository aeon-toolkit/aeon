"""Test TCN."""

__maintainer__ = []
__all__ = []

import pytest

from aeon.datasets import load_airline
from aeon.forecasting.deep_learning._tcn import TCNForecaster
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("horizon,window,epochs", [(1, 10, 2), (1, 12, 3), (1, 15, 2)])
def test_tcn_forecaster(horizon, window, epochs):
    """Test TCNForecaster with different parameter combinations."""
    import tensorflow as tf

    # Load airline dataset
    y = load_airline()

    # Initialize TCNForecaster
    forecaster = TCNForecaster(
        horizon=horizon, window=window, n_epochs=epochs, batch_size=16, verbose=0
    )

    # Fit and predict
    forecaster.fit(y)
    prediction = forecaster.predict(y)

    # Basic assertions
    assert isinstance(prediction, float)
    if isinstance(prediction, tf.Tensor):
        assert not tf.math.is_nan(prediction).numpy()


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "loader,is_univariate",
    [
        (load_airline, True),  # univariate dataset
        # (load_longley, False),  # multivariate dataset
    ],
)
def test_tcn_forecaster_uni_mutli(loader, is_univariate):
    """Test TCNForecaster on univariate (airline) and multivariate (longley) data."""
    y = loader()

    forecaster = TCNForecaster(
        horizon=1,
        window=10,
        n_epochs=2,
        batch_size=16,
        verbose=0,
    )

    # fit
    forecaster.fit(y)

    # predict
    prediction = forecaster.predict(y)
    assert isinstance(prediction, float)

    # forecast
    prediction = forecaster.forecast(y)
    assert isinstance(prediction, float)

    # iterative forecasting
    prediction = forecaster.iterative_forecast(y, 3)
    assert prediction is not None
    assert len(prediction) == 3
