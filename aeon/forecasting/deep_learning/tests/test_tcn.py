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
@pytest.mark.parametrize("horizon,window,epochs", [(1, 10, 2), (3, 12, 3), (5, 15, 2)])
def test_tcn_forecaster(horizon, window, epochs):
    """Test TCNForecaster with different parameter combinations."""
    import tensorflow as tf

    # Load airline dataset
    y = load_airline()

    # Initialize TCNForecaster
    forecaster = TCNForecaster(
        horizon=horizon, window=window, epochs=epochs, batch_size=16, verbose=0
    )

    # Fit and predict
    forecaster.fit(y)
    prediction = forecaster.predict(y)

    # Basic assertions
    assert prediction is not None
    if isinstance(prediction, tf.Tensor):
        assert not tf.math.is_nan(prediction).numpy()
