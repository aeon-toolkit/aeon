"""Test for BaseDeepForecaster class in aeon."""

import numpy as np
import pytest

from aeon.forecasting.deep_learning import BaseDeepForecaster
from aeon.utils.validation._dependencies import _check_soft_dependencies


class SimpleDeepForecaster(BaseDeepForecaster):
    """A simple concrete implementation of BaseDeepForecaster for testing."""

    def __init__(self, horizon=1, window=5, epochs=1, verbose=0):
        super().__init__(horizon=horizon, window=window, epochs=epochs, verbose=verbose)

    def _build_model(self, input_shape):
        import tensorflow as tf

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=input_shape),
                tf.keras.layers.Dense(10, activation="relu"),
                tf.keras.layers.Dense(self.horizon),
            ]
        )
        return model


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_base_deep_forecaster_fit_predict():
    """Test fitting and predicting with BaseDeepForecaster implementation."""
    # Generate synthetic data
    np.random.seed(42)
    data = np.random.randn(50)

    # Initialize forecaster
    forecaster = SimpleDeepForecaster(horizon=2, window=5, epochs=1, verbose=0)

    # Fit the model
    forecaster.fit(data)

    # Predict
    predictions = forecaster.predict(data)

    # Validate output shape
    assert (
        len(predictions) == 2
    ), f"Expected predictions of length 2, got {len(predictions)}"
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_base_deep_forecaster_insufficient_data():
    """Test error handling for insufficient data."""
    data = np.random.randn(5)
    forecaster = SimpleDeepForecaster(horizon=2, window=5, epochs=1, verbose=0)

    with pytest.raises(ValueError, match="Data length.*insufficient"):
        forecaster.fit(data)
