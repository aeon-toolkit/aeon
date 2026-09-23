"""Tests for the NBeatsNetwork."""

import numpy as np
import pytest

from aeon.networks import NBeatsNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "horizon,stacks,num_blocks_per_stack,units,share_weights,share_coefficients",
    [
        (10, ["trend"], 2, 32, True, True),
        (20, ["seasonality"], 1, 64, False, True),
        (15, ["generic"], 3, 128, True, False),
        (25, ["trend", "seasonality"], 2, 256, False, False),
        (30, ["trend", "generic"], 1, 16, True, True),
        (40, ["seasonality", "generic"], 2, 48, False, True),
        (50, ["trend", "seasonality", "generic"], 3, 80, True, False),
    ],
)
def test_nbeats_network_build_and_shapes(
    horizon, stacks, num_blocks_per_stack, units, share_weights, share_coefficients
):
    """Test NBeatsNetwork with different parameter configurations."""
    import tensorflow as tf

    input_shape = (100, 1)

    nbeats_network = NBeatsNetwork(
        horizon=horizon,
        stacks=stacks,
        num_blocks_per_stack=num_blocks_per_stack,
        units=units,
        share_weights=share_weights,
        share_coefficients=share_coefficients,
    )
    input_layer, output_layer = nbeats_network.build_network(input_shape)

    assert hasattr(input_layer, "shape"), "Input layer should have a shape attribute"
    assert len(output_layer) == 2, "Output should contain backcast and forecast"
    assert input_layer.dtype == tf.float32

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    expected_input_shape = (None, input_shape[0])
    assert input_layer.shape == expected_input_shape

    backcast_output, forecast_output = output_layer
    assert backcast_output.shape == (None, input_shape[0])
    assert forecast_output.shape == (None, horizon)

    batch_size = 4
    dummy_input = np.random.randn(batch_size, input_shape[0])
    backcast, forecast = model(dummy_input)

    assert backcast.shape == (batch_size, input_shape[0])
    assert forecast.shape == (batch_size, horizon)

    assert not tf.reduce_any(tf.math.is_nan(backcast))
    assert not tf.reduce_any(tf.math.is_nan(forecast))
