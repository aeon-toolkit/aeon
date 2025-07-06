"""Tests for the TemporalConvolutionalNetwork."""

import pytest

from aeon.networks import TemporalConvolutionalNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_tcn_network_basic():
    """Test basic TCN network creation and build_network functionality."""
    import tensorflow as tf

    input_shape = (100, 5)
    num_inputs = 5
    num_channels = [32, 64]

    tcn_network = TemporalConvolutionalNetwork(
        num_inputs=num_inputs, num_channels=num_channels
    )
    input_layer, output_layer = tcn_network.build_network(input_shape)

    # Check that layers are created correctly
    assert hasattr(input_layer, "shape"), "Input layer should have a shape attribute"
    assert hasattr(output_layer, "shape"), "Output layer should have a shape attribute"
    assert input_layer.dtype == tf.float32
    assert output_layer.dtype == tf.float32

    # Create a model to test the network structure
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    assert model is not None, "Model should be created successfully"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("num_channels", [[32], [32, 64], [16, 32, 64], [64, 32, 16]])
def test_tcn_network_different_channels(num_channels):
    """Test TCN network with different channel configurations."""
    import tensorflow as tf

    input_shape = (50, 3)
    num_inputs = 3

    tcn_network = TemporalConvolutionalNetwork(
        num_inputs=num_inputs, num_channels=num_channels
    )
    input_layer, output_layer = tcn_network.build_network(input_shape)

    # Create a model and verify it works
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    assert model is not None

    # Test with dummy data
    import numpy as np

    dummy_input = np.random.random((8,) + input_shape)
    output = model(dummy_input)
    assert output is not None, "Model should produce output"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("kernel_size", [2, 3, 5])
def test_tcn_network_kernel_sizes(kernel_size):
    """Test TCN network with different kernel sizes."""
    import tensorflow as tf

    input_shape = (80, 4)
    num_inputs = 4
    num_channels = [32, 64]

    tcn_network = TemporalConvolutionalNetwork(
        num_inputs=num_inputs,
        num_channels=num_channels,
        kernel_size=kernel_size,
    )
    input_layer, output_layer = tcn_network.build_network(input_shape)

    # Verify network builds successfully
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    assert model is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("dropout", [0.0, 0.1, 0.3, 0.5])
def test_tcn_network_dropout_rates(dropout):
    """Test TCN network with different dropout rates."""
    import tensorflow as tf

    input_shape = (60, 2)
    num_inputs = 2
    num_channels = [16, 32]

    tcn_network = TemporalConvolutionalNetwork(
        num_inputs=num_inputs, num_channels=num_channels, dropout=dropout
    )
    input_layer, output_layer = tcn_network.build_network(input_shape)

    # Verify network builds successfully
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    assert model is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_tcn_network_output_shape():
    """Test TCN network output shapes."""
    import numpy as np
    import tensorflow as tf

    input_shape = (40, 6)
    batch_size = 16
    num_inputs = 6
    num_channels = [32, 64]

    tcn_network = TemporalConvolutionalNetwork(
        num_inputs=num_inputs, num_channels=num_channels
    )
    input_layer, output_layer = tcn_network.build_network(input_shape)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Create dummy input and test output shape
    dummy_input = np.random.random((batch_size,) + input_shape)
    output = model(dummy_input)

    # Output should maintain sequence length and have final channel dimension
    expected_shape = (batch_size, num_channels[-1], input_shape[1])
    assert (
        output.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {output.shape}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_tcn_network_config():
    """Test TCN network configuration attributes."""
    tcn_network = TemporalConvolutionalNetwork(num_inputs=3, num_channels=[16, 32])

    # Check _config attributes
    assert "python_dependencies" in tcn_network._config
    assert "tensorflow" in tcn_network._config["python_dependencies"]
    assert "python_version" in tcn_network._config
    assert "structure" in tcn_network._config
    assert tcn_network._config["structure"] == "encoder"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_tcn_network_parameter_initialization():
    """Test TCN network parameter initialization."""
    num_inputs = 4
    num_channels = [32, 64, 128]
    kernel_size = 3
    dropout = 0.2

    tcn_network = TemporalConvolutionalNetwork(
        num_inputs=num_inputs,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
    )

    # Check that parameters are set correctly
    assert tcn_network.num_inputs == num_inputs
    assert tcn_network.num_channels == num_channels
    assert tcn_network.kernel_size == kernel_size
    assert tcn_network.dropout == dropout


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_tcn_network_single_layer():
    """Test TCN network with single temporal block."""
    import tensorflow as tf

    input_shape = (30, 2)
    num_inputs = 2
    num_channels = [16]  # Single layer

    tcn_network = TemporalConvolutionalNetwork(
        num_inputs=num_inputs, num_channels=num_channels
    )
    input_layer, output_layer = tcn_network.build_network(input_shape)

    # Verify single layer network works
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    assert model is not None

    # Test with dummy data
    import numpy as np

    dummy_input = np.random.random((4,) + input_shape)
    output = model(dummy_input)
    assert output.shape == (4, 16, 2)
