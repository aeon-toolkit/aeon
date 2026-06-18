"""Tests for the TCNNetwork."""

import pytest

from aeon.networks import TCNNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_tcn_network_basic():
    """Test basic TCN network creation and build_network functionality."""
    import tensorflow as tf

    input_shape = (100, 5)  # (n_timepoints, n_channels)
    n_blocks = [32, 64]
    tcn_network = TCNNetwork(n_blocks=n_blocks)

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
@pytest.mark.parametrize("n_blocks", [[32], [32, 64], [16, 32, 64], [64, 32, 16]])
def test_tcn_network_different_channels(n_blocks):
    """Test TCN network with different channel configurations."""
    import tensorflow as tf

    input_shape = (50, 3)  # (n_timepoints, n_channels)
    tcn_network = TCNNetwork(n_blocks=n_blocks)

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

    input_shape = (80, 4)  # (n_timepoints, n_channels)
    n_blocks = [32, 64]

    tcn_network = TCNNetwork(
        n_blocks=n_blocks,
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

    input_shape = (60, 2)  # (n_timepoints, n_channels)
    n_blocks = [16, 32]

    tcn_network = TCNNetwork(n_blocks=n_blocks, dropout=dropout)
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

    input_shape = (40, 6)  # (n_timepoints, n_channels)
    batch_size = 16
    n_blocks = [32, 64]

    tcn_network = TCNNetwork(n_blocks=n_blocks)
    input_layer, output_layer = tcn_network.build_network(input_shape)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Create dummy input and test output shape
    dummy_input = np.random.random((batch_size,) + input_shape)
    output = model(dummy_input)

    # Output should have the same number of channels as input
    expected_shape = (batch_size, input_shape[1])  # (batch_size, n_channels)
    assert (
        output.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {output.shape}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_tcn_network_config():
    """Test TCN network configuration attributes."""
    tcn_network = TCNNetwork(n_blocks=[16, 32])

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
    n_blocks = [32, 64, 128]
    kernel_size = 3
    dropout = 0.2

    tcn_network = TCNNetwork(
        n_blocks=n_blocks,
        kernel_size=kernel_size,
        dropout=dropout,
    )

    # Check that parameters are set correctly
    assert tcn_network.n_blocks == n_blocks
    assert tcn_network.kernel_size == kernel_size
    assert tcn_network.dropout == dropout


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_tcn_network_single_layer():
    """Test TCN network with single temporal block."""
    import tensorflow as tf

    input_shape = (30, 2)  # (n_timepoints, n_channels)
    n_blocks = [16]  # Single layer

    tcn_network = TCNNetwork(n_blocks=n_blocks)
    input_layer, output_layer = tcn_network.build_network(input_shape)

    # Verify single layer network works
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    assert model is not None

    # Test with dummy data
    import numpy as np

    dummy_input = np.random.random((4,) + input_shape)
    output = model(dummy_input)
    assert output.shape == (4, input_shape[1])  # (batch_size, n_channels)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_tcn_final_dense_weights_shape():
    """Test that the final Dense layer in TCN has the correct weight matrix shape."""
    import numpy as np
    import tensorflow as tf

    input_shape = (40, 6)  # (n_timepoints, n_channels)
    n_blocks = [32, 64]
    batch_size = 8

    tcn_network = TCNNetwork(n_blocks=n_blocks)
    input_layer, output_layer = tcn_network.build_network(input_shape)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Run model to build weights
    dummy_input = np.random.random((batch_size,) + input_shape).astype(np.float32)
    _ = model(dummy_input)

    # Directly access last layer (assuming it's Dense)
    dense_layer = model.layers[-1]
    assert isinstance(
        dense_layer, tf.keras.layers.Dense
    ), f"Expected last layer to be Dense, got {type(dense_layer)}"

    weight_shape = dense_layer.kernel.shape
    input_dim, output_dim = weight_shape

    expected_input_dim = n_blocks[-1]
    expected_output_dim = input_shape[1]

    assert (
        input_dim == expected_input_dim
    ), f"Expected input dim {expected_input_dim}, got {input_dim}"
    assert (
        output_dim == expected_output_dim
    ), f"Expected output dim {expected_output_dim}, got {output_dim}"
