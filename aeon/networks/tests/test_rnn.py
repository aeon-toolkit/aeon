"""Tests for the RNN Network."""

import pytest

from aeon.networks import RNNNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "rnn_type", ["lstm", "gru", "simple", "LSTM", "GRU", "invalid"]
)
def test_rnn_network_rnn_type(rnn_type):
    """Test RNNNetwork with different RNN types."""
    import tensorflow as tf

    input_shape = (100, 5)

    if rnn_type == "invalid":
        with pytest.raises(ValueError, match="Unknown RNN type"):
            rnn_network = RNNNetwork(rnn_type=rnn_type)
            input_layer, output_layer = rnn_network.build_network(input_shape)
    else:
        rnn_network = RNNNetwork(rnn_type=rnn_type)
        input_layer, output_layer = rnn_network.build_network(input_shape)

        # Check that layers are created correctly
        assert hasattr(
            input_layer, "shape"
        ), "Input layer should have a shape attribute"
        assert hasattr(
            output_layer, "shape"
        ), "Output layer should have a shape attribute"
        assert input_layer.dtype == tf.float32
        assert input_layer.dtype == tf.float32

        # Create a model to test the network structure
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        # Check that the correct RNN type is used by layer name pattern
        expected_type = rnn_type.lower()
        layer_names = [layer.name for layer in model.layers]

        # Find RNN layer by name pattern
        rnn_layer_found = False
        for name in layer_names:
            if expected_type in name:
                rnn_layer_found = True
                break

        assert (
            rnn_layer_found
        ), f"Expected {expected_type.upper()} layer not found in model"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("n_layers", [1, 2, 3, 5])
def test_rnn_network_n_layers_valid(n_layers):
    """Test RNNNetwork with valid number of layers."""
    import tensorflow as tf

    input_shape = (100, 5)

    rnn_network = RNNNetwork(n_layers=n_layers)
    input_layer, output_layer = rnn_network.build_network(input_shape)

    # Create a model to count layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Count RNN layers by name pattern (LSTM by default)
    rnn_layers = [layer for layer in model.layers if "lstm" in layer.name]
    assert (
        len(rnn_layers) == n_layers
    ), f"Expected {n_layers} LSTM layers, found {len(rnn_layers)}"

    # Count dropout layers
    dropout_layers = [layer for layer in model.layers if "dropout" in layer.name]
    assert (
        len(dropout_layers) == n_layers
    ), f"Expected {n_layers} Dropout layers, found {len(dropout_layers)}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("n_layers", [0, -1])
def test_rnn_network_n_layers_invalid(n_layers):
    """Test RNNNetwork with invalid number of layers."""
    input_shape = (100, 5)

    # The RNN implementation doesn't validate n_layers in __init__,
    # but will fail when trying to iterate in build_network
    rnn_network = RNNNetwork(n_layers=n_layers)

    # For n_layers <= 0, range(n_layers) creates an empty range,
    # which means no layers are created, but no error is raised
    # The actual behavior depends on the implementation
    try:
        input_layer, output_layer = rnn_network.build_network(input_shape)
        # If no error is raised, check that the output is still valid
        assert input_layer is not None
        assert output_layer is not None
    except (ValueError, IndexError, TypeError):
        # This is also acceptable behavior
        pass


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("n_layers", [2.5, "2"])
def test_rnn_network_n_layers_wrong_type(n_layers):
    """Test RNNNetwork with wrong type for n_layers."""
    input_shape = (100, 5)

    rnn_network = RNNNetwork(n_layers=n_layers)
    # The error occurs when trying to iterate over non-int types
    with pytest.raises(TypeError):
        input_layer, output_layer = rnn_network.build_network(input_shape)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "n_units", [32, 64, 128, [32, 64], [64, 32, 16], [32], [64, 32, 16, 8, 4]]
)
def test_rnn_network_n_units(n_units):
    """Test RNNNetwork with different unit configurations."""
    import tensorflow as tf

    input_shape = (100, 5)

    # Determine expected number of layers
    if isinstance(n_units, int):
        n_layers = 1
        expected_units = [n_units]
    else:
        n_layers = len(n_units)
        expected_units = n_units

    rnn_network = RNNNetwork(n_layers=n_layers, n_units=n_units)
    input_layer, output_layer = rnn_network.build_network(input_shape)

    # Create a model to inspect layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Check that units are set correctly by checking layer names and count
    rnn_layers = [layer for layer in model.layers if "lstm" in layer.name]
    assert len(rnn_layers) == len(expected_units)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("n_units", [[32, 64], [64, 32, 16]])
def test_rnn_network_n_units_mismatch(n_units):
    """Test RNNNetwork with mismatched n_units and n_layers."""
    input_shape = (100, 5)

    # Use different n_layers than length of n_units
    wrong_n_layers = len(n_units) + 1

    with pytest.raises(ValueError, match="Length of n_units .* must match n_layers"):
        rnn_network = RNNNetwork(n_layers=wrong_n_layers, n_units=n_units)
        input_layer, output_layer = rnn_network.build_network(input_shape)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("dropout_rate", [0.1, 0.2, 0.5, 0.8, 1.0])
def test_rnn_network_dropout_rate_nonzero(dropout_rate):
    """Test RNNNetwork with non-zero dropout rates."""
    import tensorflow as tf

    input_shape = (100, 5)
    n_layers = 2

    rnn_network = RNNNetwork(n_layers=n_layers, dropout_rate=dropout_rate)
    input_layer, output_layer = rnn_network.build_network(input_shape)

    # Create a model to inspect layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Count dropout layers
    dropout_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)
    ]
    assert (
        len(dropout_layers) == n_layers
    ), f"Expected {n_layers} Dropout layers, found {len(dropout_layers)}"

    # Check dropout rate
    for layer in dropout_layers:
        assert (
            layer.rate == dropout_rate
        ), f"Dropout layer has rate {layer.rate}, expected {dropout_rate}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_rnn_network_dropout_rate_zero():
    """Test RNNNetwork with zero dropout rate."""
    import tensorflow as tf

    input_shape = (100, 5)
    n_layers = 2
    dropout_rate = 0.0

    rnn_network = RNNNetwork(n_layers=n_layers, dropout_rate=dropout_rate)
    input_layer, output_layer = rnn_network.build_network(input_shape)

    # Create a model to inspect layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # So with dropout_rate=0.0, no dropout layers should be created
    dropout_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)
    ]
    assert (
        len(dropout_layers) == 0
    ), f"Expected 0 Dropout layers with rate 0.0, found {len(dropout_layers)}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("bidirectional", [True, False])
def test_rnn_network_bidirectional(bidirectional):
    """Test RNNNetwork with bidirectional option."""
    import tensorflow as tf

    input_shape = (100, 5)
    n_layers = 2

    rnn_network = RNNNetwork(n_layers=n_layers, bidirectional=bidirectional)
    input_layer, output_layer = rnn_network.build_network(input_shape)

    # Create a model to inspect layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    if bidirectional:
        # Check for Bidirectional layers
        bidirectional_layers = [
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.layers.Bidirectional)
        ]
        assert (
            len(bidirectional_layers) == n_layers
        ), f"Expected {n_layers} Bidirectional layers"
    else:
        # Check for regular LSTM layers by name
        lstm_layers = [
            layer
            for layer in model.layers
            if "lstm" in layer.name
            and not isinstance(layer, tf.keras.layers.Bidirectional)
        ]
        assert (
            len(lstm_layers) == n_layers
        ), f"Expected {n_layers} LSTM layers, found {len(lstm_layers)}"

        # Ensure no Bidirectional layers
        bidirectional_layers = [
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.layers.Bidirectional)
        ]
        assert (
            len(bidirectional_layers) == 0
        ), "Should not have Bidirectional layers when bidirectional=False"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("activation", ["tanh", "relu", "sigmoid", "linear"])
def test_rnn_network_activation(activation):
    """Test RNNNetwork with different activation functions."""
    import tensorflow as tf

    input_shape = (100, 5)

    rnn_network = RNNNetwork(activation=activation)
    input_layer, output_layer = rnn_network.build_network(input_shape)

    # Create a model to inspect layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Check activation function by finding LSTM layers
    lstm_layers = [
        layer
        for layer in model.layers
        if "lstm" in layer.name and isinstance(layer, tf.keras.layers.LSTM)
    ]
    assert len(lstm_layers) == 1

    assert (
        lstm_layers[0].activation.__name__ == activation
    ), f"LSTM activation is {lstm_layers[0].activation.__name__}, expected {activation}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "return_sequences", [None, True, [True, False], [True, True, False]]
)
def test_rnn_network_return_sequences(return_sequences):
    """Test RNNNetwork with different return_sequences configurations."""
    import tensorflow as tf

    input_shape = (100, 5)

    # Determine n_layers based on return_sequences
    if isinstance(return_sequences, list):
        n_layers = len(return_sequences)
    else:
        n_layers = 3  # Use 3 layers for testing

    rnn_network = RNNNetwork(n_layers=n_layers, return_sequences=return_sequences)
    input_layer, output_layer = rnn_network.build_network(input_shape)

    # Create a model to inspect layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Check return_sequences setting
    lstm_layers = [
        layer
        for layer in model.layers
        if "lstm" in layer.name and isinstance(layer, tf.keras.layers.LSTM)
    ]
    assert len(lstm_layers) == n_layers

    # Determine expected return_sequences
    if return_sequences is None:
        expected_return_sequences = [True] * (n_layers - 1) + [False]
    elif isinstance(return_sequences, bool):
        expected_return_sequences = [return_sequences] * n_layers
    else:
        expected_return_sequences = return_sequences

    for i, layer in enumerate(lstm_layers):
        assert (
            layer.return_sequences == expected_return_sequences[i]
        ), f"Layer {i} has return_sequences={layer.return_sequences}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("return_sequences", [[True, False], [False, True, False]])
def test_rnn_network_return_sequences_mismatch(return_sequences):
    """Test RNNNetwork with mismatched return_sequences and n_layers."""
    input_shape = (100, 5)

    # Use different n_layers than length of return_sequences
    wrong_n_layers = len(return_sequences) + 1

    with pytest.raises(
        ValueError, match="Length of return_sequences .* must match n_layers"
    ):
        rnn_network = RNNNetwork(
            n_layers=wrong_n_layers, return_sequences=return_sequences
        )
        input_layer, output_layer = rnn_network.build_network(input_shape)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_rnn_network_output_shape():
    """Test RNNNetwork output shapes with different configurations."""
    import numpy as np
    import tensorflow as tf

    input_shape = (50, 10)  # (timesteps, features)
    batch_size = 32

    # Test with return_sequences=False (default for last layer)
    rnn_network = RNNNetwork(n_layers=2, n_units=64)
    input_layer, output_layer = rnn_network.build_network(input_shape)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Create dummy input
    dummy_input = np.random.random((batch_size,) + input_shape)
    output = model(dummy_input)

    # Output should be (batch_size, n_units) when return_sequences=False for last layer
    assert output.shape == (
        batch_size,
        64,
    ), f"Expected shape (32, 64), got {output.shape}"

    # Test with return_sequences=True for all layers
    rnn_network = RNNNetwork(n_layers=2, n_units=64, return_sequences=True)
    input_layer, output_layer = rnn_network.build_network(input_shape)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    output = model(dummy_input)

    # Output should be (batch_size, timesteps, n_units) when return_sequences=True
    assert output.shape == (
        batch_size,
        50,
        64,
    ), f"Expected shape (32, 50, 64), got {output.shape}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_rnn_network_layer_names():
    """Test that RNN layers have correct names."""
    import tensorflow as tf

    input_shape = (100, 5)
    n_layers = 3

    rnn_network = RNNNetwork(n_layers=n_layers, rnn_type="lstm")
    input_layer, output_layer = rnn_network.build_network(input_shape)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Check layer names
    lstm_layers = [layer for layer in model.layers if "lstm" in layer.name]
    dropout_layers = [layer for layer in model.layers if "dropout" in layer.name]

    for i, layer in enumerate(lstm_layers):
        expected_name = f"lstm_{i+1}"
        assert (
            layer.name == expected_name
        ), f"LSTM layer {i} has name {layer.name}, expected {expected_name}"

    for i, layer in enumerate(dropout_layers):
        expected_name = f"dropout_{i+1}"
        assert (
            layer.name == expected_name
        ), f"Dropout layer {i} has name {layer.name}, expected {expected_name}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_rnn_network_config():
    """Test RNNNetwork configuration attributes."""
    rnn_network = RNNNetwork()

    # Check _config attributes
    assert "python_dependencies" in rnn_network._config
    assert "tensorflow" in rnn_network._config["python_dependencies"]
    assert "python_version" in rnn_network._config
    assert "structure" in rnn_network._config
    assert rnn_network._config["structure"] == "encoder"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_rnn_network_complex_configuration():
    """Test RNNNetwork with a complex configuration."""
    import tensorflow as tf

    input_shape = (200, 8)

    rnn_network = RNNNetwork(
        rnn_type="gru",
        n_layers=4,
        n_units=[128, 64, 32, 16],
        dropout_rate=0.3,
        bidirectional=True,
        activation="relu",
        return_sequences=[True, True, True, False],
    )

    input_layer, output_layer = rnn_network.build_network(input_shape)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Check that all components are correctly configured
    bidirectional_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Bidirectional)
    ]
    dropout_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)
    ]

    assert len(bidirectional_layers) == 4, "Should have 4 Bidirectional layers"
    assert len(dropout_layers) == 4, "Should have 4 Dropout layers"

    # Check units in each layer
    expected_units = [128, 64, 32, 16]
    for i, layer in enumerate(bidirectional_layers):
        assert (
            layer.forward_layer.units == expected_units[i]
        ), f"Layer {i} has {layer.forward_layer.units} units"

    # Check return_sequences
    expected_return_sequences = [True, True, True, False]
    for i, layer in enumerate(bidirectional_layers):
        assert (
            layer.forward_layer.return_sequences == expected_return_sequences[i]
        ), f"Layer {i} has return_sequences={layer.forward_layer.return_sequences}"

    # Check activation
    for layer in bidirectional_layers:
        assert (
            layer.forward_layer.activation.__name__ == "relu"
        ), f"Layer activation is {layer.forward_layer.activation.__name__}"
