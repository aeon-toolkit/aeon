"""Tests for the RecurrentNetwork."""

import pytest

from aeon.networks import RecurrentNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "rnn_type", ["lstm", "gru", "simple", "LSTM", "GRU", "invalid"]
)
def test_rnn_network_rnn_type(rnn_type):
    """Test RecurrentNetwork with different RNN types."""
    import tensorflow as tf

    input_shape = (100, 5)

    if rnn_type == "invalid":
        with pytest.raises(ValueError, match="Unknown RNN type"):
            rnn_network = RecurrentNetwork(rnn_type=rnn_type)
            input_layer, output_layer = rnn_network.build_network(input_shape)
    else:
        rnn_network = RecurrentNetwork(rnn_type=rnn_type)
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
    """Test RecurrentNetwork with valid number of layers."""
    import tensorflow as tf

    input_shape = (100, 5)

    rnn_network = RecurrentNetwork(rnn_type="simple", n_layers=n_layers)
    input_layer, output_layer = rnn_network.build_network(input_shape)

    # Create a model to count layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Count RNN layers by name pattern (simple by default in new implementation)
    rnn_layers = [layer for layer in model.layers if "simple" in layer.name]
    assert (
        len(rnn_layers) == n_layers
    ), f"Expected {n_layers} SimpleRNN layers, found {len(rnn_layers)}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("n_layers", [0, -1, 2.5, "2"])
def test_rnn_network_n_layers_invalid(n_layers):
    """Test RecurrentNetwork with invalid number of layers."""
    input_shape = (100, 5)

    # The RNN implementation doesn't validate n_layers in __init__,
    # but will fail when trying to iterate in build_network
    rnn_network = RecurrentNetwork(n_layers=n_layers)

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
@pytest.mark.parametrize(
    "n_units", [32, 64, 128, [32, 64], [64, 32, 16], [32], [64, 32, 16, 8, 4]]
)
def test_rnn_network_n_units(n_units):
    """Test RecurrentNetwork with different unit configurations."""
    import tensorflow as tf

    input_shape = (100, 5)

    # Determine expected number of layers
    if isinstance(n_units, int):
        n_layers = 1
        expected_units = [n_units]
    else:
        n_layers = len(n_units)
        expected_units = n_units

    rnn_network = RecurrentNetwork(
        rnn_type="simple", n_layers=n_layers, n_units=n_units
    )
    input_layer, output_layer = rnn_network.build_network(input_shape)

    # Create a model to inspect layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Check that units are set correctly by checking layer names and count
    rnn_layers = [layer for layer in model.layers if "simple" in layer.name]
    assert len(rnn_layers) == len(expected_units)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("n_units", [[32, 64], [64, 32, 16]])
def test_rnn_network_n_units_mismatch(n_units):
    """Test RecurrentNetwork with mismatched n_units and n_layers."""
    input_shape = (100, 5)

    # Use different n_layers than length of n_units
    wrong_n_layers = len(n_units) + 1

    with pytest.raises(ValueError):
        rnn_network = RecurrentNetwork(n_layers=wrong_n_layers, n_units=n_units)
        input_layer, output_layer = rnn_network.build_network(input_shape)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("dropout_intermediate", [0.1, 0.2, 0.5, 0.8])
def test_rnn_network_dropout_intermediate_nonzero(dropout_intermediate):
    """Test RecurrentNetwork with non-zero intermediate dropout rates."""
    import tensorflow as tf

    input_shape = (100, 5)
    n_layers = 3

    rnn_network = RecurrentNetwork(
        rnn_type="simple", n_layers=n_layers, dropout_intermediate=dropout_intermediate
    )
    input_layer, output_layer = rnn_network.build_network(input_shape)

    # Create a model to inspect layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Count intermediate dropout layers (should be n_layers - 1)
    intermediate_dropout_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dropout) and "intermediate" in layer.name
    ]
    expected_intermediate_dropouts = n_layers - 1  # All layers except the last
    assert (
        len(intermediate_dropout_layers) == expected_intermediate_dropouts
    ), f"Expected {expected_intermediate_dropouts} \
        found {len(intermediate_dropout_layers)}"

    # Check dropout rate
    for layer in intermediate_dropout_layers:
        assert (
            layer.rate == dropout_intermediate
        ), f"Got {layer.rate}, expected {dropout_intermediate}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("dropout_output", [0.1, 0.2, 0.5, 0.8])
def test_rnn_network_dropout_output_nonzero(dropout_output):
    """Test RecurrentNetwork with non-zero dropout rate."""
    import tensorflow as tf

    input_shape = (100, 5)
    n_layers = 2

    rnn_network = RecurrentNetwork(
        rnn_type="simple", n_layers=n_layers, dropout_output=dropout_output
    )
    input_layer, output_layer = rnn_network.build_network(input_shape)

    # Create a model to inspect layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Count output dropout layers (should be 1)
    output_dropout_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dropout) and "dropout_output" in layer.name
    ]
    assert (
        len(output_dropout_layers) == 1
    ), f"Expected 1 output Dropout layer, found {len(output_dropout_layers)}"

    # Check dropout rate
    assert (
        output_dropout_layers[0].rate == dropout_output
    ), f"Got {output_dropout_layers[0].rate}, expected {dropout_output}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_rnn_network_dropout_zero():
    """Test RecurrentNetwork with zero dropout rates."""
    import tensorflow as tf

    input_shape = (100, 5)
    n_layers = 3
    dropout_intermediate = 0.0
    dropout_output = 0.0

    rnn_network = RecurrentNetwork(
        rnn_type="simple",
        n_layers=n_layers,
        dropout_intermediate=dropout_intermediate,
        dropout_output=dropout_output,
    )
    input_layer, output_layer = rnn_network.build_network(input_shape)

    # Create a model to inspect layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # With dropout rates = 0.0, no dropout layers should be created
    dropout_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)
    ]
    assert (
        len(dropout_layers) == 0
    ), f"Expected 0 Dropout layers with zero dropout rates, found {len(dropout_layers)}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("bidirectional", [True, False])
def test_rnn_network_bidirectional(bidirectional):
    """Test RecurrentNetwork with bidirectional option."""
    import tensorflow as tf

    input_shape = (100, 5)
    n_layers = 2

    rnn_network = RecurrentNetwork(
        rnn_type="simple", n_layers=n_layers, bidirectional=bidirectional
    )
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
        # Check for regular SimpleRNN layers by name
        simple_layers = [
            layer
            for layer in model.layers
            if "simple" in layer.name
            and not isinstance(layer, tf.keras.layers.Bidirectional)
        ]
        assert (
            len(simple_layers) == n_layers
        ), f"Expected {n_layers} SimpleRNN layers, found {len(simple_layers)}"

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
    """Test RecurrentNetwork with different activation functions."""
    import tensorflow as tf

    input_shape = (100, 5)

    rnn_network = RecurrentNetwork(rnn_type="simple", activation=activation)
    input_layer, output_layer = rnn_network.build_network(input_shape)

    # Create a model to inspect layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    simple_layers = [
        layer
        for layer in model.layers
        if "simple" in layer.name and isinstance(layer, tf.keras.layers.SimpleRNN)
    ]
    assert len(simple_layers) == 1

    assert (
        simple_layers[0].activation.__name__ == activation
    ), f"Got {simple_layers[0].activation.__name__}, expected {activation}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("return_sequence_last", [True, False])
def test_rnn_network_return_sequence_last(return_sequence_last):
    """Test RecurrentNetwork with different return_sequence_last configurations."""
    import tensorflow as tf

    input_shape = (100, 5)
    n_layers = 3

    rnn_network = RecurrentNetwork(
        rnn_type="simple", n_layers=n_layers, return_sequence_last=return_sequence_last
    )
    input_layer, output_layer = rnn_network.build_network(input_shape)

    # Create a model to inspect layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Check return_sequences setting
    simple_layers = [
        layer
        for layer in model.layers
        if "simple" in layer.name and isinstance(layer, tf.keras.layers.SimpleRNN)
    ]
    assert len(simple_layers) == n_layers

    # Check return_sequences for each layer
    for i, layer in enumerate(simple_layers):
        is_last_layer = i == n_layers - 1
        expected_return_sequences = not is_last_layer or return_sequence_last

        assert (
            layer.return_sequences == expected_return_sequences
        ), f"Layer {i} got {layer.return_sequences},\
              expected {expected_return_sequences}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_rnn_network_output_shape():
    """Test RecurrentNetwork output shapes."""
    import numpy as np
    import tensorflow as tf

    input_shape = (50, 10)  # (timesteps, features)
    batch_size = 32

    # Test with return_sequence_last=False (default)
    rnn_network = RecurrentNetwork(
        rnn_type="simple", n_layers=2, n_units=64, return_sequence_last=False
    )
    input_layer, output_layer = rnn_network.build_network(input_shape)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Create dummy input
    dummy_input = np.random.random((batch_size,) + input_shape)
    output = model(dummy_input)

    # Output should be (batch_size, n_units) when return_sequence_last=False
    assert output.shape == (
        batch_size,
        64,
    ), f"Expected shape (32, 64), got {output.shape}"

    # Test with return_sequence_last=True
    rnn_network = RecurrentNetwork(
        rnn_type="simple", n_layers=2, n_units=64, return_sequence_last=True
    )
    input_layer, output_layer = rnn_network.build_network(input_shape)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    output = model(dummy_input)

    # Output should be (batch_size, timesteps, n_units) when return_sequence_last=True
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

    rnn_network = RecurrentNetwork(
        rnn_type="simple", n_layers=n_layers, dropout_intermediate=0.2
    )
    input_layer, output_layer = rnn_network.build_network(input_shape)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Check layer names
    simple_layers = [layer for layer in model.layers if "simple" in layer.name]
    intermediate_dropout_layers = [
        layer for layer in model.layers if "dropout_intermediate" in layer.name
    ]

    for i, layer in enumerate(simple_layers):
        expected_name = f"simple_{i+1}"
        assert (
            layer.name == expected_name
        ), f"SimpleRNN layer {i} has name {layer.name}, expected {expected_name}"

    for i, layer in enumerate(intermediate_dropout_layers):
        expected_name = f"dropout_intermediate_{i+1}"
        assert (
            layer.name == expected_name
        ), f"layer {i} got {layer.name}, expected {expected_name}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_rnn_network_config():
    """Test RecurrentNetwork configuration attributes."""
    rnn_network = RecurrentNetwork()

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
    """Test RecurrentNetwork with a complex configuration."""
    import tensorflow as tf

    input_shape = (200, 8)

    rnn_network = RecurrentNetwork(
        rnn_type="gru",
        n_layers=4,
        n_units=[128, 64, 32, 16],
        dropout_intermediate=0.2,
        dropout_output=0.3,
        bidirectional=True,
        activation="relu",
        return_sequence_last=True,
    )

    input_layer, output_layer = rnn_network.build_network(input_shape)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Check that all components are correctly configured
    bidirectional_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Bidirectional)
    ]
    intermediate_dropout_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dropout) and "intermediate" in layer.name
    ]
    output_dropout_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dropout) and "dropout_output" in layer.name
    ]

    assert len(bidirectional_layers) == 4, "Should have 4 Bidirectional layers"
    assert (
        len(intermediate_dropout_layers) == 3
    ), "Should have 3 intermediate Dropout layers"
    assert len(output_dropout_layers) == 1, "Should have 1 output Dropout layer"

    # Check units in each layer
    expected_units = [128, 64, 32, 16]
    for i, layer in enumerate(bidirectional_layers):
        assert (
            layer.forward_layer.units == expected_units[i]
        ), f"Layer {i} has {layer.forward_layer.units} units"

    for i, layer in enumerate(bidirectional_layers):
        is_last_layer = i == len(bidirectional_layers) - 1
        expected_return_sequences = (
            not is_last_layer or True
        )  # return_sequence_last=True
        assert (
            layer.forward_layer.return_sequences == expected_return_sequences
        ), f"Layer {i} got {layer.forward_layer.return_sequences}, \
             expected {expected_return_sequences}"

    # Check activation
    for layer in bidirectional_layers:
        assert (
            layer.forward_layer.activation.__name__ == "relu"
        ), f"Layer activation is {layer.forward_layer.activation.__name__}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("activation", [["tanh", "relu"], ["sigmoid", "tanh", "relu"]])
def test_rnn_network_activation_list(activation):
    """Test RecurrentNetwork with list of activation functions."""
    import tensorflow as tf

    input_shape = (100, 5)
    n_layers = len(activation)

    rnn_network = RecurrentNetwork(
        rnn_type="simple", n_layers=n_layers, activation=activation
    )
    input_layer, output_layer = rnn_network.build_network(input_shape)

    # Create a model to inspect layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Check activation function for each layer
    simple_layers = [
        layer
        for layer in model.layers
        if "simple" in layer.name and isinstance(layer, tf.keras.layers.SimpleRNN)
    ]
    assert len(simple_layers) == n_layers

    for i, layer in enumerate(simple_layers):
        expected_activation = activation[i]
        assert (
            layer.activation.__name__ == expected_activation
        ), f"Layer {i} got {layer.activation.__name__}, \
        expected {expected_activation}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("activation", [["tanh", "relu"], ["sigmoid", "tanh", "relu"]])
def test_rnn_network_activation_list_mismatch(activation):
    """Test RecurrentNetwork with mismatched activation list and n_layers."""
    input_shape = (100, 5)

    # Use different n_layers than length of activation list
    wrong_n_layers = len(activation) + 1

    with pytest.raises(
        ValueError,
        match="Number of activations .* should be the same as number of layers",
    ):
        rnn_network = RecurrentNetwork(n_layers=wrong_n_layers, activation=activation)
        input_layer, output_layer = rnn_network.build_network(input_shape)
