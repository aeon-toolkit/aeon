"""Tests for the MLPNetwork Model."""

import pytest

from aeon.networks import MLPNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "n_layers, n_units, activation",
    [
        (3, 500, "relu"),
        (5, [256, 128, 128, 64, 32], "sigmoid"),
        (2, 128, ["tanh", "relu"]),
    ],
)
def test_mlp_initialization(n_layers, n_units, activation):
    """Test whether MLPNetwork initializes correctly with different configurations."""
    from tensorflow.keras.layers import Dense, Dropout, Flatten, InputLayer
    from tensorflow.keras.models import Model

    mlp = MLPNetwork(n_layers=n_layers, n_units=n_units, activation=activation)
    input_layer, output_layer = mlp.build_network((1000, 5))

    # Wrap in a Model to access internal layers
    model = Model(inputs=input_layer, outputs=output_layer)
    layers = model.layers

    assert isinstance(layers[0], InputLayer), "Expected first layer to be InputLayer"

    assert isinstance(layers[1], Flatten), "Expected second layer to be Flatten"

    # Check dropout and dense layers ordering
    for i in range(n_layers):
        dropout_layer = layers[2 + 2 * i]  # Dropout before Dense
        dense_layer = layers[3 + 2 * i]  # Dense comes after Dropout

        assert isinstance(
            dropout_layer, Dropout
        ), f"Expected Dropout at index {2 + 2 * i}"
        assert isinstance(dense_layer, Dense), f"Expected Dense at index {3 + 2 * i}"

        # Assert activation function
        expected_activation = (
            activation[i] if isinstance(activation, list) else activation
        )
        assert dense_layer.activation.__name__ == expected_activation, (
            f"Expected activation {expected_activation}, "
            f"got {dense_layer.activation.__name__}"
        )

        # Assert number of units
        expected_units = n_units[i] if isinstance(n_units, list) else n_units
        assert (
            dense_layer.units == expected_units
        ), f"Expected {expected_units} units, got {dense_layer.units}"

    # Check last layer is Dropout
    assert isinstance(layers[-1], Dropout), "Expected final layer to be Dropout"

    # Assert model parameters (Just for show)
    assert mlp.n_layers == n_layers
    assert mlp.n_units == n_units
    assert mlp.activation == activation


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "dropout_rate, n_layers",
    [
        (0.2, 3),
        ([0.1, 0.2, 0.3], 3),
        pytest.param([0.1, 0.2], 3, marks=pytest.mark.xfail(raises=AssertionError)),
    ],
)
def test_mlp_dropout_rate(dropout_rate, n_layers):
    """Test MLPNetwork dropout_rate configurations."""
    from tensorflow.keras.layers import Dense, Dropout, Flatten, InputLayer
    from tensorflow.keras.models import Model

    mlp = MLPNetwork(n_layers=n_layers, dropout_rate=dropout_rate)
    input_layer, output_layer = mlp.build_network((1000, 5))

    # Wrap in a Model to access internal layers
    model = Model(inputs=input_layer, outputs=output_layer)
    layers = model.layers

    # Check first two layers
    assert isinstance(layers[0], InputLayer), "Expected first layer to be InputLayer"
    assert isinstance(layers[1], Flatten), "Expected second layer to be Flatten"

    # Check dropout and dense layers ordering
    for i in range(n_layers):
        dropout_layer = layers[2 + 2 * i]
        dense_layer = layers[3 + 2 * i]

        assert isinstance(
            dropout_layer, Dropout
        ), f"Expected Dropout at index {2 + 2 * i}"
        assert isinstance(dense_layer, Dense), f"Expected Dense at index {3 + 2 * i}"

        # Assert dropout rates match expected values
        expected_dropout = (
            dropout_rate[i] if isinstance(dropout_rate, list) else dropout_rate
        )
        assert (
            dropout_layer.rate == expected_dropout
        ), f"Expected {expected_dropout},got {dropout_layer.rate}"
    assert isinstance(layers[-1], Dropout), "Expected final layer to be Dropout"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "dropout_last",
    [0.3, 0.5, pytest.param(1.2, marks=pytest.mark.xfail(raises=AssertionError))],
)
def test_mlp_dropout_last(dropout_last):
    """Test MLPNetwork dropout_last configurations."""
    from tensorflow.keras.layers import Dropout, Flatten, InputLayer
    from tensorflow.keras.models import Model

    mlp = MLPNetwork(dropout_last=dropout_last)
    input_layer, output_layer = mlp.build_network((1000, 5))

    # Wrap in a Model to access internal layers
    model = Model(inputs=input_layer, outputs=output_layer)
    layers = model.layers

    assert isinstance(layers[0], InputLayer), "Expected first layer to be InputLayer"
    assert isinstance(layers[1], Flatten), "Expected second layer to be Flatten"
    assert isinstance(layers[-1], Dropout), "Expected final layer to be Dropout"

    assert (
        layers[-1].rate == dropout_last
    ), f"Expected {dropout_last}, got {layers[-1].rate}"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("use_bias", [True, False])
def test_mlp_use_bias(use_bias):
    """Test MLPNetwork use_bias configurations."""
    from tensorflow.keras.layers import Dense, Dropout, Flatten, InputLayer
    from tensorflow.keras.models import Model

    mlp = MLPNetwork(use_bias=use_bias)
    input_layer, output_layer = mlp.build_network((1000, 5))

    # Wrap in a Model to access internal layers
    model = Model(inputs=input_layer, outputs=output_layer)
    layers = model.layers

    assert isinstance(layers[0], InputLayer), "Expected first layer to be InputLayer"
    assert isinstance(layers[1], Flatten), "Expected second layer to be Flatten"
    assert isinstance(layers[-1], Dropout), "Expected final layer to be Dropout"

    # Find the last Dense layer before the final Dropout layer
    last_dense_layer = next(
        (layer for layer in reversed(layers) if isinstance(layer, Dense)), None
    )

    assert last_dense_layer is not None, "No Dense layer found before final Dropout"
    assert isinstance(last_dense_layer, Dense), "Expected last layer to be Dense"

    assert (
        last_dense_layer.use_bias == use_bias
    ), f"Expected use_bias {use_bias}, got {last_dense_layer.use_bias}"
