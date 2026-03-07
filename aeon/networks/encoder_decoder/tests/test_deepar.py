"""Tests for the DeepAR Network."""

import pytest

from aeon.networks import DeepARNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_default_initialization():
    """Test if the network initializes with proper default attributes."""
    model = DeepARNetwork()

    assert model.lstm_units is None
    assert model.dense_units is None
    assert model.dropout == 0.1


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_custom_initialization():
    """Test whether custom parameters are correctly set."""
    model = DeepARNetwork(
        lstm_units=50,
        dense_units=25,
        dropout=0.2,
    )

    assert model.lstm_units == 50
    assert model.dense_units == 25
    assert model.dropout == 0.2


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_calculate_units():
    """Test the _calculate_units method with different feature sizes."""
    model = DeepARNetwork()

    # Test with 1 feature
    lstm_units, dense_units = model._calculate_units(1)
    assert lstm_units >= 4
    assert dense_units >= 4

    # Test with multiple features
    lstm_units, dense_units = model._calculate_units(5)
    assert isinstance(lstm_units, int)
    assert isinstance(dense_units, int)
    assert lstm_units >= 4
    assert dense_units >= 4

    # Test with custom units
    model_custom = DeepARNetwork(lstm_units=32, dense_units=16)
    lstm_units, dense_units = model_custom._calculate_units(10)
    assert lstm_units == 32
    assert dense_units == 16


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_build_network_basic():
    """Test basic network building functionality."""
    model = DeepARNetwork()
    input_shape = (10, 1)  # 10 time steps, 1 feature

    input_layer, output = model.build_network(input_shape)

    assert input_layer is not None
    assert output is not None
    assert input_layer.shape == (None, 10, 1)

    # Output should have shape (batch_size, 2, n_features)
    # where 2 represents [mean, sigma] and n_features is 1
    assert output.shape == (None, 2, 1)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_build_network_multivariate():
    """Test network building with multiple features."""
    model = DeepARNetwork()
    input_shape = (20, 3)  # 20 time steps, 3 features

    input_layer, output = model.build_network(input_shape)

    assert input_layer is not None
    assert output is not None
    assert input_layer.shape == (None, 20, 3)
    assert output.shape == (None, 2, 3)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_build_network_custom_parameters():
    """Test network building with custom parameters."""
    model = DeepARNetwork(
        lstm_units=64,
        dense_units=32,
        dropout=0.3,
    )
    input_shape = (15, 2)

    input_layer, output = model.build_network(input_shape)

    assert input_layer is not None
    assert output is not None
    assert input_layer.shape == (None, 15, 2)
    assert output.shape == (None, 2, 2)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_edge_case_parameters():
    """Test edge cases for parameters."""
    # Test with minimum values
    model = DeepARNetwork(
        lstm_units=1,
        dense_units=1,
        dropout=0.0,
    )
    input_shape = (2, 1)  # Minimal input shape

    input_layer, output = model.build_network(input_shape)

    assert input_layer is not None
    assert output is not None
    assert output.shape == (None, 2, 1)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_invalid_parameters():
    """Test if network handles invalid parameters appropriately."""
    # Test with invalid dropout rate
    with pytest.raises((ValueError, TypeError)):
        model = DeepARNetwork(dropout=-0.1)
        model.build_network((10, 1))

    # Test with invalid dropout rate (> 1)
    with pytest.raises((ValueError, TypeError)):
        model = DeepARNetwork(dropout=1.5)
        model.build_network((10, 1))


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_output_structure():
    """Test the structure of the output layer."""
    import tensorflow as tf

    model = DeepARNetwork()
    input_shape = (10, 2)

    input_layer, output = model.build_network(input_shape)

    # Create model and test with dummy data
    full_model = tf.keras.Model(inputs=input_layer, outputs=output)
    dummy_input = tf.random.normal((1, 10, 2))
    prediction = full_model(dummy_input)

    # Check output shape and properties
    assert prediction.shape == (1, 2, 2)
    assert prediction.dtype == tf.float32

    # Extract mean and sigma
    mean = prediction[:, 0, :]
    sigma = prediction[:, 1, :]

    assert mean.shape == (1, 2)
    assert sigma.shape == (1, 2)

    # Sigma should be positive (due to softplus activation)
    assert tf.reduce_all(sigma > 0)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_network_compilation():
    """Test if the network can be compiled and used in a Keras model."""
    import tensorflow as tf

    model = DeepARNetwork()
    input_shape = (10, 1)

    input_layer, output = model.build_network(input_shape)
    keras_model = tf.keras.Model(inputs=input_layer, outputs=output)

    # Test compilation
    keras_model.compile(optimizer="adam", loss="mse")

    # Test model summary (should not raise errors)
    summary = keras_model.summary()
    assert summary is None  # summary() returns None but prints info

    # Test parameter count
    param_count = keras_model.count_params()
    assert param_count > 0
