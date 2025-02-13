"""Tests for the CNN Model."""

import pytest

from aeon.networks import TimeCNNNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies

__maintainer__ = []


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_cnn_input_shape_padding():
    """Test of CNN network with input_shape < 60."""
    input_shape = (40, 2)
    network = TimeCNNNetwork()
    input_layer, output_layer = network.build_network(input_shape=input_shape)

    assert input_layer is not None
    assert output_layer is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
def test_cnn_activation(activation):
    """Test activation configuration handling."""
    input_shape = (100, 5)
    network = TimeCNNNetwork(activation=activation)
    input_layer, output_layer = network.build_network(input_shape=input_shape)

    assert input_layer is not None
    assert output_layer is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("kernel_size,n_layers", [(7, 2), ([5, 3], 2)])
def test_cnn_kernel_size(kernel_size, n_layers):
    """Test kernel size configuration with different layer counts."""
    input_shape = (100, 5)
    network = TimeCNNNetwork(n_layers=n_layers, kernel_size=kernel_size)
    input_layer, output_layer = network.build_network(input_shape=input_shape)

    assert input_layer is not None
    assert output_layer is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("n_layers,n_filters", [(2, [8, 16]), (1, [12])])
def test_cnn_n_filters(n_layers, n_filters):
    """Test filter configuration handling."""
    input_shape = (100, 5)
    network = TimeCNNNetwork(n_layers=n_layers, n_filters=n_filters)
    input_layer, output_layer = network.build_network(input_shape=input_shape)

    assert input_layer is not None
    assert output_layer is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "pool_size,pool_strides", [(3, None), ([2, 3], 2), (4, [2, 2])]
)
def test_pooling(pool_size, pool_strides):
    """Test pooling configuration with different stride settings."""
    input_shape = (100, 5)
    network = TimeCNNNetwork(avg_pool_size=pool_size, strides_pooling=pool_strides)
    input_layer, output_layer = network.build_network(input_shape=input_shape)

    assert input_layer is not None
    assert output_layer is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_cnn_padding():
    """Test padding override behavior for small inputs."""
    # Test automatic padding override
    small_input = (50, 3)
    network = TimeCNNNetwork(padding="valid")
    input_layer, _ = network.build_network(input_shape=small_input)
    assert input_layer is not None

    # Test explicit padding
    large_input = (100, 3)
    network = TimeCNNNetwork(padding=["same", "valid"])
    input_layer, _ = network.build_network(input_shape=large_input)
    assert input_layer is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("dilation", [2, [1, 2]])
def test_dilation_rate(dilation):
    """Test dilation rate configuration."""
    input_shape = (100, 5)
    network = TimeCNNNetwork(dilation_rate=dilation)
    input_layer, output_layer = network.build_network(input_shape=input_shape)

    assert input_layer is not None
    assert output_layer is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("use_bias", [True, [True, False], False])
def test_use_bias(use_bias):
    """Test bias usage configuration."""
    input_shape = (100, 5)
    network = TimeCNNNetwork(use_bias=use_bias)
    input_layer, output_layer = network.build_network(input_shape=input_shape)

    assert input_layer is not None
    assert output_layer is not None


# Error case tests
@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_invalid_parameter_lengths():
    """Test parameter validation for list lengths."""
    with pytest.raises(ValueError):
        TimeCNNNetwork(n_layers=2, kernel_size=[5]).build_network((100, 5))

    with pytest.raises(ValueError):
        TimeCNNNetwork(n_layers=3, activation=["relu", "sigmoid"]).build_network(
            (100, 5)
        )
