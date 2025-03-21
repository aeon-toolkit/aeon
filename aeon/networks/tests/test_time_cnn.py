"""Tests for the TimeCNNNetwork Model."""

import pytest

from aeon.networks import TimeCNNNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_time_cnn_input_shape_padding():
    """Test of CNN network with input_shape < 60."""
    input_shape = (40, 2)
    network = TimeCNNNetwork()
    input_layer, output_layer = network.build_network(input_shape=input_shape)

    assert hasattr(input_layer, "shape")
    assert hasattr(output_layer, "shape")


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "activation, n_layers, should_raise",
    [
        ("relu", 2, False),
        ("sigmoid", 2, False),
        ("tanh", 2, False),
        (["relu", "sigmoid", "tanh"], 2, True),
        (["relu"], 2, True),
    ],
)
def test_time_cnn_activation(activation, n_layers, should_raise):
    """Test activation configuration handling."""
    input_shape = (100, 5)
    if should_raise:
        with pytest.raises(ValueError):
            network = TimeCNNNetwork(activation=activation, n_layers=n_layers)
            network.build_network(input_shape=input_shape)
    else:
        network = TimeCNNNetwork(activation=activation, n_layers=n_layers)
        input_layer, output_layer = network.build_network(input_shape=input_shape)

        assert hasattr(input_layer, "shape")
        assert hasattr(output_layer, "shape")


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "kernel_size, n_layers, should_raise",
    [
        (7, 2, False),
        ([5, 3], 2, False),
        ([5, 3, 2], 2, True),
        ([5], 2, True),
    ],
)
def test_time_cnn_kernel_size(kernel_size, n_layers, should_raise):
    """Test kernel size configuration with different layer counts."""
    input_shape = (100, 5)
    if should_raise:
        with pytest.raises(ValueError):
            network = TimeCNNNetwork(n_layers=n_layers, kernel_size=kernel_size)
            network.build_network(input_shape=input_shape)
    else:
        network = TimeCNNNetwork(n_layers=n_layers, kernel_size=kernel_size)
        input_layer, output_layer = network.build_network(input_shape=input_shape)

        assert hasattr(input_layer, "shape")
        assert hasattr(output_layer, "shape")


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "n_layers,n_filters,should_raise",
    [
        (2, [8, 16], False),
        (1, [12, 10, 4], True),
        (2, 8, False),
        (3, [8], True),
    ],
)
def test_time_cnn_n_filters(n_layers, n_filters, should_raise):
    """Test filter configuration handling."""
    input_shape = (100, 5)
    if should_raise:
        with pytest.raises(ValueError):
            network = TimeCNNNetwork(n_layers=n_layers, n_filters=n_filters)
            network.build_network(input_shape=input_shape)
    else:
        network = TimeCNNNetwork(n_layers=n_layers, n_filters=n_filters)
        input_layer, output_layer = network.build_network(input_shape=input_shape)

        assert hasattr(input_layer, "shape")
        assert hasattr(output_layer, "shape")


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "avg_pool_size, n_layers, should_raise",
    [
        (3, 2, False),
        ([2, 3], 2, False),
        ([2, 3, 4], 2, True),
        ([2], 2, True),
    ],
)
def test_time_cnn_avg_pool_size(avg_pool_size, n_layers, should_raise):
    """Test average pool size configuration."""
    input_shape = (100, 5)
    if should_raise:
        with pytest.raises(ValueError):
            network = TimeCNNNetwork(avg_pool_size=avg_pool_size, n_layers=n_layers)
            network.build_network(input_shape=input_shape)
    else:
        network = TimeCNNNetwork(avg_pool_size=avg_pool_size, n_layers=n_layers)
        input_layer, output_layer = network.build_network(input_shape=input_shape)

        assert hasattr(input_layer, "shape")
        assert hasattr(output_layer, "shape")


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "strides_pooling, n_layers, should_raise",
    [
        (None, 2, False),
        (2, 2, False),
        ([2, 3], 2, False),
        ([2, 3, 4], 2, True),
        ([2], 2, True),
    ],
)
def test_time_cnn_strides_pooling(strides_pooling, n_layers, should_raise):
    """Test strides pooling configuration."""
    input_shape = (100, 5)
    if should_raise:
        with pytest.raises(ValueError):
            network = TimeCNNNetwork(strides_pooling=strides_pooling, n_layers=n_layers)
            network.build_network(input_shape=input_shape)
    else:
        network = TimeCNNNetwork(strides_pooling=strides_pooling, n_layers=n_layers)
        input_layer, output_layer = network.build_network(input_shape=input_shape)

        assert hasattr(input_layer, "shape")
        assert hasattr(output_layer, "shape")


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "padding, n_layers, should_raise",
    [
        ("valid", 2, False),
        ("same", 2, False),
        (["same", "valid"], 2, False),
        (["same", "valid", "same"], 2, True),
        (["same"], 2, True),
    ],
)
def test_time_cnn_padding(padding, n_layers, should_raise):
    """Test padding override behavior for different inputs."""
    input_shape = (100, 5)
    if should_raise:
        with pytest.raises(ValueError):
            network = TimeCNNNetwork(padding=padding, n_layers=n_layers)
            network.build_network(input_shape=input_shape)
    else:
        network = TimeCNNNetwork(padding=padding, n_layers=n_layers)
        input_layer, output_layer = network.build_network(input_shape=input_shape)
        assert hasattr(input_layer, "shape")
        assert hasattr(output_layer, "shape")


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "dilation, n_layers, should_raise",
    [
        (2, 2, False),
        ([1, 2], 2, False),
        ([1, 2, 3], 2, True),
        ([1], 2, True),
    ],
)
def test_time_cnn_dilation_rate(dilation, n_layers, should_raise):
    """Test dilation rate configuration."""
    input_shape = (100, 5)
    if should_raise:
        with pytest.raises(ValueError):
            network = TimeCNNNetwork(dilation_rate=dilation, n_layers=n_layers)
            network.build_network(input_shape=input_shape)
    else:
        network = TimeCNNNetwork(dilation_rate=dilation, n_layers=n_layers)
        input_layer, output_layer = network.build_network(input_shape=input_shape)

        assert hasattr(input_layer, "shape")
        assert hasattr(output_layer, "shape")


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "strides, n_layers, should_raise",
    [
        (1, 2, False),
        ([1, 2], 2, False),
        ([1, 2, 3], 2, True),
        ([1], 2, True),
    ],
)
def test_time_cnn_strides(strides, n_layers, should_raise):
    """Test strides configuration."""
    input_shape = (100, 5)
    if should_raise:
        with pytest.raises(ValueError):
            network = TimeCNNNetwork(strides=strides, n_layers=n_layers)
            network.build_network(input_shape=input_shape)
    else:
        network = TimeCNNNetwork(strides=strides, n_layers=n_layers)
        input_layer, output_layer = network.build_network(input_shape=input_shape)

        assert hasattr(input_layer, "shape")
        assert hasattr(output_layer, "shape")


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "use_bias, n_layers, should_raise",
    [
        (True, 2, False),
        ([True, False], 2, False),
        ([True, False, True], 2, True),
        ([True], 2, True),
    ],
)
def test_time_cnn_use_bias(use_bias, n_layers, should_raise):
    """Test bias usage configuration."""
    input_shape = (100, 5)
    if should_raise:
        with pytest.raises(ValueError):
            network = TimeCNNNetwork(use_bias=use_bias, n_layers=n_layers)
            network.build_network(input_shape=input_shape)
    else:
        network = TimeCNNNetwork(use_bias=use_bias, n_layers=n_layers)
        input_layer, output_layer = network.build_network(input_shape=input_shape)

        assert hasattr(input_layer, "shape")
        assert hasattr(output_layer, "shape")
