"""Test for the FCNNetwork class."""

import pytest

from aeon.networks import FCNNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_fcnnetwork_valid():
    """Test FCNNetwork with valid configurations."""
    input_shape = (100, 5)
    model = FCNNetwork(n_layers=3)
    input_layer, output_layer = model.build_network(input_shape)

    assert hasattr(input_layer, "shape")
    assert hasattr(output_layer, "shape")


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "activation, should_raise",
    [
        (["relu", "sigmoid", "tanh"], False),
        (["relu", "sigmoid"], True),
        (
            ["relu", "sigmoid", "tanh", "softmax"],
            True,
        ),
        ("relu", False),
        ("sigmoid", False),
        ("tanh", False),
        ("softmax", False),
    ],
)
def test_fcnnetwork_activation(activation, should_raise):
    """Test FCNNetwork with valid and invalid activation configurations."""
    input_shape = (100, 5)
    if should_raise:
        with pytest.raises(ValueError):
            model = FCNNetwork(activation=activation)
            model.build_network(input_shape)
    else:
        model = FCNNetwork(activation=activation)
        input_layer, output_layer = model.build_network(input_shape)

        assert hasattr(input_layer, "shape")

        assert hasattr(output_layer, "shape")


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "kernel_size, should_raise",
    [
        ([3, 1, 2], False),
        ([1, 3], True),
        ([3, 1, 1, 3], True),
        (3, False),
    ],
)
def test_fcnnetwork_kernel_size(kernel_size, should_raise):
    """Test FCNNetwork with valid and invalid kernel_size configurations."""
    input_shape = (100, 5)
    if should_raise:
        with pytest.raises(ValueError):
            model = FCNNetwork(kernel_size=kernel_size, n_layers=3)
            model.build_network(input_shape)
    else:
        model = FCNNetwork(kernel_size=kernel_size, n_layers=3)
        input_layer, output_layer = model.build_network(input_shape)

        assert hasattr(input_layer, "shape")
        assert hasattr(output_layer, "shape")


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "dilation_rate, should_raise",
    [
        ([1, 2, 1], False),
        ([1, 4], True),
        ([1, 2, 4, 1], True),
        (1, False),
    ],
)
def test_fcnnetwork_dilation_rate(dilation_rate, should_raise):
    """Test FCNNetwork with valid and invalid dilation_rate configurations."""
    input_shape = (100, 5)
    if should_raise:
        with pytest.raises(ValueError):
            model = FCNNetwork(dilation_rate=dilation_rate, n_layers=3)
            model.build_network(input_shape)
    else:
        model = FCNNetwork(dilation_rate=dilation_rate, n_layers=3)
        input_layer, output_layer = model.build_network(input_shape)

        assert hasattr(input_layer, "shape")
        assert hasattr(output_layer, "shape")


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "strides, should_raise",
    [
        ([1, 2, 3], False),
        ([1, 1], True),
        ([1, 2, 2, 1], True),
        (1, False),
    ],
)
def test_fcnnetwork_strides(strides, should_raise):
    """Test FCNNetwork with valid and invalid strides configurations."""
    input_shape = (100, 5)
    if should_raise:
        with pytest.raises(ValueError):
            model = FCNNetwork(strides=strides, n_layers=3)
            model.build_network(input_shape)
    else:
        model = FCNNetwork(strides=strides, n_layers=3)
        input_layer, output_layer = model.build_network(input_shape)

        assert hasattr(input_layer, "shape")
        assert hasattr(output_layer, "shape")


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "padding, should_raise",
    [
        (["same", "same", "valid"], False),
        (["valid", "same"], True),
        (["same", "valid", "same", "valid"], True),
        ("same", False),
        ("valid", False),
    ],
)
def test_fcnnetwork_padding(padding, should_raise):
    """Test FCNNetwork with valid and invalid padding configurations."""
    input_shape = (100, 5)
    if should_raise:
        with pytest.raises(ValueError):
            model = FCNNetwork(padding=padding, n_layers=3)
            model.build_network(input_shape)
    else:
        model = FCNNetwork(padding=padding, n_layers=3)
        input_layer, output_layer = model.build_network(input_shape)

        assert hasattr(input_layer, "shape")
        assert hasattr(output_layer, "shape")


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "n_filters, should_raise",
    [
        ([32, 64, 128], False),  # Valid case with a list of filters
        ([32, 64], True),  # Invalid case with fewer filters than layers
        ([32, 64, 128, 256], True),  # Invalid case with more filters than layers
        (32, False),  # Valid case with a single filter value
    ],
)
def test_fcnnetwork_n_filters(n_filters, should_raise):
    """Test FCNNetwork with valid and invalid n_filters configurations."""
    input_shape = (100, 5)
    if should_raise:
        with pytest.raises(ValueError):
            model = FCNNetwork(n_filters=n_filters, n_layers=3)
            model.build_network(input_shape)
    else:
        model = FCNNetwork(n_filters=n_filters, n_layers=3)
        input_layer, output_layer = model.build_network(input_shape)

        assert hasattr(input_layer, "shape")
        assert hasattr(output_layer, "shape")
