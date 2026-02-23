"""Tests for the ResNet Model."""

import pytest

from aeon.networks import ResNetNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_resnet_default_initialization():
    """Test if the network initializes with proper attributes."""
    model = ResNetNetwork()
    assert isinstance(
        model, ResNetNetwork
    ), "Model initialization failed: Incorrect type"
    assert model.n_residual_blocks == 3, "Default residual blocks count mismatch"
    assert (
        model.n_conv_per_residual_block == 3
    ), "Default convolution blocks count mismatch"
    assert model.n_filters is None, "Default n_filters should be None"
    assert model.kernel_size is None, "Default kernel_size should be None"
    assert model.strides == 1, "Default strides value mismatch"
    assert model.dilation_rate == 1, "Default dilation rate mismatch"
    assert model.activation == "relu", "Default activation mismatch"
    assert model.use_bias is True, "Default use_bias mismatch"
    assert model.padding == "same", "Default padding mismatch"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_resnet_custom_initialization():
    """Test whether custom kwargs are correctly set."""
    model = ResNetNetwork(
        n_residual_blocks=3,
        n_conv_per_residual_block=3,
        n_filters=[64, 128, 128],
        kernel_size=[8, 5, 3],
        activation="relu",
        strides=1,
        padding="same",
    )
    model.build_network((128, 1))
    assert isinstance(
        model, ResNetNetwork
    ), "Custom initialization failed: Incorrect type"
    assert model._n_filters == [64, 128, 128], "n_filters list mismatch"
    assert model._kernel_size == [8, 5, 3], "kernel_size list mismatch"
    assert model._activation == ["relu", "relu", "relu"], "activation list mismatch"
    assert model._strides == [1, 1, 1], "strides list mismatch"
    assert model._padding == ["same", "same", "same"], "padding list mismatch"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_resnet_invalid_initialization():
    """Test if the network raises valid exceptions for invalid configurations."""
    with pytest.raises(ValueError, match=".*same as number of residual blocks.*"):
        ResNetNetwork(n_filters=[64, 128], n_residual_blocks=3).build_network((128, 1))

    with pytest.raises(ValueError, match=".*same as number of convolution layers.*"):
        ResNetNetwork(kernel_size=[8, 5], n_conv_per_residual_block=3).build_network(
            (128, 1)
        )

    with pytest.raises(ValueError, match=".*same as number of convolution layers.*"):
        ResNetNetwork(strides=[1, 2], n_conv_per_residual_block=3).build_network(
            (128, 1)
        )


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_resnet_build_network():
    """Test network building with various input shapes."""
    model = ResNetNetwork()

    input_shapes = [(128, 1), (256, 3), (512, 1)]
    for shape in input_shapes:
        input_layer, output_layer = model.build_network(shape)
        assert hasattr(input_layer, "shape"), "Input layer type mismatch"
        assert hasattr(output_layer, "shape"), "Output layer type mismatch"
        assert input_layer.shape[1:] == shape, "Input shape mismatch"
        assert output_layer.shape[-1] == 128, "Output layer mismatch"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_resnet_shortcut_layer():
    """Test the shortcut layer functionality."""
    model = ResNetNetwork()

    input_shape = (128, 64)
    input_layer, output_layer = model.build_network(input_shape)

    shortcut = model._shortcut_layer(input_layer, output_layer)

    assert hasattr(shortcut, "shape"), "Shortcut layer output type mismatch"
    assert shortcut.shape[-1] == 128, "Shortcut output shape mismatch"
