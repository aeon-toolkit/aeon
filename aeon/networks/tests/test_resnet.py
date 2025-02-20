"""Tests for the ResNet Model."""

import pytest
import tensorflow as tf

from aeon.networks import ResNetNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_default_initialization():
    """Test if the network initializes with proper attributes."""
    model = ResNetNetwork()
    assert model.n_residual_blocks == 3
    assert model.n_conv_per_residual_block == 3
    assert model.n_filters is None
    assert model.kernel_size is None
    assert model.strides == 1
    assert model.dilation_rate == 1
    assert model.activation == "relu"
    assert model.use_bias is True
    assert model.padding == "same"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_custom_initialization():
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
    assert model.n_residual_blocks == 3
    assert model.n_conv_per_residual_block == 3
    assert model._n_filters == [64, 128, 128]
    assert model._kernel_size == [8, 5, 3]
    assert model._activation == ["relu", "relu", "relu"]
    assert model._strides == [1, 1, 1]
    assert model._padding == ["same", "same", "same"]


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_edge_case_initialization():
    """Tests edge cases for minimal configuration."""
    model = ResNetNetwork(
        n_residual_blocks=1,
        n_conv_per_residual_block=1,
        n_filters=[64],
        kernel_size=[8],
    )
    model.build_network((128, 1))
    assert model.n_residual_blocks == 1
    assert model.n_conv_per_residual_block == 1
    assert model._n_filters == [64]
    assert model._kernel_size == [8]


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_invalid_initialization():
    """Test if the network raises valid exceptions for invalid configurations."""
    with pytest.raises(ValueError):
        ResNetNetwork(n_filters=[64, 128], n_residual_blocks=3).build_network((128, 1))

    with pytest.raises(ValueError):
        ResNetNetwork(kernel_size=[8, 5], n_conv_per_residual_block=3).build_network(
            (128, 1)
        )

    with pytest.raises(ValueError):
        ResNetNetwork(strides=[1, 2], n_conv_per_residual_block=3).build_network(
            (128, 1)
        )

    with pytest.raises(ValueError):
        ResNetNetwork(dilation_rate=[1, 2], n_conv_per_residual_block=3).build_network(
            (128, 1)
        )

    with pytest.raises(ValueError):
        ResNetNetwork(
            padding=["same", "valid"], n_conv_per_residual_block=3
        ).build_network((128, 1))

    with pytest.raises(ValueError):
        ResNetNetwork(
            activation=["relu", "tanh"], n_conv_per_residual_block=3
        ).build_network((128, 1))

    with pytest.raises(ValueError):
        ResNetNetwork(
            use_bias=[True, False], n_conv_per_residual_block=3
        ).build_network((128, 1))


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_list_parameters():
    """Test correct handling of list parameters."""
    model = ResNetNetwork(
        n_residual_blocks=3,
        n_conv_per_residual_block=3,
        n_filters=[64, 128, 128],
        kernel_size=[8, 5, 3],
        strides=[1, 1, 1],
        dilation_rate=[1, 1, 1],
        padding=["same", "same", "same"],
        activation=["relu", "relu", "relu"],
        use_bias=[True, True, True],
    )
    model.build_network((128, 1))

    assert model._n_filters == [64, 128, 128]
    assert model._kernel_size == [8, 5, 3]
    assert model._strides == [1, 1, 1]
    assert model._dilation_rate == [1, 1, 1]
    assert model._padding == ["same", "same", "same"]
    assert model._activation == ["relu", "relu", "relu"]
    assert model._use_bias == [True, True, True]


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_build_network():
    """Test network building with various input shapes."""
    model = ResNetNetwork()

    input_shapes = [(128, 1), (256, 3), (512, 1)]
    for shape in input_shapes:
        input_layer, output_layer = model.build_network(shape)
        assert input_layer.shape[1:] == shape
        assert output_layer.shape[-1] == 128


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_shortcut_layer():
    """Test the shortcut layer functionality."""
    model = ResNetNetwork()
    input_tensor = tf.keras.layers.Input((128, 64))
    output_tensor = tf.keras.layers.Conv1D(128, 8, padding="same")(input_tensor)

    shortcut = model._shortcut_layer(input_tensor, output_tensor)
    assert shortcut.shape[-1] == 128
