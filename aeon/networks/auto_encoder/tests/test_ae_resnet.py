"""Tests for the AEResNetNetwork Model."""

import pytest

from aeon.networks import AEResNetNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "activation",
    [
        "relu",  # Test with relu activation
        "sigmoid",  # Test with sigmoid activation
        "tanh",  # Test with tanh activation
        ["relu", "tanh", "sigmoid"],  # Test with variable activation
    ],
)
def test_ae_res_unit_activation(activation):
    """Test whether AEResNetNetwork initializes correctly with different activations."""
    import tensorflow as tf

    aeres = AEResNetNetwork(
        activation=activation,
    )
    aeres.build_network((1000, 5))
    autoencoder = aeres.get_autoencoder_model()

    if isinstance(activation, str):
        activation = [activation] * 3
    else:
        activation = activation + activation[::-1]

    # iterate over every activation layer and check they are right
    i = 0
    for layer in autoencoder.layers:
        if isinstance(layer, tf.keras.layers.Activation):
            assert layer.activation.__name__ == activation[i]
            i += 1


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "use_bias, n_conv_per_residual_block",
    [
        ([True, False, True], 3),  # list case
        (True, 3),  # scalar broadcast case
        pytest.param(
            [True, False], 4, marks=pytest.mark.xfail(raises=ValueError)
        ),  # error case
    ],
)
def test_use_bias(use_bias, n_conv_per_residual_block):
    """Test AEResNetNetwork use_bias configurations."""
    aeres = AEResNetNetwork(
        latent_space_dim=128,
        use_bias=use_bias,
        n_conv_per_residual_block=n_conv_per_residual_block,
    )
    encoder, decoder = aeres.build_network((1000, 5))
    assert encoder is not None
    assert decoder is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "n_filters, n_residual_blocks",
    [
        (64, 3),  # scalar case
        ([64, 128, 256], 3),  # list case matching residual blocks
        pytest.param(
            [64, 128], 3, marks=pytest.mark.xfail(raises=ValueError)
        ),  # error case
    ],
)
def test_n_filters(n_filters, n_residual_blocks):
    """Test AEResNetNetwork n_filters configurations."""
    aeres = AEResNetNetwork(
        latent_space_dim=128,
        n_filters=n_filters,
        n_residual_blocks=n_residual_blocks,
    )
    encoder, decoder = aeres.build_network((1000, 5))
    assert encoder is not None
    assert decoder is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "kernel_size, n_conv_per_residual_block",
    [
        (8, 3),  # scalar case
        ([8, 5, 3], 3),  # list case matching conv layers
        pytest.param(
            [8, 5], 3, marks=pytest.mark.xfail(raises=ValueError)
        ),  # error case
    ],
)
def test_kernel_size(kernel_size, n_conv_per_residual_block):
    """Test AEResNetNetwork kernel_size configurations."""
    aeres = AEResNetNetwork(
        latent_space_dim=128,
        kernel_size=kernel_size,
        n_conv_per_residual_block=n_conv_per_residual_block,
    )
    encoder, decoder = aeres.build_network((1000, 5))
    assert encoder is not None
    assert decoder is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "strides, n_conv_per_residual_block",
    [
        (1, 3),  # scalar case
        pytest.param(
            [1, 2], 3, marks=pytest.mark.xfail(raises=ValueError)
        ),  # error case
    ],
)
def test_strides(strides, n_conv_per_residual_block):
    """Test AEResNetNetwork strides configurations."""
    aeres = AEResNetNetwork(
        latent_space_dim=128,
        strides=strides,
        n_conv_per_residual_block=n_conv_per_residual_block,
    )
    encoder, decoder = aeres.build_network((1000, 5))
    assert encoder is not None
    assert decoder is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "dilation_rate, n_conv_per_residual_block",
    [
        (1, 3),  # scalar case
        pytest.param(
            [1, 2], 3, marks=pytest.mark.xfail(raises=ValueError)
        ),  # error case
    ],
)
def test_dilation_rate(dilation_rate, n_conv_per_residual_block):
    """Test AEResNetNetwork dilation_rate configurations."""
    aeres = AEResNetNetwork(
        latent_space_dim=128,
        dilation_rate=dilation_rate,
        n_conv_per_residual_block=n_conv_per_residual_block,
    )
    encoder, decoder = aeres.build_network((1000, 5))
    assert encoder is not None
    assert decoder is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "padding, n_conv_per_residual_block",
    [
        ("same", 3),  # scalar case
        #    (['same', 'valid', 'same'], 3),  # list case matching conv layers
        pytest.param(
            ["same", "valid"], 3, marks=pytest.mark.xfail(raises=ValueError)
        ),  # error case
    ],
)
def test_padding(padding, n_conv_per_residual_block):
    """Test AEResNetNetwork padding configurations."""
    aeres = AEResNetNetwork(
        latent_space_dim=128,
        padding=padding,
        n_conv_per_residual_block=n_conv_per_residual_block,
    )
    encoder, decoder = aeres.build_network((1000, 5))
    assert encoder is not None
    assert decoder is not None
