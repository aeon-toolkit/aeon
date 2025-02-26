"""Test for the AEFCNNetwork class."""

import pytest

from aeon.networks import AEFCNNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_aefcn_default():
    """Default testing for aefcn."""
    model = AEFCNNetwork()
    assert model.latent_space_dim == 128
    assert model.temporal_latent_space is False
    assert model.n_layers == 3
    assert model.n_filters is None
    assert model.kernel_size is None
    assert model.activation == "relu"
    assert model.padding == "same"
    assert model.strides == 1
    assert model.dilation_rate == 1
    assert model.use_bias is True


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("latent_space_dim", [64, 128, 256])
def test_aefcn_latent_space(latent_space_dim):
    """Test AEFCNNetwork with different latent space dimensions."""
    import tensorflow as tf

    aefcn = AEFCNNetwork(latent_space_dim=latent_space_dim)
    encoder, decoder = aefcn.build_network((1000, 5))
    assert isinstance(encoder, tf.keras.models.Model)
    assert isinstance(decoder, tf.keras.models.Model)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "kernel_size, should_raise",
    [
        ([8, 5, 3], False),
        (3, False),
        ([5, 5], True),
        ([3, 3, 3, 3], True),
    ],
)
def test_aefcnnetwork_kernel_size(kernel_size, should_raise):
    """Test AEFCNNetwork with different kernel sizes."""
    import tensorflow as tf

    if should_raise:
        with pytest.raises(
            ValueError,
            match="Number of kernels .* should be the same as number of layers",
        ):
            AEFCNNetwork(kernel_size=kernel_size, n_layers=3).build_network((1000, 5))
    else:
        aefcn = AEFCNNetwork(kernel_size=kernel_size, n_layers=3)
        encoder, decoder = aefcn.build_network((1000, 5))
        assert isinstance(encoder, tf.keras.models.Model)
        assert isinstance(decoder, tf.keras.models.Model)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "n_filters, should_raise",
    [
        ([128, 256, 128], False),
        (32, False),
        ([32, 64], True),
        ([16, 32, 64, 128], True),
    ],
)
def test_aefcnnetwork_n_filters(n_filters, should_raise):
    """Test AEFCNNetwork with different number of filters."""
    import tensorflow as tf

    if should_raise:
        with pytest.raises(
            ValueError,
            match="Number of filters .* should be the same as number of layers",
        ):
            AEFCNNetwork(n_filters=n_filters, n_layers=3).build_network((1000, 5))
    else:
        aefcn = AEFCNNetwork(n_filters=n_filters, n_layers=3)
        encoder, decoder = aefcn.build_network((1000, 5))
        assert isinstance(encoder, tf.keras.models.Model)
        assert isinstance(decoder, tf.keras.models.Model)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "dilation_rate, should_raise",
    [
        ([1, 2, 1], False),
        (2, False),
        ([1, 2], True),
        ([1, 2, 2, 1], True),
    ],
)
def test_aefcnnetwork_dilation_rate(dilation_rate, should_raise):
    """Test AEFCNNetwork with different dilation rates."""
    import tensorflow as tf

    if should_raise:
        with pytest.raises(
            ValueError,
            match="Number of dilations .* should be the same as number of layers",
        ):
            AEFCNNetwork(dilation_rate=dilation_rate, n_layers=3).build_network(
                (1000, 5)
            )
    else:
        aefcn = AEFCNNetwork(dilation_rate=dilation_rate, n_layers=3)
        encoder, decoder = aefcn.build_network((1000, 5))
        assert isinstance(encoder, tf.keras.models.Model)
        assert isinstance(decoder, tf.keras.models.Model)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "strides, should_raise",
    [
        ([1, 2, 1], False),
        (2, False),
        ([1, 2], True),
        ([1, 2, 2, 1], True),
    ],
)
def test_aefcnnetwork_strides(strides, should_raise):
    """Test AEFCNNetwork with different strides."""
    import tensorflow as tf

    if should_raise:
        with pytest.raises(
            ValueError,
            match="Number of strides .* should be the same as number of layers",
        ):
            AEFCNNetwork(strides=strides, n_layers=3).build_network((1000, 5))
    else:
        aefcn = AEFCNNetwork(strides=strides, n_layers=3)
        encoder, decoder = aefcn.build_network((1000, 5))
        assert isinstance(encoder, tf.keras.models.Model)
        assert isinstance(decoder, tf.keras.models.Model)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "padding, should_raise",
    [
        (["same", "valid", "same"], False),
        ("same", False),
        (["same", "valid"], True),
        (
            ["same", "valid", "same", "valid"],
            True,
        ),
    ],
)
def test_aefcnnetwork_padding(padding, should_raise):
    """Test AEFCNNetwork with different paddings."""
    import tensorflow as tf

    if should_raise:
        with pytest.raises(
            ValueError,
            match="Number of paddings .* should be the same as number of layers",
        ):
            AEFCNNetwork(padding=padding, n_layers=3).build_network((1000, 5))
    else:
        aefcn = AEFCNNetwork(padding=padding, n_layers=3)
        encoder, decoder = aefcn.build_network((1000, 5))
        assert isinstance(encoder, tf.keras.models.Model)
        assert isinstance(decoder, tf.keras.models.Model)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "activation, should_raise",
    [
        (["relu", "sigmoid", "tanh"], False),
        ("sigmoid", False),
        (["relu", "sigmoid"], True),
        (
            ["relu", "sigmoid", "tanh", "softmax"],
            True,
        ),
    ],
)
def test_aefcnnetwork_activation(activation, should_raise):
    """Test AEFCNNetwork with different activations."""
    import tensorflow as tf

    if should_raise:
        with pytest.raises(
            ValueError,
            match="Number of activations .* should be the same as number of layers",
        ):
            AEFCNNetwork(activation=activation, n_layers=3).build_network((1000, 5))
    else:
        aefcn = AEFCNNetwork(activation=activation, n_layers=3)
        encoder, decoder = aefcn.build_network((1000, 5))
        assert isinstance(encoder, tf.keras.models.Model)
        assert isinstance(decoder, tf.keras.models.Model)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "use_bias, should_raise",
    [
        ([True, False, True], False),
        (True, False),
        ([True, False], True),
        ([True, False, True, False], True),
    ],
)
def test_aefcnnetwork_use_bias(use_bias, should_raise):
    """Test AEFCNNetwork with different use_bias values."""
    import tensorflow as tf

    if should_raise:
        with pytest.raises(
            ValueError,
            match="Number of biases .* should be the same as number of layers",
        ):
            AEFCNNetwork(use_bias=use_bias, n_layers=3).build_network((1000, 5))
    else:
        aefcn = AEFCNNetwork(use_bias=use_bias, n_layers=3)
        encoder, decoder = aefcn.build_network((1000, 5))
        assert isinstance(encoder, tf.keras.models.Model)
        assert isinstance(decoder, tf.keras.models.Model)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("temporal_latent_space", [True, False])
def test_aefcnnetwork_temporal_latent_space(temporal_latent_space):
    """Test for temporal latent space."""
    import tensorflow as tf

    input_shape = (1000, 5)

    aefcn = AEFCNNetwork(
        latent_space_dim=128, temporal_latent_space=temporal_latent_space
    )

    encoder, decoder = aefcn.build_network(input_shape)

    assert isinstance(encoder, tf.keras.models.Model)
    assert isinstance(decoder, tf.keras.models.Model)

    if temporal_latent_space:
        assert any(
            isinstance(layer, tf.keras.layers.Conv1D) for layer in encoder.layers
        ), "Expected Conv1D layer in encoder but not found."
    else:
        assert any(
            isinstance(layer, tf.keras.layers.Dense) for layer in decoder.layers
        ), "Expected Dense layer in decoder but not found."
