"""Tests for the Attention Bidirectional Network."""

import pytest

from aeon.networks import AEAttentionBiGRUNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("latent_space_dim", [64, 128, 256])
def test_aeattentionbigrunetwork_latent_space(latent_space_dim):
    """Test AEAttentionBiGRUNetwork with different latent space dimensions."""
    import tensorflow as tf

    input_shape = (1000, 5)
    aeattentionbigru = AEAttentionBiGRUNetwork(latent_space_dim=latent_space_dim)
    encoder, decoder = aeattentionbigru.build_network(input_shape)

    # Check instance types
    assert isinstance(encoder, tf.keras.models.Model)
    assert isinstance(decoder, tf.keras.models.Model)

    # Check encoder output shape matches the specified latent_space_dim
    dummy_input = tf.keras.layers.Input(shape=input_shape)
    encoder_output = encoder(dummy_input)
    assert encoder_output.shape[-1] == latent_space_dim

    # Check decoder input shape matches the specified latent_space_dim
    decoder_input_shape = decoder.input_shape
    assert decoder_input_shape[-1] == latent_space_dim

    # Test the full autoencoder pipeline
    autoencoder = tf.keras.models.Model(
        inputs=encoder.inputs, outputs=decoder(encoder.outputs[0])
    )
    assert isinstance(autoencoder, tf.keras.models.Model)

    # Check output shape matches input shape
    output = autoencoder(dummy_input)
    assert output.shape[1:] == input_shape


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("n_layers_decoder", [1, 2, 3, 2.5, "1"])
def test_aeattentionbigrunetwork_n_layers_decoder(n_layers_decoder):
    """Test AEAttentionBiGRUNetwork with different number of decoder layers."""
    import tensorflow as tf

    if not isinstance(n_layers_decoder, int):
        with pytest.raises(TypeError):
            aeattentionbigru = AEAttentionBiGRUNetwork(
                n_layers_decoder=n_layers_decoder
            )
            encoder, decoder = aeattentionbigru.build_network((1000, 5))
    else:
        aeattentionbigru = AEAttentionBiGRUNetwork(n_layers_decoder=n_layers_decoder)
        encoder, decoder = aeattentionbigru.build_network((1000, 5))
        assert isinstance(encoder, tf.keras.models.Model)
        assert isinstance(decoder, tf.keras.models.Model)

        # Count the number of Bidirectional layers in the decoder
        bidirectional_layers = [
            layer
            for layer in decoder.layers
            if isinstance(layer, tf.keras.layers.Bidirectional)
        ]
        assert len(bidirectional_layers) == n_layers_decoder, (
            f"Expected {n_layers_decoder} Bidirectional layers in decoder, "
            f"but found {len(bidirectional_layers)}"
        )

        # Check if all Bidirectional layers contain GRU as expected
        for i, layer in enumerate(bidirectional_layers):
            assert isinstance(
                layer.forward_layer, tf.keras.layers.GRU
            ), f"Layer {i} is not using GRU as expected"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("n_layers_encoder", [1, 2, 3, 3.5, "2"])
def test_aeattentionbigrunetwork_n_layers_encoder(n_layers_encoder):
    """Test AEAttentionBiGRUNetwork with different number of encoder layers."""
    import tensorflow as tf

    if not isinstance(n_layers_encoder, int):
        with pytest.raises(TypeError):
            aeattentionbigru = AEAttentionBiGRUNetwork(
                n_layers_encoder=n_layers_encoder
            )
            encoder, decoder = aeattentionbigru.build_network((1000, 5))
    else:
        aeattentionbigru = AEAttentionBiGRUNetwork(n_layers_encoder=n_layers_encoder)
        encoder, decoder = aeattentionbigru.build_network((1000, 5))
        assert isinstance(encoder, tf.keras.models.Model)
        assert isinstance(decoder, tf.keras.models.Model)

        # Count the number of GRU layers in the encoder
        gru_layers = [
            layer for layer in encoder.layers if isinstance(layer, tf.keras.layers.GRU)
        ]

        # Each encoder layer has 2 GRUs (forward and backward)
        assert len(gru_layers) == 2 * n_layers_encoder, (
            f"Expected {2 * n_layers_encoder} GRU layers in encoder, "
            f"but found {len(gru_layers)}"
        )

        # Count attention layers
        attention_layers = [
            layer
            for layer in encoder.layers
            if isinstance(layer, tf.keras.layers.Attention)
        ]
        assert len(attention_layers) == n_layers_encoder, (
            f"Expected {n_layers_encoder} Attention layers, "
            f"but found {len(attention_layers)}"
        )


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "activation_decoder",
    ["relu", "tanh", "sigmoid", ["relu", "tanh"], ["sigmoid", "relu", "tanh"]],
)
def test_aeattentionbigrunetwork_activation_decoder(activation_decoder):
    """Test AEAttentionBiGRUNetwork with different decoder activations."""
    import tensorflow as tf

    n_layers = 3 if isinstance(activation_decoder, list) else 1
    if isinstance(activation_decoder, list) and len(activation_decoder) != n_layers:
        with pytest.raises(ValueError):
            aeattentionbigru = AEAttentionBiGRUNetwork(
                activation_encoder="relu",
                activation_decoder=activation_decoder,
                n_layers_decoder=n_layers,
            )
            encoder, decoder = aeattentionbigru.build_network((1000, 5))
    else:
        aeattentionbigru = AEAttentionBiGRUNetwork(
            activation_encoder="relu",
            activation_decoder=activation_decoder,
            n_layers_decoder=n_layers,
        )
        encoder, decoder = aeattentionbigru.build_network((1000, 5))
        assert isinstance(encoder, tf.keras.models.Model)
        assert isinstance(decoder, tf.keras.models.Model)

        # Check that the activations are correctly set in the Bidirectional GRU layers
        bidirectional_layers = [
            layer
            for layer in decoder.layers
            if isinstance(layer, tf.keras.layers.Bidirectional)
        ]

        expected_activations = (
            activation_decoder
            if isinstance(activation_decoder, list)
            else [activation_decoder] * n_layers
        )

        for i, layer in enumerate(bidirectional_layers):
            assert (
                layer.forward_layer.activation.__name__
                == expected_activations[n_layers - i - 1]
            ), (
                f"Layer {i} has activation {layer.forward_layer.activation.__name__}, "
                f"expected {expected_activations[n_layers-i-1]}"
            )


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "activation_encoder",
    ["relu", "tanh", "sigmoid", ["relu", "tanh"], ["sigmoid", "relu", "tanh"]],
)
def test_aeattentionbigrunetwork_activation_encoder(activation_encoder):
    """Test AEAttentionBiGRUNetwork with different encoder activations."""
    import tensorflow as tf

    n_layers = 3 if isinstance(activation_encoder, list) else 1
    if isinstance(activation_encoder, list) and len(activation_encoder) != n_layers:
        with pytest.raises(ValueError):
            aeattentionbigru = AEAttentionBiGRUNetwork(
                activation_encoder=activation_encoder,
                activation_decoder="relu",
                n_layers_encoder=n_layers,
            )
            encoder, decoder = aeattentionbigru.build_network((1000, 5))
    else:
        aeattentionbigru = AEAttentionBiGRUNetwork(
            activation_encoder=activation_encoder,
            activation_decoder="relu",
            n_layers_encoder=n_layers,
        )
        encoder, decoder = aeattentionbigru.build_network((1000, 5))
        assert isinstance(encoder, tf.keras.models.Model)
        assert isinstance(decoder, tf.keras.models.Model)

        # Extract GRU layers from the encoder
        gru_layers = [
            layer for layer in encoder.layers if isinstance(layer, tf.keras.layers.GRU)
        ]

        expected_activations = (
            activation_encoder
            if isinstance(activation_encoder, list)
            else [activation_encoder] * n_layers
        )

        # Check activations for forward GRU layers (first half of GRU layers)
        for i in range(n_layers):
            assert gru_layers[i * 2].activation.__name__ == expected_activations[i], (
                f"Forward GRU layer {i} has activation "
                f"{gru_layers[i*2].activation.__name__}, expected "
                f"{expected_activations[i]}"
            )


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("temporal_latent_space", [False, True])
def test_aeattentionbigrunetwork_temporal_latent_space(temporal_latent_space):
    """Test AEAttentionBiGRUNetwork with temporal latent space enabled/disabled."""
    import tensorflow as tf

    input_shape = (1000, 5)
    latent_space_dim = 64
    aeattentionbigru = AEAttentionBiGRUNetwork(
        temporal_latent_space=temporal_latent_space, latent_space_dim=latent_space_dim
    )
    encoder, decoder = aeattentionbigru.build_network(input_shape)
    assert isinstance(encoder, tf.keras.models.Model)
    assert isinstance(decoder, tf.keras.models.Model)

    # Check the encoder output shape based on temporal_latent_space setting
    dummy_input = tf.keras.layers.Input(shape=input_shape)
    encoder_output = encoder(dummy_input)

    if temporal_latent_space:
        assert len(encoder_output.shape) == 3
        assert encoder_output.shape[-1] == latent_space_dim
        assert encoder_output.shape[1] == input_shape[0]  # Time dimension preserved
    else:
        assert len(encoder_output.shape) == 2
        assert encoder_output.shape[-1] == latent_space_dim

    # Check decoder input shape
    if temporal_latent_space:
        assert len(decoder.input_shape) == 3
        assert decoder.input_shape[-1] == latent_space_dim
    else:
        assert len(decoder.input_shape) == 2
        assert decoder.input_shape[-1] == latent_space_dim

    # Check for specific layers based on temporal_latent_space setting
    if not temporal_latent_space:
        # Should have a RepeatVector layer in decoder
        repeat_vector_layers = [
            layer
            for layer in decoder.layers
            if isinstance(layer, tf.keras.layers.RepeatVector)
        ]
        assert len(repeat_vector_layers) == 1
        assert repeat_vector_layers[0].n == input_shape[0]
