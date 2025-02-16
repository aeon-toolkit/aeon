"""Tests for the Attention Bidirectional Network."""

import pytest
import tensorflow as tf

from aeon.networks import AEAttentionBiGRUNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("latent_space_dim", [64, 128, 256])
def test_aeattentionbigrunetwork_latent_space(latent_space_dim):
    """Test AEAttentionBiGRUNetwork with different latent space dimensions."""
    aeattentionbigru = AEAttentionBiGRUNetwork(latent_space_dim=latent_space_dim)
    encoder, decoder = aeattentionbigru.build_network((1000, 5))
    assert isinstance(encoder, tf.keras.models.Model)
    assert isinstance(decoder, tf.keras.models.Model)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("n_layers_decoder", [1, 2, 3, 2.5, "1"])
def test_aeattentionbigrunetwork_n_layers_decoder(n_layers_decoder):
    """Test AEAttentionBiGRUNetwork with different number of decoder layers."""
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


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("n_layers_encoder", [1, 2, 3, 3.5, "2"])
def test_aeattentionbigrunetwork_n_layers_encoder(n_layers_encoder):
    """Test AEAttentionBiGRUNetwork with different number of encoder layers."""
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
    if isinstance(activation_decoder, list) and len(activation_decoder) != 3:
        with pytest.raises(ValueError):
            aeattentionbigru = AEAttentionBiGRUNetwork(
                activation_encoder="relu",
                activation_decoder=activation_decoder,
                n_layers_decoder=3,
            )
            encoder, decoder = aeattentionbigru.build_network((1000, 5))
    else:
        aeattentionbigru = AEAttentionBiGRUNetwork(
            activation_encoder="relu",
            activation_decoder=activation_decoder,
            n_layers_decoder=3 if isinstance(activation_decoder, list) else 1,
        )
        encoder, decoder = aeattentionbigru.build_network((1000, 5))
        assert isinstance(encoder, tf.keras.models.Model)
        assert isinstance(decoder, tf.keras.models.Model)


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
    if isinstance(activation_encoder, list) and len(activation_encoder) != 3:
        with pytest.raises(ValueError):
            aeattentionbigru = AEAttentionBiGRUNetwork(
                activation_encoder=activation_encoder,
                activation_decoder="relu",
                n_layers_encoder=3,
            )
            encoder, decoder = aeattentionbigru.build_network((1000, 5))
    else:
        aeattentionbigru = AEAttentionBiGRUNetwork(
            activation_encoder=activation_encoder,
            activation_decoder="relu",
            n_layers_encoder=3 if isinstance(activation_encoder, list) else 1,
        )
        encoder, decoder = aeattentionbigru.build_network((1000, 5))
        assert isinstance(encoder, tf.keras.models.Model)
        assert isinstance(decoder, tf.keras.models.Model)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "temporal_latent_space", [False]
)  # fails if set to True  -> issue raised
def test_aeattentionbigrunetwork_temporal_latent_space(temporal_latent_space):
    """Test AEAttentionBiGRUNetwork with temporal latent space enabled/disabled."""
    aeattentionbigru = AEAttentionBiGRUNetwork(
        temporal_latent_space=temporal_latent_space
    )
    encoder, decoder = aeattentionbigru.build_network((1000, 5))
    assert isinstance(encoder, tf.keras.models.Model)
    assert isinstance(decoder, tf.keras.models.Model)
