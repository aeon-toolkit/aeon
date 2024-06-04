"""Tests for the AEDRNN Model."""

import random

import pytest

from aeon.networks import AEDRNNNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "latent_space_dim,n_layers_encoder,n_layers_decoder,temporal_latent_space",
    [
        (32, 1, 1, True),
        (128, 2, 2, False),
        (256, 3, 1, True),
        (64, 4, 2, False),
    ],
)
def test_aedrnnnetwork_init(
    latent_space_dim,
    n_layers_encoder,
    n_layers_decoder,
    temporal_latent_space,
):
    """Test whether AEDRNNNetwork initializes correctly for various parameters."""
    aedrnn = AEDRNNNetwork(
        latent_space_dim=latent_space_dim,
        n_layers_encoder=n_layers_encoder,
        n_layers_decoder=n_layers_decoder,
        temporal_latent_space=temporal_latent_space,
        activation_encoder=random.choice(["relu", "tanh"]),
        activation_decoder=random.choice(["relu", "tanh"]),
        n_units=[random.choice([50, 25, 100]) for _ in range(n_layers_encoder)],
    )
    encoder, decoder = aedrnn.build_network((1000, 5))
    assert encoder is not None
    assert decoder is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "activation_encoder,activation_decoder",
    [("relu", "relu"), ("tanh", "relu"), ("relu", "tanh"), ("tanh", "tanh")],
)
def test_aedrnnnetwork_activations(activation_encoder, activation_decoder):
    """Test whether AEDRNNNetwork initializes correctly with different activations."""
    aedrnn = AEDRNNNetwork(
        latent_space_dim=64,
        n_layers_encoder=2,
        n_layers_decoder=2,
        temporal_latent_space=True,
        activation_encoder=activation_encoder,
        activation_decoder=activation_decoder,
        n_units=[50, 50],
    )
    encoder, decoder = aedrnn.build_network((1000, 5))
    assert encoder is not None
    assert decoder is not None
