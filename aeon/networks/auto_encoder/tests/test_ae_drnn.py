"""Tests for the AEDRNN Model."""

import random

import pytest

from aeon.networks import AEDRNNNetwork
from aeon.typing import LATENT_SPACE
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "latent_space_dim,n_layers_encoder,n_layers_decoder,latent_space_type",
    [
        (32, 1, 1, LATENT_SPACE.TIME),
        (128, 2, 2, LATENT_SPACE.FLAT),
        (256, 3, 1, LATENT_SPACE.TIME),
        (64, 4, 2, LATENT_SPACE.FLAT),
    ],
)
def test_aedrnnnetwork_init(
    latent_space_dim,
    n_layers_encoder,
    n_layers_decoder,
    latent_space_type,
):
    """Test whether AEDRNNNetwork initializes correctly for various parameters."""
    aedrnn = AEDRNNNetwork(
        latent_space_dim=latent_space_dim,
        n_layers_encoder=n_layers_encoder,
        n_layers_decoder=n_layers_decoder,
        latent_space_type=latent_space_type,
        activation_encoder=random.choice(["relu", "tanh"]),
        activation_decoder=random.choice(["relu", "tanh"]),
        n_units_encoder=[random.choice([50, 25, 100]) for _ in range(n_layers_encoder)],
    )
    aedrnn.build_network((150, 5))
    encoder, decoder = aedrnn.get_encoder_decoder_models()
    assert encoder is not None
    assert decoder is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "activation_encoder,activation_decoder",
    [("relu", "relu"), (["relu", "tanh"], ["tanh", "relu"])],
)
def test_aedrnnnetwork_activations(activation_encoder, activation_decoder):
    """Test whether AEDRNNNetwork initializes correctly with different activations."""
    aedrnn = AEDRNNNetwork(
        latent_space_dim=64,
        n_layers_encoder=2,
        n_layers_decoder=2,
        latent_space_type=LATENT_SPACE.TIME,
        activation_encoder=activation_encoder,
        activation_decoder=activation_decoder,
        n_units_encoder=[50, 50],
    )
    aedrnn.build_network((150, 5))
    encoder, decoder = aedrnn.get_encoder_decoder_models()
    assert encoder is not None
    assert decoder is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "dilation_rate_encoder,dilation_rate_decoder",
    [(None, None), ([2**i for i in range(1, 4)], [2**i for i in range(1, 4)]), (4, 4)],
)
def test_aedrnnnetwork_dilation_rate(dilation_rate_encoder, dilation_rate_decoder):
    """Test whether AEDRNNNetwork initializes properly with different dilation-rates."""
    aedrnn = AEDRNNNetwork(
        latent_space_dim=64,
        n_layers_decoder=3,
        dilation_rate_encoder=dilation_rate_encoder,
        dilation_rate_decoder=dilation_rate_decoder,
    )
    encoder, decoder = aedrnn.build_network((150, 5))
    assert encoder is not None
    assert decoder is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_aedrnnnetwork_exceptions():
    """Test whether AEDRNNNetwork raises exceptions correctly."""
    aedrnn = AEDRNNNetwork(
        latent_space_dim=64,
        n_layers_decoder=3,
        n_layers_encoder=5,
        dilation_rate_encoder=[1, 2],
    )

    with pytest.raises(ValueError):
        aedrnn.build_network((150, 5))

    aedrnn = AEDRNNNetwork(
        latent_space_dim=64,
        n_layers_decoder=3,
        n_layers_encoder=5,
        dilation_rate_decoder=[1, 2],
    )

    with pytest.raises(ValueError):
        aedrnn.build_network((150, 5))
