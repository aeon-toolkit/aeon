"""Tests for the AEBiGRU Model."""

import random

import pytest

from aeon.networks import AERecurrentNetwork
from aeon.typing import LATENT_SPACE
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "latent_space_dim,n_layers,latent_space_type",
    [
        (32, 1, LATENT_SPACE.TIME),
        (128, 2, LATENT_SPACE.FLAT),
        (256, 3, LATENT_SPACE.TIME),
        (64, 4, LATENT_SPACE.REPEATED),
    ],
)
def test_ae_recurrent_network_init(latent_space_dim, n_layers, latent_space_type):
    """Test whether AERecurrentNetwork initializes correctly for various parameters."""
    aebigru = AERecurrentNetwork(
        latent_space_dim=latent_space_dim,
        n_layers=n_layers,
        latent_space_type=latent_space_type,
        activation=random.choice(["relu", "tanh"]),
        n_units=[random.choice([50, 25, 100]) for _ in range(n_layers)],
    )
    encoder, decoder = aebigru.build_network((1000, 5))
    assert encoder is not None
    assert decoder is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_ae_recurrent_network_activations(activation):
    """Test AERecurrentNetwork initialization with different activations."""
    aebigru = AERecurrentNetwork(
        latent_space_dim=64,
        n_layers=2,
        latent_space_type=LATENT_SPACE.TIME,
        activation=activation,
        n_units=[50, 50],
    )
    encoder, decoder = aebigru.build_network((1000, 5))
    assert encoder is not None
    assert decoder is not None
