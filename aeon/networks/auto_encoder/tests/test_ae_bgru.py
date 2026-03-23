"""Tests for the AEBiGRU Model."""

import random

import pytest

from aeon.networks import AEBiGRUNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "latent_space_dim,n_layers,temporal_latent_space",
    [
        (32, 1, True),
        (128, 2, False),
        (256, 3, True),
        (64, 4, False),
    ],
)
def test_aebigrunetwork_init(latent_space_dim, n_layers, temporal_latent_space):
    """Test whether AEBiGRUNetwork initializes correctly for various parameters."""
    aebigru = AEBiGRUNetwork(
        latent_space_dim=latent_space_dim,
        n_layers=n_layers,
        temporal_latent_space=temporal_latent_space,
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
def test_aebigrunetwork_activations(activation):
    """Test whether AEBiGRUNetwork initializes correctly with different activations."""
    aebigru = AEBiGRUNetwork(
        latent_space_dim=64,
        n_layers=2,
        temporal_latent_space=True,
        activation=activation,
        n_units=[50, 50],
    )
    encoder, decoder = aebigru.build_network((1000, 5))
    assert encoder is not None
    assert decoder is not None
