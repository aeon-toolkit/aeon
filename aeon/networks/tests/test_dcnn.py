"""Tests for the DCNN Model."""

import random

import pytest

from aeon.networks import DCNNNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "latent_space_dim,num_layers,temporal_latent_space",
    [
        (32, 1, True),
        (128, 2, False),
        (256, 3, True),
        (64, 4, False),
    ],
)
def test_dcnnnetwork_init(latent_space_dim, num_layers, temporal_latent_space):
    """Test whether DCNNNetwork initializes correctly for various parameters."""
    dcnnnet = DCNNNetwork(
        latent_space_dim=latent_space_dim,
        num_layers=num_layers,
        temporal_latent_space=temporal_latent_space,
        activation=random.choice(["relu", "tanh"]),
        num_filters=[random.choice([50, 25, 100]) for _ in range(num_layers)],
    )
    model = dcnnnet.build_network((1000, 5))
    assert model is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_dcnnnetwork_activations(activation):
    """Test whether DCNNNetwork initializes correctly with different activations."""
    dcnnnet = DCNNNetwork(
        latent_space_dim=64,
        num_layers=2,
        temporal_latent_space=True,
        activation=activation,
        num_filters=[50, 50],
    )
    model = dcnnnet.build_network((150, 5))
    assert model is not None
