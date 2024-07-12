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
    "latent_space_dim,num_layers",
    [
        (32, 1),
        (128, 2),
        (256, 3),
        (64, 4),
    ],
)
def test_dcnnnetwork_init(latent_space_dim, num_layers):
    """Test whether DCNNNetwork initializes correctly for various parameters."""
    dcnnnet = DCNNNetwork(
        latent_space_dim=latent_space_dim,
        num_layers=num_layers,
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
        activation=activation,
        num_filters=[50, 50],
    )
    model = dcnnnet.build_network((150, 5))
    assert model is not None
