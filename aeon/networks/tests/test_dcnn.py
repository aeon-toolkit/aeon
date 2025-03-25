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
    "latent_space_dim,n_layers",
    [
        (32, 1),
        (128, 2),
        (256, 3),
        (64, 4),
    ],
)
def test_dcnnnetwork_init(latent_space_dim, n_layers):
    """Test whether DCNNNetwork initializes correctly for various parameters."""
    dcnnnet = DCNNNetwork(
        latent_space_dim=latent_space_dim,
        n_layers=n_layers,
        activation=random.choice(["relu", "tanh"]),
        n_filters=[random.choice([50, 25, 100]) for _ in range(n_layers)],
    )
    model = dcnnnet.build_network((1000, 5))
    assert isinstance(model, object)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_dcnnnetwork_activations(activation):
    """Test whether DCNNNetwork initializes correctly with different activations."""
    dcnnnet = DCNNNetwork(
        latent_space_dim=64,
        n_layers=2,
        activation=activation,
        n_filters=[50, 50],
    )
    model = dcnnnet.build_network((150, 5))
    assert isinstance(model, object)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "dilation_rate,kernel_size,activation,padding,n_filters",
    [
        (None, None, None, None, None),
        (1, 5, "relu", "causal", 50),
        ([1, 1], [3, 5], ["relu", "tanh"], ["causal", "same"], [25, 50]),
    ],
)
def test_dcnnnetwork_params(dilation_rate, kernel_size, activation, padding, n_filters):
    """Test DCNNNetwork initialization with different parameters."""
    dcnnnet = DCNNNetwork(
        latent_space_dim=64,
        n_layers=2,
        dilation_rate=dilation_rate,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
        n_filters=n_filters,
    )
    model = dcnnnet.build_network((150, 5))
    assert isinstance(model, object)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "dilation_rate,kernel_size,activation,padding,n_filters",
    [
        (None, None, None, None, None),
        (1, 3, "relu", "causal", 50),
        (
            [1, 1, 1],
            [3, 5, 3],
            ["relu", "tanh", "relu"],
            ["causal", "same", "causal"],
            [25, 50, 100],
        ),
    ],
)
def test_dcnnnetwork_varied_layers(
    dilation_rate, kernel_size, activation, padding, n_filters
):
    """Test DCNNNetwork with varied layers and parameters."""
    dcnnnet = DCNNNetwork(
        latent_space_dim=128,
        n_layers=3,
        dilation_rate=dilation_rate,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
        n_filters=n_filters,
    )
    model = dcnnnet.build_network((200, 10))
    assert isinstance(model, object)
