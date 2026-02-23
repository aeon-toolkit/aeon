"""Test TimeMCL Self-supervised transformer."""

import tempfile

import numpy as np
import pytest

from aeon.networks import LITENetwork
from aeon.networks.tests.test_network_base import DummyDeepNetwork
from aeon.transformations.collection.self_supervised import TimeMCL
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("latent_space_dim", [2, 3])
def test_time_mcl_latent_space_dim(latent_space_dim):
    """Test TimeMCL with possible latent_space_dim setups."""
    X = np.random.random((100, 2, 5))
    with tempfile.TemporaryDirectory() as tmp:

        ssl = TimeMCL(
            latent_space_dim=latent_space_dim,
            backbone_network=DummyDeepNetwork(),
            n_epochs=3,
            file_path=tmp,
        )

        ssl.fit(X=X)

        X_transformed = ssl.transform(X=X)

        assert len(X_transformed.shape) == 2
        assert int(X_transformed.shape[-1]) == latent_space_dim


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("latent_space_activation", ["linear", "relu"])
def test_time_mcl_latent_space_activation(latent_space_activation):
    """Test file_path with possible latent_space_activation setups."""
    X = np.random.random((100, 2, 5))
    with tempfile.TemporaryDirectory() as tmp:

        ssl = TimeMCL(
            latent_space_activation=latent_space_activation,
            backbone_network=DummyDeepNetwork(),
            latent_space_dim=2,
            n_epochs=3,
            file_path=tmp,
        )

        ssl.fit(X=X)

        X_transformed = ssl.transform(X=X)

        assert len(X_transformed.shape) == 2
        assert int(X_transformed.shape[-1]) == 2


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("backbone_network", [None, DummyDeepNetwork, LITENetwork])
def test_time_mcl_backbone_network(backbone_network):
    """Test TimeMCL with possible backbone_network setups."""
    X = np.random.random((100, 2, 5))
    with tempfile.TemporaryDirectory() as tmp:

        if backbone_network is not None:
            ssl = TimeMCL(
                backbone_network=backbone_network(),
                latent_space_dim=2,
                n_epochs=3,
                file_path=tmp,
            )
        else:
            ssl = TimeMCL(
                backbone_network=backbone_network,
                latent_space_dim=2,
                n_epochs=3,
                file_path=tmp,
            )

        ssl.fit(X=X)

        X_transformed = ssl.transform(X=X)

        assert len(X_transformed.shape) == 2
        assert int(X_transformed.shape[-1]) == 2
