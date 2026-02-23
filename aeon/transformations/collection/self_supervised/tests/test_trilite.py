"""Test TRILITE Self-supervised transformer."""

import tempfile

import numpy as np
import pytest

from aeon.networks import LITENetwork
from aeon.networks.tests.test_network_base import DummyDeepNetwork
from aeon.transformations.collection.self_supervised import TRILITE
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("use_mixing_up", [True, False])
def test_trilite_use_mixing_up(use_mixing_up):
    """Test TRILITE with possible mixing up setups."""
    X = np.random.random((100, 2, 5))
    with tempfile.TemporaryDirectory() as tmp:

        ssl = TRILITE(
            use_mixing_up=use_mixing_up,
            latent_space_dim=2,
            backbone_network=DummyDeepNetwork(),
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
@pytest.mark.parametrize("use_masking", [True, False])
def test_trilite_use_masking(use_masking):
    """Test TRILITE with possible masking setups."""
    X = np.random.random((100, 2, 5))
    with tempfile.TemporaryDirectory() as tmp:

        ssl = TRILITE(
            use_masking=use_masking,
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
@pytest.mark.parametrize("z_normalize_pos_neg", [True, False])
def test_trilite_z_normalize_pos_neg(z_normalize_pos_neg):
    """Test TRILITE with possible znorm pos and neg setups."""
    X = np.random.random((100, 2, 5))
    with tempfile.TemporaryDirectory() as tmp:

        ssl = TRILITE(
            z_normalize_pos_neg=z_normalize_pos_neg,
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
@pytest.mark.parametrize("alpha", [1e-1, 1e-2])
def test_trilite_alpha(alpha):
    """Test TRILITE with possible alpha setups."""
    X = np.random.random((100, 2, 5))
    with tempfile.TemporaryDirectory() as tmp:

        ssl = TRILITE(
            alpha=alpha,
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
@pytest.mark.parametrize("weight_ref_min", [0.5, 0.6])
def test_trilite_weight_ref_min(weight_ref_min):
    """Test TRILITE with possible weight_ref_min setups."""
    X = np.random.random((100, 2, 5))
    with tempfile.TemporaryDirectory() as tmp:

        ssl = TRILITE(
            weight_ref_min=weight_ref_min,
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
@pytest.mark.parametrize("percentage_mask_length", [0.2, 0.3])
def test_trilite_percentage_mask_length(percentage_mask_length):
    """Test TRILITE with possible percentage_mask_length setups."""
    X = np.random.random((100, 2, 5))
    with tempfile.TemporaryDirectory() as tmp:

        ssl = TRILITE(
            percentage_mask_length=percentage_mask_length,
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
@pytest.mark.parametrize("latent_space_dim", [2, 3])
def test_trilite_latent_space_dim(latent_space_dim):
    """Test TRILITE with possible latent_space_dim setups."""
    X = np.random.random((100, 2, 5))
    with tempfile.TemporaryDirectory() as tmp:

        ssl = TRILITE(
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
def test_trilite_latent_space_activation(latent_space_activation):
    """Test TRILITE with possible latent_space_activation setups."""
    X = np.random.random((100, 2, 5))
    with tempfile.TemporaryDirectory() as tmp:

        ssl = TRILITE(
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
def test_trilite_backbone_network(backbone_network):
    """Test TRILITE with possible backbone_network setups."""
    X = np.random.random((100, 2, 5))
    with tempfile.TemporaryDirectory() as tmp:

        if backbone_network is not None:
            ssl = TRILITE(
                backbone_network=backbone_network(),
                latent_space_dim=2,
                n_epochs=3,
                file_path=tmp,
            )
        else:
            ssl = TRILITE(
                backbone_network=backbone_network,
                latent_space_dim=2,
                n_epochs=3,
                file_path=tmp,
            )

        ssl.fit(X=X)

        X_transformed = ssl.transform(X=X)

        assert len(X_transformed.shape) == 2
        assert int(X_transformed.shape[-1]) == 2
