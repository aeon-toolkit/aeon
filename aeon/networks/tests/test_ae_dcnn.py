"""Tests for the AEDCNN Model."""

import pytest

from aeon.networks import AEDCNNNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_default_initialization():
    """Test if the network initializes with proper attributes."""
    model = AEDCNNNetwork()
    assert model.latent_space_dim == 128
    assert model.kernel_size == 3
    assert model.n_layers == 4
    assert model.dilation_rate == 1
    assert model.activation == "relu"
    assert not model.temporal_latent_space


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_custom_initialization():
    """Test whether custom kwargs are correctly set."""
    model = AEDCNNNetwork(
        latent_space_dim=64,
        temporal_latent_space=True,
        n_layers=3,
        kernel_size=5,
        activation="sigmoid",
        dilation_rate=[1, 2, 4],
    )
    model.build_network((100, 5))
    assert model.latent_space_dim == 64
    assert model._kernel_size_encoder == [5 for _ in range(model.n_layers)]
    assert model.n_layers == 3
    assert model.dilation_rate == [1, 2, 4]
    assert model.activation == "sigmoid"
    assert model.temporal_latent_space


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_edge_case_initialization():
    """Tests edge cases are correct or not."""
    model = AEDCNNNetwork(
        latent_space_dim=0,
        n_layers=0,
        kernel_size=0,
        dilation_rate=[],
    )
    assert model.latent_space_dim == 0
    assert model.kernel_size == 0
    assert model.n_layers == 0
    assert model.dilation_rate == []


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_invalid_initialization():
    """Test if the network raises valid exceptions or not."""
    with pytest.raises(AssertionError):
        AEDCNNNetwork(n_filters=[32, 64], n_layers=3).build_network((100, 10))

    with pytest.raises(AssertionError):
        AEDCNNNetwork(dilation_rate=[1, 2], n_layers=3).build_network((100, 10))


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_build_network():
    """Test call to the build_network method."""
    model = AEDCNNNetwork()
    input_shape = (100, 10)  # Example input shape
    encoder, decoder = model.build_network(input_shape)
    assert encoder is not None
    assert decoder is not None
    assert encoder.input_shape == (None, 100, 10)
    assert decoder.input_shape is not None
