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
    assert model.kernel_size_encoder == 3
    assert model.kernel_size_decoder is None
    assert model.n_filters_decoder is None
    assert model.n_filters_decoder is None
    assert model.n_layers_encoder == 4
    assert model.n_layers_decoder == 4
    assert model.dilation_rate_encoder is None
    assert model.dilation_rate_decoder is None
    assert model.activation_encoder == "relu"
    assert model.activation_decoder == "relu"
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
        n_layers_encoder=3,
        n_layers_decoder=5,
        kernel_size_encoder=5,
        kernel_size_decoder=7,
        activation_encoder="sigmoid",
        activation_decoder="tanh",
        n_filters_decoder=[32, 64, 128],
        n_filters_decoder=[128, 64, 32, 16, 8],
        dilation_rate_encoder=[1, 2, 4],
        dilation_rate_decoder=[4, 2, 1, 2, 4],
    )
    model.build_network((100, 5))
    assert model.latent_space_dim == 64
    assert model._kernel_size_encoder == [5 for _ in range(model.n_layers_encoder)]
    assert model._kernel_size_decoder == [7 for _ in range(model.n_layers_decoder)]
    assert model.n_filters_decoder == [32, 64, 128]
    assert model.n_filters_decoder == [128, 64, 32, 16, 8]
    assert model.n_layers_encoder == 3
    assert model.n_layers_decoder == 5
    assert model.dilation_rate_encoder == [1, 2, 4]
    assert model.dilation_rate_decoder == [4, 2, 1, 2, 4]
    assert model.activation_encoder == "sigmoid"
    assert model.activation_decoder == "tanh"
    assert model.temporal_latent_space


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_edge_case_initialization():
    """Tests edge cases are correct or not."""
    model = AEDCNNNetwork(
        latent_space_dim=0,
        n_layers_encoder=0,
        n_layers_decoder=0,
        kernel_size_encoder=0,
        kernel_size_decoder=0,
        n_filters_decoder=[],
        n_filters_decoder=[],
        dilation_rate_encoder=[],
        dilation_rate_decoder=[],
    )
    assert model.latent_space_dim == 0
    assert model.kernel_size_encoder == 0
    assert model.kernel_size_decoder == 0
    assert model.n_filters_decoder == []
    assert model.n_filters_decoder == []
    assert model.n_layers_encoder == 0
    assert model.n_layers_decoder == 0
    assert model.dilation_rate_encoder == []
    assert model.dilation_rate_decoder == []


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_invalid_initialization():
    """Test if the network raises valid exceptions or not."""
    with pytest.raises(AssertionError):
        AEDCNNNetwork(n_filters_decoder=[32, 64], n_layers_encoder=3).build_network(
            (100, 10)
        )

    with pytest.raises(AssertionError):
        AEDCNNNetwork(dilation_rate_encoder=[1, 2], n_layers_encoder=3).build_network(
            (100, 10)
        )


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
