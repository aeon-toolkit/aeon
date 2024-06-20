"""Tests for the AEDCNN Model."""

import pytest

from aeon.networks import AEDCNNNetwork


def test_default_initialization():
    """Test if the network initializes with proper attributes."""
    model = AEDCNNNetwork()
    assert model.latent_space_dim == 128
    assert model.kernel_size_encoder == 3
    assert model.kernel_size_decoder is None
    assert model.num_filters_encoder is None
    assert model.num_filters_decoder is None
    assert model.num_layers_encoder == 4
    assert model.num_layers_decoder == 4
    assert model.dilation_rate_encoder is None
    assert model.dilation_rate_decoder is None
    assert model.activation_encoder == "relu"
    assert model.activation_decoder == "relu"
    assert not model.temporal_latent_space


def test_custom_initialization():
    """Test whether custom kwargs are correctly set."""
    model = AEDCNNNetwork(
        latent_space_dim=64,
        temporal_latent_space=True,
        num_layers_encoder=3,
        num_layers_decoder=5,
        kernel_size_encoder=5,
        kernel_size_decoder=7,
        activation_encoder="sigmoid",
        activation_decoder="tanh",
        num_filters_encoder=[32, 64, 128],
        num_filters_decoder=[128, 64, 32, 16, 8],
        dilation_rate_encoder=[1, 2, 4],
        dilation_rate_decoder=[4, 2, 1, 2, 4],
    )
    model.build_network((100, 5))
    assert model.latent_space_dim == 64
    assert model._kernel_size_encoder == [5 for _ in range(model.num_layers_encoder)]
    assert model._kernel_size_decoder == [7 for _ in range(model.num_layers_decoder)]
    assert model.num_filters_encoder == [32, 64, 128]
    assert model.num_filters_decoder == [128, 64, 32, 16, 8]
    assert model.num_layers_encoder == 3
    assert model.num_layers_decoder == 5
    assert model.dilation_rate_encoder == [1, 2, 4]
    assert model.dilation_rate_decoder == [4, 2, 1, 2, 4]
    assert model.activation_encoder == "sigmoid"
    assert model.activation_decoder == "tanh"
    assert model.temporal_latent_space


def test_edge_case_initialization():
    """Tests edge cases are correct or not."""
    model = AEDCNNNetwork(
        latent_space_dim=0,
        num_layers_encoder=0,
        num_layers_decoder=0,
        kernel_size_encoder=0,
        kernel_size_decoder=0,
        num_filters_encoder=[],
        num_filters_decoder=[],
        dilation_rate_encoder=[],
        dilation_rate_decoder=[],
    )
    assert model.latent_space_dim == 0
    assert model.kernel_size_encoder == 0
    assert model.kernel_size_decoder == 0
    assert model.num_filters_encoder == []
    assert model.num_filters_decoder == []
    assert model.num_layers_encoder == 0
    assert model.num_layers_decoder == 0
    assert model.dilation_rate_encoder == []
    assert model.dilation_rate_decoder == []


def test_invalid_initialization():
    """Test if the network raises valid exceptions or not."""
    with pytest.raises(AssertionError):
        AEDCNNNetwork(num_filters_encoder=[32, 64], num_layers_encoder=3).build_network(
            (100, 10)
        )

    with pytest.raises(AssertionError):
        AEDCNNNetwork(dilation_rate_encoder=[1, 2], num_layers_encoder=3).build_network(
            (100, 10)
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
