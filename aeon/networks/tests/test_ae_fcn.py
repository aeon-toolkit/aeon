"""Tests for the AEFCN Model."""

import pytest

from aeon.networks import AEFCNNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_default_initialization():
    """Test if the network initializes with proper attributes."""
    model = AEFCNNetwork()
    assert model.latent_space_dim == 128
    assert model.n_layers == 3
    assert model.n_filters is None
    assert model.kernel_size is None
    assert model.dilation_rate == 1
    assert model.strides == 1
    assert model.padding == "same"
    assert model.activation == "relu"
    assert model.use_bias is True
    assert model.temporal_latent_space is False


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_custom_initialization():
    """Test whether custom kwargs are correctly set."""
    model = AEFCNNetwork(
        latent_space_dim=64,
        temporal_latent_space=True,
        n_layers=4,
        n_filters=[32, 64, 128, 256],
        kernel_size=[9, 7, 5, 3],
        activation="sigmoid",
        dilation_rate=[1, 2, 4, 8],
    )
    model.build_network((100, 5))
    assert model.latent_space_dim == 64
    assert model.n_layers == 4
    assert model._n_filters == [32, 64, 128, 256]
    assert model._kernel_size == [9, 7, 5, 3]
    assert model.dilation_rate == [1, 2, 4, 8]
    assert model.activation == "sigmoid"
    assert model.temporal_latent_space is True


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_edge_case_initialization():
    """Tests edge cases with minimal values."""
    model = AEFCNNetwork(
        latent_space_dim=0, n_layers=0, kernel_size=0, dilation_rate=[]
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
    with pytest.raises(ValueError):
        AEFCNNetwork(n_filters=[32, 64], n_layers=3).build_network((100, 10))

    with pytest.raises(ValueError):
        AEFCNNetwork(dilation_rate=[1, 2], n_layers=3).build_network((100, 10))


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_build_network():
    """Test call to the build_network method."""
    import tensorflow as tf

    model = AEFCNNetwork()
    input_shape = (100, 10)
    encoder, decoder = model.build_network(input_shape)

    assert isinstance(encoder, tf.keras.Model)
    assert isinstance(decoder, tf.keras.Model)
    assert encoder.input_shape == (None, 100, 10)
    assert decoder.input_shape is not None
