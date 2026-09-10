"""Tests for the DisjointCNN Network."""

import pytest

from aeon.networks import DisjointCNNNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_disjoint_cnn_netowkr_kernel_initializer():
    """Test DisjointCNN for different kernel_initializer per layer."""
    input_layer, output_layer = DisjointCNNNetwork(
        n_layers=2,
        kernel_initializer=["he_uniform", "glorot_uniform"],
        kernel_size=[2, 2],
    ).build_network(input_shape=((10, 2)))

    assert len(output_layer.shape) == 2
    assert len(input_layer.shape) == 3


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_disjoint_cnn_final_projection_receives_final_filter_count():
    """Final block is not permuted before pooling."""
    import tensorflow as tf

    network = DisjointCNNNetwork(n_layers=2, n_filters=[4, 8], kernel_size=[3, 3])
    inputs, outputs = network.build_network((12, 3))
    model = tf.keras.Model(inputs, outputs)
    assert model.layers[-1].input.shape[-1] == 8
