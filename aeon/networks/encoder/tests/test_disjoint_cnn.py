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
