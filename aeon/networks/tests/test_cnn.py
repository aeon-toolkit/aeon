"""Tests for the CNN Model."""

import pytest

from aeon.networks import TimeCNNNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies

__maintainer__ = []


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_cnn_input_shape_padding():
    """Test of CNN network with input_shape < 60."""
    input_shape = (40, 2)
    network = TimeCNNNetwork()
    input_layer, output_layer = network.build_network(input_shape=input_shape)

    assert input_layer is not None
    assert output_layer is not None
