"""Tests for the Inception Model."""

import pytest

from aeon.networks import InceptionNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies

__maintainer__ = []


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_inceptionnetwork_bottleneck():
    """Test of Inception network without bottleneck."""
    input_shape = (100, 2)
    inception = InceptionNetwork(use_bottleneck=False)
    input_layer, output_layer = inception.build_network(input_shape=input_shape)

    assert input_layer is not None
    assert output_layer is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_inceptionnetwork_max_pooling():
    """Test of Inception network without max pooling."""
    input_shape = (100, 2)
    inception = InceptionNetwork(use_max_pooling=False)
    input_layer, output_layer = inception.build_network(input_shape=input_shape)

    assert input_layer is not None
    assert output_layer is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_inceptionnetwork_custom_filters():
    """Test of Inception network with custom filters."""
    input_shape = (100, 2)
    inception = InceptionNetwork(use_custom_filters=True)
    input_layer, output_layer = inception.build_network(input_shape=input_shape)

    assert input_layer is not None
    assert output_layer is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_inceptionnetwork_list_parameters():
    """Test of Inception network with list parameters not in test_all_networks."""
    input_shape = (100, 2)
    depth = 6
    n_conv_per_layer = [3] * depth
    use_max_pooling = [True] * depth
    max_pool_size = [3] * depth

    inception = InceptionNetwork(
        depth=depth,
        n_conv_per_layer=n_conv_per_layer,
        use_max_pooling=use_max_pooling,
        max_pool_size=max_pool_size,
    )
    input_layer, output_layer = inception.build_network(input_shape=input_shape)

    assert input_layer is not None
    assert output_layer is not None
