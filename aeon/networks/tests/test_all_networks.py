"""Tests for all networks."""

import inspect

from aeon import networks
from aeon.utils.validation._dependencies import (
    _check_python_version,
    _check_soft_dependencies,
)

__maintainer__ = []


def test_all_networks_functionality():
    """Test the functionality of all networks."""
    network_classes = [
        member[1] for member in inspect.getmembers(networks, inspect.isclass)
    ]
    input_shape = (100, 2)

    for i in range(len(network_classes)):

        if "BaseDeepLearningNetwork" in str(network_classes[i]):
            continue

        if _check_soft_dependencies(
            ["tensorflow", "tensorflow-addons"], severity="none"
        ) and _check_python_version(network_classes[i], severity="none"):
            my_network = network_classes[i]()
        else:
            continue

        if network_classes[i]._tags["auto_encoder"]:
            encoder, decoder = my_network.build_network(input_shape=input_shape)
            assert encoder.layers[-1].output_shape[1:] == (my_network.latent_space_dim,)
            assert encoder.layers[0].input_shape[0] == decoder.layers[-1].output_shape
        else:
            input_layer, output_layer = my_network.build_network(
                input_shape=input_shape
            )
            assert input_layer is not None
            assert output_layer is not None
            if _check_soft_dependencies(["tensorflow"], severity="none"):
                from tensorflow.keras import layers

                assert isinstance(input_layer, layers.Layer)
                assert isinstance(output_layer, layers.Layer)
