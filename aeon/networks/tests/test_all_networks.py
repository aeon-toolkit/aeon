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

        assert "python_dependencies" in network_classes[i]._config.keys()
        assert "python_version" in network_classes[i]._config.keys()
        assert "auto-encoder" in network_classes[i]._config.keys()
        assert isinstance(network_classes[i]._config["python_dependencies"], str) and (
            "tensorflow" in network_classes[i]._config["python_dependencies"]
        )
        assert isinstance(network_classes[i]._config["python_version"], str)
        assert isinstance(network_classes[i]._config["auto-encoder"], bool)

        if "BaseDeepLearningNetwork" in str(network_classes[i]):
            continue

        if _check_soft_dependencies(
            network_classes[i]._config["python_dependencies"], severity="none"
        ) and _check_python_version(
            network_classes[i]._config["python_version"], severity="none"
        ):
            if "EncoderNetwork" in str(network_classes[i]):
                if _check_soft_dependencies(["tensorflow-addons"], severity="none"):
                    my_network = network_classes[i]()
            else:
                my_network = network_classes[i]()
        else:
            continue

        if network_classes[i]._config["auto_encoder"]:
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
