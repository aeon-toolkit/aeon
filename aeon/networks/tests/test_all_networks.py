"""Tests for all networks."""

import inspect

import pytest

from aeon import networks
from aeon.utils.validation._dependencies import (
    _check_python_version,
    _check_soft_dependencies,
)

__maintainer__ = []

_networks = network_classes = [
    member[1] for member in inspect.getmembers(networks, inspect.isclass)
]


@pytest.mark.parametrize("network", _networks)
def test_network_config(network):
    """Tests if the config dictionary of classes is correctly configured."""
    assert "python_dependencies" in network._config.keys()
    assert "python_version" in network._config.keys()
    assert "auto-encoder" in network._config.keys()
    assert isinstance(network._config["python_dependencies"], str) and (
        "tensorflow" in network._config["python_dependencies"]
    )
    assert isinstance(network._config["python_version"], str)
    assert isinstance(network._config["auto-encoder"], bool)


@pytest.mark.parametrize("network", _networks)
def test_all_networks_functionality(network):
    """Test the functionality of all networks."""
    input_shape = (100, 2)

    if "BaseDeepLearningNetwork" != str(network):
        if _check_soft_dependencies(
            network._config["python_dependencies"], severity="none"
        ) and _check_python_version(network._config["python_version"], severity="none"):
            if "EncoderNetwork" == str(network):
                if _check_soft_dependencies(["tensorflow-addons"], severity="none"):
                    my_network = network()
            else:
                my_network = network()

        if network._config["auto_encoder"]:
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
                import tensorflow as tf

                assert isinstance(input_layer, tf.keras.layers.Layer)
                assert isinstance(output_layer, tf.keras.layers.Layer)
