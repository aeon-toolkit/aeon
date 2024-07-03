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
    assert "structure" in network._config.keys()
    assert isinstance(network._config["python_dependencies"], list) and (
        "tensorflow" in network._config["python_dependencies"]
    )
    assert isinstance(network._config["python_version"], str)
    assert isinstance(network._config["structure"], str)


@pytest.mark.parametrize("network", _networks)
def test_all_networks_functionality(network):
    """Test the functionality of all networks."""
    input_shape = (100, 2)

    if not (
        network.__name__
        in ["BaseDeepNetwork", "BaseDeepLearningNetwork", "EncoderNetwork"]
    ):
        if _check_soft_dependencies(
            network._config["python_dependencies"], severity="none"
        ) and _check_python_version(network._config["python_version"], severity="none"):
            my_network = network()

            if network._config["structure"] == "auto-encoder":
                encoder, decoder = my_network.build_network(input_shape=input_shape)
                assert encoder.output_shape[1:] == (my_network.latent_space_dim,)
                assert encoder.input_shape == decoder.output_shape
            elif network._config["structure"] == "encoder":
                import tensorflow as tf

                input_layer, output_layer = my_network.build_network(
                    input_shape=input_shape
                )
                assert input_layer is not None
                assert output_layer is not None
                assert tf.keras.backend.is_keras_tensor(input_layer)
                assert tf.keras.backend.is_keras_tensor(output_layer)
        else:
            pytest.skip(
                f"{network.__name__} dependencies not satisfied or invalid \
                Python version."
            )
    else:
        pytest.skip(f"{network.__name__} not to be tested since its a base class.")
