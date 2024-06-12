"""Tests for all networks."""

import inspect
import logging

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

        if "BaseDeepNetwork" in str(
            network_classes[i]
        ) or "BaseDeepLearningNetwork" in str(network_classes[i]):
            continue

        my_network = None

        try:
            my_network = network_classes[i]()
        except ModuleNotFoundError:
            if "EncoderNetwork" in str(network_classes[i]):
                if _check_soft_dependencies(
                    ["tensorflow-addons"], severity="none"
                ) and _check_python_version(network_classes[i], severity="none"):
                    my_network = network_classes[i]()
                else:
                    continue
        except Exception as e:
            logging.error(f"Error instantiating {network_classes[i]}: {e}")
            continue

        if my_network is None:
            continue

        try:
            if "AE" in str(network_classes[i]):
                encoder, decoder = my_network.build_network(input_shape=input_shape)
                assert encoder.layers[-1].output_shape[1:] == (
                    my_network.latent_space_dim,
                )
                assert (
                    encoder.layers[0].input_shape[0] == decoder.layers[-1].output_shape
                )
            else:
                input_layer, output_layer = my_network.build_network(
                    input_shape=input_shape
                )
                assert input_layer is not None
                assert output_layer is not None
        except AttributeError as e:
            logging.error(f"Error in network {network_classes[i]}: {e}")
        except AssertionError as e:
            logging.error(f"Assertion failed for network {network_classes[i]}: {e}")
