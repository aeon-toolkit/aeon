"""Tests for all networks."""

import inspect

import pytest

from aeon import networks
from aeon.networks import BaseDeepLearningNetwork
from aeon.utils.validation._dependencies import (
    _check_python_version,
    _check_soft_dependencies,
)

network_classes = [
    member[1] for member in inspect.getmembers(networks, inspect.isclass)
]
network_classes.remove(BaseDeepLearningNetwork)


def test_network_classes():
    """Test that all network classes are correctly defined."""
    for cls in network_classes:
        assert issubclass(cls, BaseDeepLearningNetwork)


def _check_network(network):
    """Check if we can run the tests for this network."""
    if not _check_soft_dependencies(
        network._config["python_dependencies"], severity="none"
    ):
        return False, f"{network.__name__} dependencies not satisfied."
    if not _check_python_version(network._config["python_version"], severity="none"):
        return False, f"Invalid Python version for {network.__name__}."
    return True, None


@pytest.mark.parametrize("network", network_classes)
def test_network_config(network):
    """Tests if the config dictionary of classes is correctly configured."""
    assert hasattr(network, "_config")
    assert isinstance(network._config, dict)

    assert "python_dependencies" in network._config.keys()
    assert "python_version" in network._config.keys()
    assert "structure" in network._config.keys()

    assert isinstance(network._config["python_dependencies"], (str, list))
    assert isinstance(network._config["python_version"], str)
    assert isinstance(network._config["structure"], str)
    assert network._config["structure"] in [
        "auto-encoder",
        "encoder",
        "encoder-decoder",
    ]


@pytest.mark.parametrize("network", network_classes)
def test_all_networks_functionality(network):
    """Test the functionality of all networks."""
    skip, reason = _check_network(network)
    if skip:
        pytest.skip(reason)

    input_shape = (100, 2)
    my_network = network()

    if network._config["structure"] == "auto-encoder":
        encoder, decoder = my_network.build_network(input_shape=input_shape)
        assert encoder.output_shape[1:] == (my_network.latent_space_dim,)
        assert encoder.input_shape == decoder.output_shape
    elif network._config["structure"] == "encoder":
        import tensorflow as tf

        input_layer, output_layer = my_network.build_network(input_shape=input_shape)
        assert input_layer is not None
        assert output_layer is not None
        assert tf.keras.backend.is_keras_tensor(input_layer)
        assert tf.keras.backend.is_keras_tensor(output_layer)
    else:
        # todo this does not cover deepar
        pytest.skip(
            f"{network.__name__} not to be tested (uncovered structure type: {network._config['structure']})."
        )


@pytest.mark.parametrize("network", network_classes)
def test_all_networks_params(network):
    """Test the functionality of all networks."""
    skip, reason = _check_network(network)
    if skip:
        pytest.skip(reason)

    # todo: figure out these issues and re-enable testing if possible
    # also applies to the ["kernel_size", "padding"] skip below
    if network._config["structure"] == "auto-encoder":
        pytest.skip(
            f"{network.__name__} not to be tested (AE networks have their own tests)."
        )
    # LITENetwork does not seem to work with list args
    if network.__name__ == "LITENetwork" or network.__name__ == "MLPNetwork":
        pytest.skip(f"Skipping {network.__name__} due to unresolved issue.")

    input_shape = (100, 2)

    # check with default parameters
    my_network = network()
    my_network.build_network(input_shape=input_shape)

    # check with list parameters
    params = dict()
    for attrname in [
        "kernel_size",
        "n_filters",
        "avg_pool_size",
        "activation",
        "padding",
        "strides",
        "dilation_rate",
        "use_bias",
    ]:
        # Exceptions to fix
        if attrname in ["kernel_size", "padding"]:
            continue

        # Here we use 'None' string as default to differentiate with None values
        attr = getattr(my_network, attrname, "None")
        if attr != "None":
            if attr is None:
                attr = 3
            elif isinstance(attr, list):
                attr = attr[0]
            else:
                if network.__name__ in ["ResNetNetwork"]:
                    attr = [attr] * my_network.n_conv_per_residual_block
                elif network.__name__ in ["InceptionNetwork"]:
                    attr = [attr] * my_network.depth
                else:
                    attr = [attr] * my_network.n_layers
            params[attrname] = attr

    if params:
        my_network = network(**params)
        my_network.build_network(input_shape=input_shape)
