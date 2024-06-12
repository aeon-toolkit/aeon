"""Tests for all networks."""

import inspect

from aeon import networks

__maintainer__ = []


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow", "tensorflow_addons"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_all_networks_functionality():
    """Test the functionality of all networks."""
    network_classes = [
        member[1] for member in inspect.getmembers(networks, inspect.isclass)
    ]
    input_shape = (100, 2)

    for i in range(len(network_classes)):
        if (
            "BaseDeepNetwork" in str(network_classes[i])
            or "BaseDeepLearningNetwork" in str(network_classes[i])
            or "AEFCNNetwork" in str(network_classes[i])
            or "EncoderNetwork" in str(network_classes[i])
        ):
            continue

        try:
            my_network = network_classes[i]()
        except ModuleNotFoundError:
            continue

        input_layer, output_layer = my_network.build_network(input_shape=input_shape)

        assert input_layer is not None
        assert output_layer is not None
