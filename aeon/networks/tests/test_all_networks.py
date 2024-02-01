import inspect
import os
import time

import pytest

from aeon import networks
from aeon.utils.validation._dependencies import _check_soft_dependencies

__author__ = ["hadifawaz1999"]


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_all_networks_functionality():
    network_classes = [
        member[1] for member in inspect.getmembers(networks, inspect.isclass)
    ]
    input_shape = (100, 2)

    for i in range(len(network_classes)):
        if "BaseDeepNetwork" in str(network_classes[i]) or "AEFCNNetwork" in str(
            network_classes[i]
        ):
            continue

        my_network = network_classes[i]()

        input_layer, output_layer = my_network.build_network(input_shape=input_shape)

        file_name = str(time.time_ns())
        my_network.plot_network(input_shape=input_shape, file_name=file_name)
        if os.path.exists(file_name + ".pdf"):
            os.remove(file_name + ".pdf")

        assert input_layer is not None
        assert output_layer is not None
