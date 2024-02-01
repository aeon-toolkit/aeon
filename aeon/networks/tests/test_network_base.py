import os
import time

import pytest

from aeon.networks.base import BaseDeepNetwork
from aeon.testing.utils.data_gen import make_3d_test_data
from aeon.utils.validation._dependencies import _check_soft_dependencies

__author__ = ["hadifawaz1999"]


class DummyDeepNetwork(BaseDeepNetwork):
    def __init__(self):
        """Dummy Deep Network for testing empty base network class save utilities."""
        super().__init__()

    def build_network(self, input_shape, **kwargs):
        import tensorflow as tf

        input_layer = tf.keras.layers.Input(input_shape)
        output_layer = tf.keras.layers.Dense(units=10)(input_layer)

        return input_layer, output_layer


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow", "pydot"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_dummy_deep_network():
    dummy_network = DummyDeepNetwork()

    X, y = make_3d_test_data()

    file_name = str(time.time_ns())
    dummy_network.plot_network(input_shape=X.shape, file_name=file_name)
    if os.path.exists(file_name + ".pdf"):
        os.remove(file_name + ".pdf")

    input_layer, output_layer = dummy_network.build_network(input_shape=X.shape)

    assert input_layer is not None
    assert output_layer is not None
    assert output_layer.shape[-1] == 10
