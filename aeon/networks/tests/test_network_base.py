import pytest

from aeon.networks.base import BaseDeepNetwork
from aeon.testing.utils.data_gen import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies

__maintainer__ = []


class DummyDeepNetwork(BaseDeepNetwork):
    """A Dummy Deep Network for testing empty base network class save utilities."""

    def __init__(self):
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

    X, y = make_example_3d_numpy()

    input_layer, output_layer = dummy_network.build_network(input_shape=X.shape)

    assert input_layer is not None
    assert output_layer is not None
    assert output_layer.shape[-1] == 10
