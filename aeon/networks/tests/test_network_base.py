"""Base for deep learning network base."""

import pytest

from aeon.networks.base import BaseDeepLearningNetwork
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies

__maintainer__ = []


class DummyDeepNetwork(BaseDeepLearningNetwork):
    """A Dummy Deep Network for testing empty base network class save utilities."""

    def __init__(self):
        super().__init__()

    def _check_params(self):
        pass  # no parameters to check for the dummy network

    def build_base_graph(self, x):
        """Build the base graph for the dummy network."""
        import tensorflow as tf

        self._check_params()

        x = tf.keras.layers.Flatten()(x)
        return x

    def build_network(self, input_shape, **kwargs):
        """Build a neural network."""
        import tensorflow as tf

        input_layer = tf.keras.layers.Input(input_shape)
        flatten_layer = self.build_base_graph(input_layer)
        output_layer = tf.keras.layers.Dense(units=10)(flatten_layer)

        return input_layer, output_layer


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_dummy_deep_network():
    """Test the dummy network."""
    dummy_network = DummyDeepNetwork()

    X, y = make_example_3d_numpy()

    input_layer, output_layer = dummy_network.build_network(input_shape=X.shape)

    assert input_layer is not None
    assert output_layer is not None
    assert output_layer.shape[-1] == 10


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize(
    "blocks, depth, param, expected, expected_exception",
    [
        # depth is an int
        (3, 2, 64, [[64, 64], [64, 64], [64, 64]], None),
        (3, 2, [64, 32], [[64, 32], [64, 32], [64, 32]], None),
        (3, 2, [[64, 32], [64, 32], [64, 32]], [[64, 32], [64, 32], [64, 32]], None),
        # depth is a list of int
        (3, [2, 2, 2], 64, [[64, 64], [64, 64], [64, 64]], None),
        (3, [2, 2, 2], [64, 32], [[64, 32], [64, 32], [64, 32]], None),
        (
            3,
            [1, 2, 3],
            64,
            [
                [
                    64,
                ],
                [64, 64],
                [64, 64, 64],
            ],
            None,
        ),
        (
            3,
            [1, 2, 3],
            [
                [
                    16,
                ],
                [16, 32],
                [16, 32, 64],
            ],
            [
                [
                    16,
                ],
                [16, 32],
                [16, 32, 64],
            ],
            None,
        ),
        # depth is an int
        (3, 2, [64, 32, 16], None, ValueError),
        (3, 2, [], None, ValueError),
        # depth is a list of int
        (3, [2, 2], 64, None, ValueError),
        (3, [], [16, 32], None, ValueError),
        (3, [2, 2, 2], [64, 32, 16], None, ValueError),
        (3, [2, 2, 3], [64, 32], None, ValueError),
        (3, [2, 3, 2], [[16, 32], [16, 32, 64]], None, ValueError),
        (3, [2, 3, 2], [[16, 32], [16, 32, 64], [16]], None, ValueError),
    ],
)
def test_check_block_param(blocks, depth, param, expected, expected_exception):
    """Test the _check_block_param method."""
    if expected_exception:
        with pytest.raises(expected_exception):
            BaseDeepLearningNetwork._check_block_param(
                blocks=blocks,
                depth=depth,
                param=param,
                param_name="filters",
                block_name="residual block",
                block_depth_name="conv",
                allow_none=True,
            )
    else:
        result = BaseDeepLearningNetwork._check_block_param(
            blocks=blocks,
            depth=depth,
            param=param,
            param_name="filters",
            block_name="block",
            block_depth_name="conv layers",
            allow_none=True,
        )

        assert result == expected
