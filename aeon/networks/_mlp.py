"""Multi Layer Perceptron Network (MLPNetwork)."""

__maintainer__ = ["hadifawaz1999"]


import typing

from aeon.networks.base import BaseDeepLearningNetwork


class MLPNetwork(BaseDeepLearningNetwork):
    """Establish the network structure for a MLP.

    Adapted from the implementation used in [1]_

    Parameters
    ----------
    n_layers : int, optional (default=3)
        The number of dense layers in the MLP.
    n_units : Union[int, List[int]], optional (default=200)
        Number of units in each dense layer.
    activation : Union[str, List[str]], optional (default='relu')
        Activation function(s) for each dense layer.
    dropout_rate : Union[int, float, List[Union[int, float]]], optional (default=None)
        Dropout rate(s) for each dense layer. If None, a default rate of 0.2 is used.

    Notes
    -----
    Adapted from the implementation from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py

    References
    ----------
    .. [1]  Wang et al. Time series classification from scratch with deep neural
    networks: A strong baseline, IJCNN, 2017.
    """

    def __init__(
        self,
        n_layers: int = 3,
        n_units: typing.Union[int, list[int]] = 200,
        activation: typing.Union[str, list[str]] = "relu",
        dropout_rate: typing.Union[int, list[int]] = None,
    ):
        super().__init__()

        self._n_layers = n_layers

        if isinstance(activation, str):
            self._activation = [activation] * self._n_layers
        elif isinstance(activation, list):
            assert (
                len(activation) == self._n_layers
            ), "There should be an `activation` function associated with each layer."
            assert all(
                isinstance(a, str) for a in activation
            ), "Activation must be a list of strings."
            assert (
                len(activation) == n_layers
            ), "Activation list length must match number of layers."
            self._activation = activation

        if dropout_rate is None:
            self._dropout_rate = [0.2] * self._n_layers
        elif isinstance(dropout_rate, (int, float)):
            self._dropout_rate = [float(dropout_rate)] * self._n_layers
        elif isinstance(dropout_rate, list):
            assert (
                len(dropout_rate) == self._n_layers
            ), "There should be a `dropout_rate` associated with each layer."
            assert all(
                isinstance(d, (int, float)) for d in dropout_rate
            ), "Dropout rates must be int or float."
            assert (
                len(dropout_rate) == n_layers
            ), "Dropout list length must match number of layers."
            self._dropout_rate = [float(d) for d in dropout_rate]

        if isinstance(n_units, int):
            self._n_units = [n_units] * self._n_layers
        elif isinstance(n_units, list):
            assert all(
                isinstance(u, int) for u in n_units
            ), "`n_units` must be int for all layers."
            assert (
                len(n_units) == n_layers
            ), "`n_units` length must match number of layers."
            self._n_units = n_units

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple of shape = (n_timepoints (m), n_channels (d))
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        from tensorflow import keras

        input_layer = keras.layers.Input(input_shape)
        input_layer_flattened = keras.layers.Flatten()(input_layer)

        x = keras.layers.Dropout(self._dropout_rate[0])(input_layer_flattened)
        x = keras.layers.Dense(self._n_units[0], activation=self._activation[0])(x)

        for idx in range(1, self._n_layers):
            x = keras.layers.Dropout(self._dropout_rate[idx])(x)
            x = keras.layers.Dense(
                self._n_units[idx], activation=self._activation[idx]
            )(x)

        output_layer = keras.layers.Dropout(0.3)(x)

        return input_layer, output_layer
