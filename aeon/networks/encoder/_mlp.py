"""Multi Layer Perceptron Network (MLPNetwork)."""

__maintainer__ = ["hadifawaz1999"]

from aeon.networks.base import BaseDeepLearningNetwork


class MLPNetwork(BaseDeepLearningNetwork):
    """Establish the network structure for a MLP.

    Adapted from the implementation used in [1]_

    Parameters
    ----------
    n_layers : int, default = 3
        The number of dense layers in the MLP.
    n_units : int or list of int, default = 500
        Number of units in each dense layer, if not a list, the same units number
        is used for all layers, len(list) should be n_layers.
    activation : str or list of str, default = 'relu'
        Activation function(s) for each dense layer, if not a list, the same activation
        function is used for all layers, len(list) should be n_layers.
    dropout_rate : float or list of float, default = None
        Dropout rate(s) for each dense layer. If None, a default rate of 0.2 is used,
        except the first element, being 0.1. Dropout rate(s) are typically a number
        in the interval [0, 1]. If not a list, the same dropout rate is used for all
        layers, len(list) should be n_layers.
    dropout_last : float, default = 0.3
        The dropout rate of the last layer.
    use_bias : bool or list of bool, default = True
        Condition on whether or not to use bias values for dense layers, if not
        a list, the same condition is used for all layers, len(list) should be n_layers.

    Notes
    -----
    Adapted from the implementation from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py

    References
    ----------
    .. [1]  Wang et al. Time series classification from scratch with deep neural
    networks: A strong baseline, IJCNN, 2017.
    """

    _config = {
        **BaseDeepLearningNetwork._config,
        "structure": "encoder",
    }

    def __init__(
        self,
        n_layers: int = 3,
        n_units: int | list[int] = 500,
        activation: str | list[str] = "relu",
        dropout_rate: float | list[float] = None,
        dropout_last: float = 0.3,
        use_bias: bool = True,
    ):
        self.n_layers = n_layers
        self.n_units = n_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout_last = dropout_last
        self.use_bias = use_bias

        super().__init__()

    def _check_params(self):
        default_dropout = [0.1] + [0.2] * (self.n_layers - 1)
        self._n_units = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.n_units, "units", default=500
        )
        self._activation = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.activation, "activation", allow_none=True
        )
        self._dropout_rate = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.dropout_rate, "dropout rate", default_dropout
        )
        self._use_bias = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.use_bias, "use bias", default=True
        )
        self._dropout_last = self.dropout_last if self.dropout_last is not None else 0.3

    def build_base_graph(self, x):
        import tensorflow as tf

        self._check_params()

        input_layer_flattened = tf.keras.layers.Flatten()(x)

        x = input_layer_flattened

        for idx in range(0, self.n_layers):
            x = tf.keras.layers.Dropout(self._dropout_rate[idx])(x)
            x = tf.keras.layers.Dense(
                self._n_units[idx],
                activation=self._activation[idx],
                use_bias=self.use_bias,
            )(x)

        return x

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
        import tensorflow as tf

        input_layer = tf.keras.layers.Input(input_shape)
        x = self.build_base_graph(input_layer)
        output_layer = tf.keras.layers.Dropout(self._dropout_last)(x)

        return input_layer, output_layer
