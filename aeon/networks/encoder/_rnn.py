"""Implements a Recurrent Neural Network (RNN) for time series forecasting."""

__maintainer__ = []

from enum import StrEnum, auto, unique

from aeon.networks.base import BaseDeepLearningNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies(["tensorflow"], severity="none"):
    import tensorflow as tf

    class ConstantMultiply(tf.keras.layers.Layer):
        def __init__(self, w, **kwargs):
            super().__init__(**kwargs)
            self.w = w

        def call(self, inputs):
            return inputs * self.w

        def get_config(self):
            config = super().get_config()
            config.update({"w": self.w})
            return config


@unique
class RNN_TYPE(StrEnum):
    LSTM = auto()
    GRU = auto()
    SIMPLE = auto()

    @staticmethod
    def exists(rnn_type: str) -> bool:
        """Check if the given rnn_type exists in the RNN_TYPE enum."""
        return rnn_type.lower() in (item.value for item in RNN_TYPE)

    @staticmethod
    def _check_params(rnn_type: str):
        if not RNN_TYPE.exists(rnn_type):
            raise ValueError(
                f"Invalid value for 'rnn_type' ({rnn_type}). "
                f"Valid options are: {[item.value for item in RNN_TYPE]}"
            )


class RecurrentNetwork(BaseDeepLearningNetwork):
    """
    Implements a Recurrent Neural Network (RNN) for time series forecasting.

    This implementation provides a flexible RNN architecture that can be configured
    to use different types of recurrent cells including Simple RNN, Long Short-Term
    Memory (LSTM) [1], and Gated Recurrent Unit (GRU) [2]. The network supports
    multiple layers, bidirectional processing, and various dropout configurations
    for regularization.

    Parameters
    ----------
    rnn_type : RNN_TYPE or str, default=RNN_TYPE.LSTM
        Type of RNN cell to use ('lstm', 'gru', or 'simple').
    n_layers : int, default=1
        Number of recurrent layers.
    n_units : list or int, default=64
        Number of units in each recurrent layer. If an int, the same number
        of units is used in each layer. If a list, specifies the number of
        units for each layer and must match the number of layers.
    dropout_intermediate : float, default=0.0
        Dropout rate applied after each intermediate recurrent layer (not last layer).
    dropout_output : float, default=0.0
        Dropout rate applied after the last recurrent layer.
    residual : list or int, default=0
        Residual connections strength for each layer.
        If an int, the same residual strength is used for all layers.
    bidirectional : bool, default=False
        Whether to use bidirectional recurrent layers.
    activation : str or list of str, default='tanh'
        Activation function(s) for the recurrent layers. If a string, the same
        activation is used for all layers. If a list, specifies activation for
        each layer and must match the number of layers.
    return_sequence_last : bool, default=False
        Whether the last recurrent layer returns the full sequence (True)
        or just the last output (False).

    References
    ----------
    .. [1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
       Neural computation, 9(8), 1735-1780.
    .. [2] Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
       Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using
       RNN encoder-decoder for statistical machine translation.
       arXiv preprint arXiv:1406.1078.
    """

    _config = {
        **BaseDeepLearningNetwork._config,
        "structure": "encoder",
    }

    def __init__(
        self,
        rnn_type=RNN_TYPE.SIMPLE,
        n_layers=1,
        n_units=64,
        dropout_intermediate=0.0,
        dropout_output=0.0,
        residual=0,
        bidirectional=False,
        activation="tanh",
        return_sequence_last=False,
    ):
        self.rnn_type = rnn_type.lower()
        self.n_layers = n_layers
        self.n_units = n_units
        self.dropout_intermediate = dropout_intermediate
        self.dropout_output = dropout_output
        self.residual = residual
        self.bidirectional = bidirectional
        self.activation = activation
        self.return_sequence_last = return_sequence_last

        super().__init__()

    def _check_params(self):
        RNN_TYPE._check_params(self.rnn_type)

        self._n_units = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.n_units, "units", default=64
        )
        self._activation = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.activation, "activations", allow_none=True
        )
        self._residual = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.residual, "residual", default=0
        )

    def _get_rnn_cell(self):

        if self.rnn_type == RNN_TYPE.LSTM:
            return tf.keras.layers.LSTM
        elif self.rnn_type == RNN_TYPE.GRU:
            return tf.keras.layers.GRU
        elif self.rnn_type == RNN_TYPE.SIMPLE:
            return tf.keras.layers.SimpleRNN

    def _dropout_layer(self, x, i):

        # if last layer, apply output dropout; otherwise, apply intermediate dropout
        if i == (self.n_layers - 1):
            if self.dropout_output > 0:
                return tf.keras.layers.Dropout(
                    self.dropout_output, name="dropout_output"
                )(x)
        else:
            if self.dropout_intermediate > 0:
                return tf.keras.layers.Dropout(
                    self.dropout_intermediate, name=f"dropout_intermediate_{i+1}"
                )(x)

        return x

    def build_base_graph(self, x):

        self._check_params()
        self._rnn_cell = self._get_rnn_cell()

        # Build RNN layers
        for i in range(self.n_layers):

            # Store residual connection for the current layer
            x_skip = x

            # Create the recurrent layer
            cell = self._rnn_cell(
                units=self._n_units[i],
                activation=self._activation[i],
                return_sequences=True,
                name=f"{self.rnn_type}_{i+1}",
            )

            if self.bidirectional:
                x = tf.keras.layers.Bidirectional(cell)(x)
            else:
                x = cell(x)

            x = self._dropout_layer(x, i)

            # Add residual connection
            if self._residual[i] > 0:
                # match shape skip shape if different from x
                if x_skip.shape[-1] != x.shape[-1]:
                    x_skip = tf.keras.layers.Dense(
                        units=x.shape[-1],
                        activation=None,
                        name=f"residual_dense_{i+1}",
                    )(x_skip)

                # if _residual is a weight
                if self._residual[i] != 1:
                    # create tf constant with the weight
                    # weight = tf.ones_like(x_skip) * self._residual[i]
                    # x_skip = tf.keras.layers.Multiply()([x_skip, weight])
                    x_skip = ConstantMultiply(self._residual[i])(x_skip)

                x = tf.keras.layers.Add(name=f"residual_{i+1}")([x, x_skip])

            if i == (self.n_layers - 1) and not self.return_sequence_last:
                x = x[:, -1, :]

        return x

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer (n_timepoints, n_features)
        kwargs : dict
            Additional keyword arguments to be passed to the network

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        input_layer = tf.keras.layers.Input(shape=input_shape)
        x = self.build_base_graph(input_layer)

        return input_layer, x
