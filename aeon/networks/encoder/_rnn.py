"""Implements a Recurrent Neural Network (RNN) for time series forecasting."""

__maintainer__ = []

import numpy as np

from aeon.networks.base import BaseDeepLearningNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies(["tensorflow"], severity="none"):
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="aeon")
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
    rnn_type : str, default='lstm'
        Type of RNN cell to use ('lstm', 'gru', or 'simple').
    n_layers : int, default=1
        Number of recurrent layers.
    n_units : int or list of int, default=64
        Number of units in each recurrent layer. If an int, the same number
        of units is used in each layer. If a list, specifies the number of
        units for each layer and must match the number of layers.
    dropout_intermediate : float or list of float, default=0.0
        Dropout rate applied after each intermediate recurrent layer (not last layer).
    dropout_output : float, default=0.0
        Dropout rate applied after the last recurrent layer.
    residual : float or list of float or 2D numpy array of float, default=0
        Residual connections strength for each layer (see [3]).
        Zero means no residual connection. One means full residual connection.
        A float value between 0 and 1 means a weighted residual connection.
        If a residual connect two layers with different number of units,
        a conv1D layer with kernel_size=1 is added to match the shape.
        When a 2D numpy array is provided, it should have shape (n_layers, n_layers) and
        represent residual connections from input of layer i to output of layer j.
        No connections where 'j' is less than 'i' are allowed.
    bidirectional : bool, default=False
        Whether to use bidirectional recurrent layers.
    activation : str or list of str, default='tanh'
        Activation function(s) for the recurrent layers. If a string, the same
        activation is used for all layers. If a list, specifies activation for
        each layer and must match the number of layers.
    return_sequence_last : bool, default=False
        Whether the last recurrent layer returns the full sequence (True)
        or just the last output (False).
    attention : bool or list of bool, default=False
        Whether to apply self-attention mechanism after each recurrent layer (see [4]).
    use_bias : bool or list of bool, default = True
        Condition on whether or not to use bias values in the convolution layers in
        one residual block, if not a list, the same kernel size is used in all
        convolution layers.

    References
    ----------
    .. [1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
       Neural computation, 9(8), 1735-1780.
    .. [2] Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
       Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using
       RNN encoder-decoder for statistical machine translation.
       arXiv preprint arXiv:1406.1078.
    .. [3] Kim, J., El-Khamy, M., & Lee, J. (2017). Residual LSTM: Design
       of a deep recurrent architecture for distant speech recognition.
       arXiv preprint arXiv:1701.03360.
    .. [4] Wen, X., & Li, W. (2023). Time series prediction based on
       LSTM-attention-LSTM model. IEEE access, 11, 48322-48331.
    """

    _config = {
        **BaseDeepLearningNetwork._config,
        "structure": "encoder",
    }

    def __init__(
        self,
        rnn_type="simple",
        n_layers=1,
        n_units=64,
        dropout_intermediate=0.0,
        dropout_output=0.0,
        residual=0,
        bidirectional=False,
        activation="tanh",
        return_sequence_last=False,
        attention=False,
        use_bias=True,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.n_layers = n_layers
        self.n_units = n_units
        self.dropout_intermediate = dropout_intermediate
        self.dropout_output = dropout_output
        self.residual = residual
        self.bidirectional = bidirectional
        self.activation = activation
        self.return_sequence_last = return_sequence_last
        self.attention = attention
        self.use_bias = use_bias

    def _check_params(self):
        if self.rnn_type not in ["lstm", "gru", "simple"]:
            raise ValueError(
                f"Unknown RNN type: {self.rnn_type}. "
                "Should be 'lstm', 'gru' or 'simple'"
            )

        self._n_units = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.n_units, "units", default=64
        )
        self._dropout_intermediate = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers - 1, self.dropout_intermediate, "dropout", default=0
        )
        self._bidirectional = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.bidirectional, "bidirectional", default=False
        )
        self._activation = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.activation, "activations", allow_none=True
        )
        self._attention = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.attention, "attention", default=False
        )
        self._use_bias = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.use_bias, "biases", default=True
        )
        self._residual = RecurrentNetwork._check_residual_matrix(
            self.n_layers, self.residual
        )
        self._rnn_cell = RecurrentNetwork._check_rnn_cell(
            self.rnn_type,
        )

    @staticmethod
    def _check_residual_matrix(n_layers, residual):
        # if not matrix return diagonal :
        if not isinstance(residual, np.ndarray) or residual.ndim <= 1:
            residual = BaseDeepLearningNetwork._check_layer_param(
                n_layers, residual, "residual", default=0
            )
            # if given as a single value, remove
            # the useless redisual between input layer and first layer
            if isinstance(residual, (int, float)):
                residual[0] = 0

            return np.diag(residual)

        # check matrix shape
        if residual.shape != (n_layers, n_layers):
            raise ValueError(
                f"Residual matrix shape {residual.shape} does not match "
                f"the number of layers {n_layers}. "
                "It should be a square matrix of shape (n_layers, n_layers)."
            )
        # check that no connections exists where 'j' is less than 'i'
        for i in range(n_layers):
            for j in range(i):
                if residual[i, j] != 0:
                    raise ValueError(
                        f"Residual connection from layer {i} to layer {j} "
                        "is not allowed. Only connections where 'j' is greater "
                        "than or equal to 'i' are allowed."
                    )
        return residual

    @staticmethod
    def _check_rnn_cell(rnn_type):
        if rnn_type == "lstm":
            return tf.keras.layers.LSTM
        elif rnn_type == "gru":
            return tf.keras.layers.GRU
        elif rnn_type == "simple":
            return tf.keras.layers.SimpleRNN

        raise ValueError(
            f"Unknown RNN type: {rnn_type}. " "Should be 'lstm', 'gru' or 'simple'"
        )

    def _build_dropout_layer(self, x, i):
        # if last layer, apply output dropout; otherwise, apply intermediate dropout
        if i == (self.n_layers - 1):
            if self.dropout_output > 0:
                return tf.keras.layers.Dropout(
                    self.dropout_output, name="dropout_output"
                )(x)
        else:
            if self._dropout_intermediate[i] > 0:
                return tf.keras.layers.Dropout(
                    self._dropout_intermediate[i], name=f"dropout_intermediate_{i+1}"
                )(x)
        return x

    def _build_rnn_cell(self, x, i):
        # Create the recurrent layer
        cell = self._rnn_cell(
            units=self._n_units[i],
            activation=self._activation[i],
            return_sequences=True,
            use_bias=self._use_bias[i],
            name=f"{self.rnn_type}_{i+1}",
        )

        if self._bidirectional[i]:
            x = tf.keras.layers.Bidirectional(cell)(x)
        else:
            x = cell(x)

        if self._attention[i]:
            x = tf.keras.layers.Attention(name=f"attention_{i+1}")([x, x])

        x = self._build_dropout_layer(x, i)
        return x

    def _build_skip_connection(self, x, x_save, to):
        x_skip = []
        for fr in range(to + 1):
            if self._residual[fr, to] > 0:
                # match shape skip shape if different from x
                sk = x_save[fr]
                if x_save[fr].shape[-1] != x.shape[-1]:
                    sk = tf.keras.layers.Conv1D(
                        filters=x.shape[-1],
                        kernel_size=1,
                        activation=None,
                        name=f"residual_reshape[{fr+1}-{to+1}]",
                    )(sk)

                # if _residual is a weight
                if self._residual[fr, to] != 1:
                    # create tf constant with the weight
                    w = round(self._residual[fr, to], 2)
                    sk = ConstantMultiply(
                        self._residual[fr, to], name=f"multiply[{fr+1}-{to+1}]_x{w}"
                    )(sk)

                x_skip.append(sk)

        if len(x_skip) > 0:
            x_skip.append(x)
            x = tf.keras.layers.Add(name=f"residuals_of_layer_{to+1}")(x_skip)

        return x

    def build_base_graph(self, x):
        self._check_params()
        x_save = []

        # Build RNN layers
        for i in range(self.n_layers):

            # Store residual connection for the current layer
            x_save.append(x)

            x = self._build_rnn_cell(x, i)
            x = self._build_skip_connection(x, x_save, to=i)

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
