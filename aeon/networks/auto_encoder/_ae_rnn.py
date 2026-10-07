"""Auto-Encoder using recurrent neural networks."""

__maintainer__ = []

import numpy as np

from aeon.networks.base import BaseDeepAENetwork
from aeon.networks.encoder._rnn import RecurrentNetwork
from aeon.typing import LATENT_SPACE, RNN_TYPE


class AERecurrentNetwork(BaseDeepAENetwork):
    """
    A class to implement an Auto-Encoder based on Recurrent Neural Networks (RNNs).

    Parameters
    ----------
    latent_space_dim : int, default=128
        Dimension of the latent space.
    latent_space_type : LATENT_SPACE, default = LATENT_SPACE.REPEATED
        Type of latent space to use. Options are:
        - LATENT_SPACE.FLAT: The latent space is a flattened vector.
        - LATENT_SPACE.TIME: The latent space is a time series.
        - LATENT_SPACE.REPEATED: The latent space is a repeated vector.
    rnn_type : RNN_TYPE, default=RNN_TYPE.LSTM
        Type of RNN cell to use. Options are:
        - RNN_TYPE.LSTM: Long Short-Term Memory cell.
        - RNN_TYPE.GRU: Gated Recurrent Unit cell.
        - RNN_TYPE.SIMPLE: Simple RNN cell.
    n_layers : int, default=1
        Number of recurrent layers.
    n_units : int or list of int, default=64
        Number of units in each recurrent layer. If an int, the same number
        of units is used in each layer. If a list, specifies the number of
        units for each layer and must match the number of layers.
    dropout : float or list of float, default=0.0
        Dropout rate applied to each recurrent layer.
    residual : float or list of float or 2D numpy array of float, default=0
        Residual connections strength for each layer.
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
    attention : bool or list of bool, default=False
        Whether to apply self-attention mechanism after each recurrent layer.
    use_bias : bool or list of bool, default=True
        Condition on whether or not to use bias values in the RNN layers.
    """

    _config = {
        **BaseDeepAENetwork._config,
        "structure": "auto-encoder",
    }

    def __init__(
        self,
        latent_space_dim=128,
        latent_space_type=LATENT_SPACE.REPEATED,
        rnn_type=RNN_TYPE.LSTM,
        n_layers=1,
        n_units=64,
        dropout=0.0,
        residual=0,
        bidirectional=False,
        activation="tanh",
        attention=False,
        use_bias=True,
    ):
        super().__init__(latent_space_dim, latent_space_type)
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.n_units = n_units
        self.dropout = dropout
        self.residual = residual
        self.bidirectional = bidirectional
        self.activation = activation
        self.attention = attention
        self.use_bias = use_bias

    def _check_params(self):
        self._n_layers = self.n_layers

        self._latent_space_type = LATENT_SPACE._check_param(
            self.latent_space_type,
        )
        self._rnn_type = RNN_TYPE._check_params(
            self.rnn_type,
        )
        self._n_units = BaseDeepAENetwork._check_layer_param(
            self._n_layers, self.n_units, "units", 50
        )
        self._dropout = BaseDeepAENetwork._check_layer_param(
            self._n_layers, self.dropout, "dropout", 0.0
        )
        self._bidirectional = BaseDeepAENetwork._check_layer_param(
            self._n_layers, self.bidirectional, "bidirectional", False
        )
        self._activation = BaseDeepAENetwork._check_layer_param(
            self._n_layers, self.activation, "activation", "tanh"
        )
        self._attention = BaseDeepAENetwork._check_layer_param(
            self._n_layers, self.attention, "attention", False
        )
        self._residual = RecurrentNetwork._check_residual_matrix(
            self.n_layers, self.residual
        )
        self._use_bias = BaseDeepAENetwork._check_layer_param(
            self._n_layers, self.use_bias, "use_bias", True
        )

    def _build_encoder_graph(self, x):
        encoder = RecurrentNetwork(
            rnn_type=self.rnn_type,
            n_layers=self._n_layers,
            n_units=self._n_units,
            dropout_intermediate=self._dropout[:-1],
            dropout_output=self._dropout[-1],
            residual=self._residual,
            bidirectional=self._bidirectional,
            activation=self._activation,
            return_sequence_last=True,
            attention=self._attention,
            name_prefix="encoder_",
            # use_bias=self._use_bias, # TODO
        )
        return encoder.build_base_graph(x)

    def _build_flat_latent_graph(self, x):
        import tensorflow as tf

        enc_out_shape = x.shape[1:]
        decoder_units = int(np.prod(enc_out_shape))

        x = x[:, -1, :]  # take the last time step
        x = self._build_latent_projection(x)
        x = tf.keras.layers.Dense(units=decoder_units)(x)
        x = tf.keras.layers.Reshape(target_shape=enc_out_shape)(x)
        return x

    def _build_repeated_latent_graph(self, x):
        import tensorflow as tf

        enc_out_shape = x.shape[1:]
        x = x[:, -1, :]  # take the last time step
        x = self._build_latent_projection(x)
        x = tf.keras.layers.RepeatVector(enc_out_shape[0])(x)
        return x

    @staticmethod
    def _transpose_residual_matrix(residual):
        """Transpose the residual matrix to match the decoder layers."""
        return np.flip(residual, axis=(-2, -1)).swapaxes(-1, -2)

    def _build_decoder_graph(self, x):
        residual = AERecurrentNetwork._transpose_residual_matrix(self._residual)

        decoder = RecurrentNetwork(
            rnn_type=self.rnn_type,
            n_layers=self._n_layers,
            n_units=self._n_units[::-1],
            dropout_intermediate=self._dropout[::-1][1:],
            dropout_output=self._dropout[0],
            residual=residual,
            bidirectional=self._bidirectional[::-1],
            activation=self._activation[::-1],
            return_sequence_last=True,
            attention=self._attention[::-1],
            name_prefix="decoder_",
        )
        return decoder.build_base_graph(x)
