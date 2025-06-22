"""Implements a Recurrent Neural Network (RNN) for time series forecasting."""

__maintainer__ = []

from aeon.networks.base import BaseDeepLearningNetwork


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
    n_units : list or int, default=64
        Number of units in each recurrent layer. If an int, the same number
        of units is used in each layer. If a list, specifies the number of
        units for each layer and must match the number of layers.
    dropout_intermediate : float, default=0.0
        Dropout rate applied after each intermediate recurrent layer (not last layer).
    dropout_output : float, default=0.0
        Dropout rate applied after the last recurrent layer.
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
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
        "structure": "encoder",
    }

    def __init__(
        self,
        rnn_type="simple",
        n_layers=1,
        n_units=64,
        dropout_intermediate=0.0,
        dropout_output=0.0,
        bidirectional=False,
        activation="tanh",
        return_sequence_last=False,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.n_layers = n_layers
        self.n_units = n_units
        self.dropout_intermediate = dropout_intermediate
        self.dropout_output = dropout_output
        self.bidirectional = bidirectional
        self.activation = activation
        self.return_sequence_last = return_sequence_last
        self._rnn_cell = None

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
        import tensorflow as tf

        # Validate parameters
        if self.rnn_type not in ["lstm", "gru", "simple"]:
            raise ValueError(
                f"Unknown RNN type: {self.rnn_type}. Should be 'lstm', 'gru' 'simple'"
            )

        # Process n_units to a list
        if isinstance(self.n_units, list):
            if len(self.n_units) != self.n_layers:
                raise ValueError(
                    f"Number of units {len(self.n_units)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._n_units = self.n_units
        else:
            self._n_units = [self.n_units] * self.n_layers

        # Process activation to a list
        if isinstance(self.activation, list):
            if len(self.activation) != self.n_layers:
                raise ValueError(
                    f"Number of activations {len(self.activation)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._activation = self.activation
        else:
            self._activation = [self.activation] * self.n_layers

        # Select RNN cell type
        if self.rnn_type == "lstm":
            self._rnn_cell = tf.keras.layers.LSTM
        elif self.rnn_type == "gru":
            self._rnn_cell = tf.keras.layers.GRU
        else:  # simple
            self._rnn_cell = tf.keras.layers.SimpleRNN

        # Create input layer
        input_layer = tf.keras.layers.Input(shape=input_shape)
        x = input_layer

        # Build RNN layers
        for i in range(self.n_layers):
            # Determine return_sequences for current layer
            # All layers except the last must return sequences for stacking
            # The last layer uses the return_sequence_last parameter
            is_last_layer = i == (self.n_layers - 1)
            return_sequences = (not is_last_layer) or self.return_sequence_last

            # Create the recurrent layer
            if self.bidirectional:
                x = tf.keras.layers.Bidirectional(
                    self._rnn_cell(
                        units=self._n_units[i],
                        activation=self._activation[i],
                        return_sequences=return_sequences,
                        name=f"{self.rnn_type}_{i+1}",
                    )
                )(x)
            else:
                x = self._rnn_cell(
                    units=self._n_units[i],
                    activation=self._activation[i],
                    return_sequences=return_sequences,
                    name=f"{self.rnn_type}_{i+1}",
                )(x)

            # Add appropriate dropout based on layer position
            if is_last_layer:
                # Apply output dropout to the last layer
                if self.dropout_output > 0:
                    x = tf.keras.layers.Dropout(
                        self.dropout_output, name="dropout_output"
                    )(x)
            else:
                # Apply intermediate dropout to all layers except the last
                if self.dropout_intermediate > 0:
                    x = tf.keras.layers.Dropout(
                        self.dropout_intermediate, name=f"dropout_intermediate_{i+1}"
                    )(x)

        # Return input and output layers
        return input_layer, x
