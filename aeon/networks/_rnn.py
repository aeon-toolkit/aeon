"""Implements a Recurrent Neural Network (RNN) for time series forecasting."""

__maintainer__ = []

from aeon.networks.base import BaseDeepLearningNetwork


class RNNNetwork(BaseDeepLearningNetwork):
    """
    A class to implement a Recurrent Neural Network (RNN) for time series forecasting.

    Parameters
    ----------
    rnn_type : str, default='lstm'
        Type of RNN cell to use ('lstm', 'gru', or 'simple').
    n_layers : int, default=1
        Number of recurrent layers.
    n_units : list or int, default=64
        Number of units in each recurrent layer. If an int, the same number
        of units is used in each layer.
    dropout_rate : float, default=0.2
        Dropout rate for regularization.
    bidirectional : bool, default=False
        Whether to use bidirectional recurrent layers.
    activation : str or list of str, default='tanh'
        Activation function for the recurrent layers. If a string, the same
        activation is used for all layers. If a list, specifies activation
        for each layer.
    return_sequences : bool or list, default=None
        Whether to return the full sequence (True) or just the last output (False).
        If None, returns sequences for all layers except the last one.
        If a list, specifies return_sequences for each layer.

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
        rnn_type="lstm",
        n_layers=1,
        n_units=64,
        dropout_rate=0.2,
        bidirectional=False,
        activation="tanh",
        return_sequences=None,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.n_layers = n_layers
        self.n_units = n_units
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.activation = activation
        self.return_sequences = return_sequences

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
                f"Unknown type: {self.rnn_type}. Choose from 'lstm', 'gru', or 'simple'"
            )

        # Process n_units to a list
        if isinstance(self.n_units, int):
            self.n_units = [self.n_units] * self.n_layers
        elif len(self.n_units) != self.n_layers:
            raise ValueError(
                f"Length of n_units ({len(self.n_units)}) must match "
                f"n_layers ({self.n_layers})"
            )

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

        # Process return_sequences to a list
        if self.return_sequences is None:
            self.return_sequences = [True] * (self.n_layers - 1) + [False]
        elif isinstance(self.return_sequences, bool):
            self.return_sequences = [self.return_sequences] * self.n_layers
        elif len(self.return_sequences) != self.n_layers:
            raise ValueError(
                f"Length of return_sequences ({len(self.return_sequences)}) must match "
                f"n_layers ({self.n_layers})"
            )

        # Select RNN cell type
        if self.rnn_type == "lstm":
            rnn_cell = tf.keras.layers.LSTM
        elif self.rnn_type == "gru":
            rnn_cell = tf.keras.layers.GRU
        else:  # simple
            rnn_cell = tf.keras.layers.SimpleRNN

        # Create input layer
        input_layer = tf.keras.layers.Input(shape=input_shape)
        x = input_layer

        # Build RNN layers
        for i in range(self.n_layers):
            # Create the recurrent layer
            if self.bidirectional:
                x = tf.keras.layers.Bidirectional(
                    rnn_cell(
                        units=self.n_units[i],
                        activation=self._activation[i],
                        return_sequences=self.return_sequences[i],
                        name=f"{self.rnn_type}_{i+1}",
                    )
                )(x)
            else:
                x = rnn_cell(
                    units=self.n_units[i],
                    activation=self._activation[i],
                    return_sequences=self.return_sequences[i],
                    name=f"{self.rnn_type}_{i+1}",
                )(x)

            # Add dropout for regularization
            if self.dropout_rate > 0:
                x = tf.keras.layers.Dropout(self.dropout_rate, name=f"dropout_{i+1}")(x)

        # Return input and output layers
        return input_layer, x
