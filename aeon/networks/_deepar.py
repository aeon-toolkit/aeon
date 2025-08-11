"""Implementation of DeepAR Network for probabilistic time series forecasting."""

__maintainer__ = []

from aeon.networks.base import BaseDeepLearningNetwork


class DeepARNetwork(BaseDeepLearningNetwork):
    """DeepAR Network for probabilistic time series forecasting.

    A deep learning architecture that combines LSTM encoding with probabilistic
    output layers to generate forecasts with uncertainty estimates.

    Parameters
    ----------
    lstm_units : int, default=None
        Number of LSTM units. If None, calculated as 4 * (1 + log(n_features)^4).
    dense_units : int, default=None
        Number of dense layer units. If None, calculated as 4 * (1 + log(n_features)).
    dropout : float, default=0.1
        Dropout rate applied in LSTM layer for regularization.

    Notes
    -----
    DeepAR uses probabilistic outputs by modeling the conditional distribution
    of future values given past observations. The network outputs parameters
    of a Gaussian distribution for each prediction step.

    References
    ----------
    .. [1] Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020).
           DeepAR: Probabilistic forecasting with autoregressive recurrent networks.
           International journal of forecasting, 36(3), 1181-1191.

    Examples
    --------
    >>> from aeon.networks._deepar import DeepARNetwork
    >>> network = DeepARNetwork()
    >>> input_layer, output = network.build_network(input_shape=(10, 3))
    >>> input_layer.shape, output.shape
    ((None, 10, 3), (None, 2, 3))
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
        "structure": "encoder-decoder",
    }

    def __init__(
        self,
        lstm_units: int = None,
        dense_units: int = None,
        dropout: float = 0.1,
    ):
        """Initialize the DeepAR architecture.

        Parameters
        ----------
        lstm_units : int, optional
            Number of LSTM units. Auto-calculated if None.
        dense_units : int, optional
            Number of dense units. Auto-calculated if None.
        dropout : float, default=0.1
            Dropout rate for LSTM regularization.
        """
        super().__init__()
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout = dropout

    def _calculate_units(self, n_features: int) -> tuple:
        """Calculate optimal number of units based on input features.

        Parameters
        ----------
        n_features : int
            Number of input features.

        Returns
        -------
        tuple
            (lstm_units, dense_units) calculated based on feature dimensions.
        """
        import math

        if self.lstm_units is None:
            lstm_units = max(4, int(4 * (1 + pow(math.log(max(n_features, 2)), 4))))
        else:
            lstm_units = self.lstm_units

        if self.dense_units is None:
            dense_units = max(4, int(4 * (1 + math.log(max(n_features, 2)))))
        else:
            dense_units = self.dense_units

        return lstm_units, dense_units

    def _create_gaussian_output_layer(self, input_tensor, output_dim: int):
        """Create Gaussian output layer that outputs mean and sigma parameters.

        This layer creates two separate dense layers for mean and variance
        parameters of the Gaussian distribution, avoiding Lambda layers.

        Parameters
        ----------
        input_tensor : tf.keras.layers.Layer
            Input tensor from previous layer.
        output_dim : int
            Output dimensionality.

        Returns
        -------
        list
            List containing [mean_output, sigma_output] layers.
        """
        import tensorflow as tf

        # Mean parameter (can be any real value)
        mean_output = tf.keras.layers.Dense(
            units=output_dim,
            activation="linear",
            kernel_initializer="glorot_normal",
            bias_initializer="glorot_normal",
            name="gaussian_mean",
        )(input_tensor)

        # Sigma parameter (must be positive)
        # Using softplus activation to ensure positive values
        sigma_output = tf.keras.layers.Dense(
            units=output_dim,
            activation="softplus",
            kernel_initializer="glorot_normal",
            bias_initializer="glorot_normal",
            name="gaussian_sigma",
        )(input_tensor)

        # Add small epsilon for numerical stability using only keras layers
        # Create a constant layer for epsilon
        # epsilon_layer = tf.keras.layers.Dense(
        #     units=output_dim,
        #     kernel_initializer='zeros',
        #     bias_initializer=tf.keras.initializers.Constant(1e-6),
        #     trainable=False,
        #     name='epsilon_constant'
        # )(input_tensor)

        # # Add epsilon to sigma for numerical stability
        # sigma_stabilized = tf.keras.layers.Add(name='sigma_stabilized')([
        #     sigma_output,
        #     epsilon_layer
        # ])

        mean_reshaped = tf.keras.layers.Reshape((1, output_dim))(mean_output)
        sigma_reshaped = tf.keras.layers.Reshape((1, output_dim))(sigma_output)

        concatenated = tf.keras.layers.Concatenate(
            axis=1, name="gaussian_distribution_params"
        )([mean_reshaped, sigma_reshaped])

        return concatenated

    def _build_encoder(self, input_layer, n_features: int):
        """Build the LSTM encoder part of the network.

        Parameters
        ----------
        input_layer : tf.keras.layers.Input
            Input layer of the network.
        n_features : int
            Number of input features.

        Returns
        -------
        tf.keras.layers.Layer
            Encoded representation from LSTM.
        """
        import tensorflow as tf

        lstm_units, _ = self._calculate_units(n_features)

        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(
                f"Dropout rate must be between 0 and 1, got {self.dropout}. "
            )
        # LSTM encoder layer
        lstm_output = tf.keras.layers.LSTM(
            units=lstm_units,
            return_sequences=False,
            dropout=self.dropout,
            name="deepar_lstm_encoder",
        )(input_layer)

        return lstm_output

    def _build_decoder(self, encoded_input, n_features: int):
        """Build the decoder part with Gaussian output.

        Parameters
        ----------
        encoded_input : tf.keras.layers.Layer
            Encoded input from LSTM layer.
        n_features : int
            Number of output features.

        Returns
        -------
        list
            List containing [mean, sigma] output layers.
        """
        import tensorflow as tf

        _, dense_units = self._calculate_units(n_features)

        # Dense hidden layer
        dense_output = tf.keras.layers.Dense(
            units=dense_units, activation="relu", name="deepar_dense_hidden"
        )(encoded_input)

        # Gaussian output layer
        gaussian_outputs = self._create_gaussian_output_layer(
            dense_output, output_dim=n_features
        )

        return gaussian_outputs

    def build_network(self, input_shape: tuple, **kwargs) -> tuple:
        """Build the complete DeepAR architecture.

        Constructs an LSTM encoder followed by dense layers that output
        parameters for a Gaussian distribution (mean and variance).

        Parameters
        ----------
        input_shape : tuple
            Shape of input data (n_timepoints, n_channels).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        tuple
            A tuple containing (input_layer, gaussian_outputs) where
            gaussian_outputs is a list [mean, sigma] representing the
            Gaussian distribution parameters.

        Notes
        -----
        The network outputs two tensors representing the mean and standard
        deviation of a Gaussian distribution for probabilistic forecasting.
        """
        import tensorflow as tf

        # Create input layer
        input_layer = tf.keras.layers.Input(shape=input_shape, name="deepar_input")

        # Extract number of features
        n_features = input_shape[1]  # (n_timepoints, n_features)

        # Build encoder (LSTM)
        encoded = self._build_encoder(input_layer, n_features)

        # Build decoder (Dense + Gaussian output)
        gaussian_outputs = self._build_decoder(encoded, n_features)

        return input_layer, gaussian_outputs
