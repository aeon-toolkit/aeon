"""TCNForecaster module for deep learning forecasting in aeon."""

from __future__ import annotations

__maintainer__ = []
__all__ = ["TCNForecaster"]

from aeon.forecasting.deep_learning.base import BaseDeepForecaster
from aeon.networks._tcn import TemporalConvolutionalNetwork


class TCNForecaster(BaseDeepForecaster):
    """A deep learning forecaster using Temporal Convolutional Network (TCN).

    It leverages the `TemporalConvolutionalNetwork` from aeon's network module
    to build the architecture suitable for forecasting tasks.

    Parameters
    ----------
    horizon : int, default=1
        Forecasting horizon, the number of steps ahead to predict.
    window : int, default=10
        The window size for creating input sequences.
    batch_size : int, default=32
        Batch size for training the model.
    epochs : int, default=100
        Number of epochs to train the model.
    verbose : int, default=0
        Verbosity mode (0, 1, or 2).
    optimizer : str or tf.keras.optimizers.Optimizer, default='adam'
        Optimizer to use for training.
    loss : str or tf.keras.losses.Loss, default='mse'
        Loss function for training.
    random_state : int, default=None
        Seed for random number generators.
    axis : int, default=0
        Axis along which to apply the forecaster.
    num_inputs : int, default=1
        Number of input channels/features in the input sequence.
    num_channels : list of int, default=[16, 16, 16]
        List specifying the number of output channels for each layer of the
        TCN. The length determines the depth of the network.
    kernel_size : int, default=2
        Size of the convolutional kernel in the TCN.
    dropout : float, default=0.2
        Dropout rate applied after each convolutional layer for
        regularization.


    """

    def __init__(
        self,
        horizon=1,
        window=10,
        batch_size=32,
        epochs=100,
        verbose=0,
        optimizer="adam",
        loss="mse",
        random_state=None,
        axis=0,
        num_inputs=1,
        num_channels=None,
        kernel_size=2,
        dropout=0.2,
    ):
        super().__init__(
            horizon=horizon,
            window=window,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            optimizer=optimizer,
            loss=loss,
            random_state=random_state,
            axis=axis,
        )
        self.num_inputs = num_inputs
        self.num_channels = num_channels if num_channels is not None else [16, 16, 16]
        self.kernel_size = kernel_size
        self.dropout = dropout

    def _add_linear_layer(self, x, output_units):
        """Add a linear layer to the output of the TCN network.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor from the TCN output, typically of shape
            (batch_size, channels, sequence_length).
        output_units : int
            Number of output units for the linear layer, typically matching
            the forecasting horizon.

        Returns
        -------
        tf.Tensor
            Output tensor after applying the linear layer.
        """
        import tensorflow as tf

        # Take the last time step's output for forecasting
        x_last = x[:, -1, :]
        # Apply a dense layer to map to the desired output size (horizon)
        output = tf.keras.layers.Dense(output_units)(x_last)
        return output

    def _build_model(self, input_shape):
        """Build the TCN model for forecasting.

        Parameters
        ----------
        input_shape : tuple
            Shape of input data, typically (window, num_inputs).

        Returns
        -------
        model : tf.keras.Model
            Compiled Keras model with TCN architecture.
        """
        import tensorflow as tf

        # Initialize the TCN network with the updated parameters
        network = TemporalConvolutionalNetwork(
            num_inputs=self.num_inputs,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )

        # Build the network with the given input shape
        input_layer, output = network.build_network(input_shape=input_shape)

        # Adjust the output layer to match the forecasting horizon
        output = self._add_linear_layer(output, self.horizon)

        # Create the final model
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        return model
