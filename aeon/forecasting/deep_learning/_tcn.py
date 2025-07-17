"""TCNForecaster module for deep learning forecasting in aeon."""

from __future__ import annotations

__maintainer__ = []

__all__ = ["TCNForecaster"]

from typing import Any

from aeon.forecasting.deep_learning.base import BaseDeepForecaster
from aeon.networks._tcn import TCNNetwork


class TCNForecaster(BaseDeepForecaster):
    """A deep learning forecaster using Temporal Convolutional Network (TCN).

    It leverages the `TCNNetwork` from aeon's network module
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
    n_blocks : list of int, default=[16, 16, 16]
        List specifying the number of output channels for each layer of the
        TCN. The length determines the depth of the network.
    kernel_size : int, default=2
        Size of the convolutional kernel in the TCN.
    dropout : float, default=0.2
        Dropout rate applied after each convolutional layer for
        regularization.
    """

    _tags = {
        "python_dependencies": ["tensorflow"],
        "capability:horizon": True,
        "capability:multivariate": True,
        "capability:exogenous": False,
        "capability:univariate": True,
    }

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
        n_blocks=None,
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
            random_state=random_state,
            axis=axis,
            loss=loss,
        )
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.dropout = dropout

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
        network = TCNNetwork(
            n_blocks=self.n_blocks if self.n_blocks is not None else [16, 16, 16],
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )

        # Build the network with the given input shape
        input_layer, output = network.build_network(input_shape=input_shape)

        # Create the final model
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        return model

    # Added to handle __name__ in tests (class-level access)
    @classmethod
    def _get_test_params(
        cls, parameter_set: str = "default"
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For forecasters, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        param = {
            "epochs": 10,
            "batch_size": 4,
            "n_blocks": [8, 8],
            "kernel_size": 2,
            "dropout": 0.1,
        }
        return [param]
