"""Implementation of N-BEATS network for aeon."""

__maintainer__ = ["melpiro"]

import warnings

from aeon.networks.base import BaseDeepLearningNetwork


class NBeatsNetwork(BaseDeepLearningNetwork):
    """
    Implementation of the N-BEATS network architecture.

    N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series
    Forecasting) is a deep learning model for univariate time series
    forecasting. It uses a backward and forward residual connection structure
    and is composed of stacks of fully connected layers.

    The architecture consists of multiple stacks, each containing several blocks.
    Each block generates a forecast and a backcast. The backcast is subtracted
    from the block's input, and the residual is passed to the next block. The
    forecasts from all blocks are summed up to produce the final prediction.

    This implementation is based on the paper by _[1].

    Parameters
    ----------
    horizon : int
        The length of the forecast horizon.
    stacks : list of str, default=["trend", "seasonality"]
        A list of stack types. Allowed types are "trend", "seasonality",
        and "generic".
    num_blocks_per_stack : int, default=3
        The number of blocks within each stack.
    units : int, default=256
        The number of hidden units in the fully connected layers of each block.
    num_trend_coefficients : int, default=3
        The number of polynomial coefficients for the trend block.
    num_seasonal_coefficients : int, default=1
        The number of Fourier coefficients for the seasonality block.
    num_generic_coefficients : int, default="deprecated"
        The number of coefficients for the generic block.
    share_weights : bool, default=True
        If True, weights of the fully connected layers are shared across
        all blocks within a stack.
    share_coefficients : bool, default="deprecated"
        If True, the backcast and forecast of each block share the same
        basis expansion coefficients.

    References
    ----------
    .. [1] Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019).
       N-BEATS: Neural basis expansion analysis for interpretable time series
       forecasting. arXiv preprint arXiv:1905.10437.
    """

    _config = {
        **BaseDeepLearningNetwork._config,
        "structure": "auto-encoder",
    }

    def __init__(
        self,
        horizon=1,
        stacks=None,
        num_blocks_per_stack=3,
        units=256,
        num_trend_coefficients=3,
        num_seasonal_coefficients=1,
        num_generic_coefficients="deprecated",
        share_weights=True,
        share_coefficients="deprecated",
    ):
        """Initialize the N-BEATS network."""
        super().__init__()
        self.horizon = horizon
        self.stacks = stacks
        self.num_blocks_per_stack = num_blocks_per_stack
        self.units = units
        self.num_trend_coefficients = num_trend_coefficients
        self.num_seasonal_coefficients = num_seasonal_coefficients
        self.num_generic_coefficients = num_generic_coefficients
        if num_generic_coefficients != "deprecated":
            warn = "The 'num_generic_coefficients' parameter will be removed in v1.7.0."
            warnings.warn(warn, FutureWarning)
        self.share_weights = share_weights
        self.share_coefficients = share_coefficients
        if share_coefficients != "deprecated":
            warn = "The 'share_coefficients' parameter will be removed in v1.7.0."
            warnings.warn(warn, FutureWarning)

    def _build_dense_block(self, units, activation="relu", n_layers=4):
        import tensorflow as tf

        return tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units, activation=activation)
                for _ in range(n_layers)
            ]
        )

    def _build_output_block(self, stack_type):
        from aeon.utils.networks.nbeats_layers import (
            NbeatsGenericLayer,
            NbeatsSeasonalityLayer,
            NbeatsTrendLayer,
        )

        if stack_type == "trend":
            return NbeatsTrendLayer(
                degree=self.num_trend_coefficients,
                lookback=self.lookback_period,
                horizon=self.horizon,
            )
        elif stack_type == "seasonality":
            return NbeatsSeasonalityLayer(
                harmonics=self.num_seasonal_coefficients,
                lookback=self.lookback_period,
                horizon=self.horizon,
            )
        elif stack_type == "generic":
            return NbeatsGenericLayer(
                lookback=self.lookback_period,
                horizon=self.horizon,
            )
        else:
            raise ValueError(f"Unknown stack type: {stack_type}")

    def build_network(
        self,
        input_shape,
        **kwargs,
    ):
        """
        Build the N-BEATS network.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data (n_timepoints, n_channels).
        **kwargs : dict
            Additional keyword arguments (unused).

        Returns
        -------
        tuple
            (input_layer, output_layer) representing the network.
        """
        import tensorflow as tf

        if self.stacks is None:
            self.stacks = ["trend", "seasonality"]

        if input_shape[-1] > 1:
            raise ValueError("NBeats only supports univariate series .")

        self.lookback_period = input_shape[0]
        input_layer = tf.keras.layers.Input(shape=(self.lookback_period,))

        x = input_layer
        outputs = []
        for stack_type in self.stacks:
            dense = self._build_dense_block(self.units)
            output = self._build_output_block(stack_type)

            for _ in range(self.num_blocks_per_stack):
                h = dense(x)
                backcast_out, forecast_out = output(h)
                x = x - backcast_out  # Update the input for the next block
                outputs.append(forecast_out)

                if not self.share_weights:
                    # create a new set of weights for the next call
                    dense = self._build_dense_block(self.units)
                    output = self._build_output_block(stack_type)

        forecast = tf.keras.layers.Add()(outputs)
        return input_layer, forecast
