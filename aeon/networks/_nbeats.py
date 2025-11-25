"""Implementation of N-BEATS network for aeon."""

__maintainer__ = []

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
    num_seasonal_coefficients : int, default=5
        The number of Fourier coefficients for the seasonality block.
    num_generic_coefficients : int, default=7
        The number of coefficients for the generic block.
    share_weights : bool, default=True
        If True, weights of the fully connected layers are shared across
        all blocks within a stack.
    share_coefficients : bool, default=True
        If True, the backcast and forecast of each block share the same
        basis expansion coefficients.

    References
    ----------
    .. [1] Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019).
       N-BEATS: Neural basis expansion analysis for interpretable time series
       forecasting. arXiv preprint arXiv:1905.10437.
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
        "structure": "auto-encoder",
    }

    def __init__(
        self,
        horizon=1,
        stacks=None,
        num_blocks_per_stack=2,
        units=30,
        num_trend_coefficients=3,
        num_seasonal_coefficients=5,
        num_generic_coefficients=7,
        share_weights=True,
        share_coefficients=True,
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
        self.share_weights = share_weights
        self.share_coefficients = share_coefficients

    def _trend_model(self, theta, t_, p):
        """
        Generate the trend backcast or forecast.

        Parameters
        ----------
        theta : tf.Tensor
            Basis expansion coefficients, 2-dimensional tensor with shape (N, p)
            where N is the batch size and p is the number of polynomial terms.
        t_ : tf.Tensor
            Time index, 1-dimensional tensor with length t for the trend backcast
            or with length H for the trend forecast.
        p : int
            Number of polynomial terms.

        Returns
        -------
        tf.Tensor
            Predicted trend, 2-dimensional tensor with shape (N, t) when
            backcasting and (N, H) when forecasting where N is the batch size.
        """
        import tensorflow as tf

        T_ = tf.map_fn(lambda i: t_**i, tf.range(p, dtype=tf.float32))
        return tf.tensordot(theta, T_, axes=1)

    def _seasonality_model(self, theta, t_, p):
        """
        Generate the seasonality backcast or forecast.

        Parameters
        ----------
        theta : tf.Tensor
            Basis expansion coefficients, 2-dimensional tensor with shape
            (N, 2 * p) where N is the batch size and 2 * p is the total number of
            Fourier terms, that is p sine functions plus p cosine functions.
        t_ : tf.Tensor
            Time index, 1-dimensional tensor with length t for the seasonality
            backcast or with length H for the seasonality forecast.
        p : int
            Number of Fourier terms.

        Returns
        -------
        tf.Tensor
            Predicted seasonality, 2-dimensional tensor with shape (N, t) when
            backcasting and (N, H) when forecasting where N is the batch size.
        """
        import numpy as np
        import tensorflow as tf

        s1 = tf.map_fn(
            lambda i: tf.math.cos(2 * np.pi * i * t_), tf.range(p, dtype=tf.float32)
        )
        s2 = tf.map_fn(
            lambda i: tf.math.sin(2 * np.pi * i * t_), tf.range(p, dtype=tf.float32)
        )
        S = tf.concat([s1, s2], axis=0)
        return tf.tensordot(theta, S, axes=1)

    def _trend_block(self, h, p, t_b, t_f):
        """
        Derive the trend basis expansion coefficients.

        Parameters
        ----------
        h : tf.Tensor
            Output of 4-layer fully connected stack, 2-dimensional tensor with
            shape (N, k) where N is the batch size and k is the number of hidden
            units of each fully connected layer. Note that all fully connected
            layers have the same number of units.
        p : int
            Number of polynomial terms.
        t_b : tf.Tensor
            Input time index, 1-dimensional tensor with length t used for
            generating the backcast.
        t_f : tf.Tensor
            Output time index, 1-dimensional tensor with length H used for
            generating the forecast.

        Returns
        -------
        tuple
            backcast: tf.Tensor. Trend backcast, 2-dimensional tensor with shape
            (N, t) where N is the batch size and t is the length of the lookback
            period.
            forecast: tf.Tensor. Trend forecast, 2-dimensional tensor with shape
            (N, H) where N is the batch size and H is the length of the forecast
            period.
        """
        import tensorflow as tf

        if self.share_coefficients:
            theta = tf.keras.layers.Dense(units=p, use_bias=False)(h)
            backcast = tf.keras.layers.Lambda(
                self._trend_model, arguments={"p": p, "t_": t_b}
            )(theta)
            forecast = tf.keras.layers.Lambda(
                self._trend_model, arguments={"p": p, "t_": t_f}
            )(theta)
        else:
            theta_b = tf.keras.layers.Dense(units=p, use_bias=False)(h)
            theta_f = tf.keras.layers.Dense(units=p, use_bias=False)(h)
            backcast = tf.keras.layers.Lambda(
                self._trend_model, arguments={"p": p, "t_": t_b}
            )(theta_b)
            forecast = tf.keras.layers.Lambda(
                self._trend_model, arguments={"p": p, "t_": t_f}
            )(theta_f)
        return backcast, forecast

    def _seasonality_block(self, h, p, t_b, t_f):
        """
        Derive the seasonality basis expansion coefficients.

        Parameters
        ----------
        h : tf.Tensor
            Output of 4-layer fully connected stack, 2-dimensional tensor with
            shape (N, k) where N is the batch size and k is the number of hidden
            units of each fully connected layer. Note that all fully connected
            layers have the same number of units.
        p : int
            Number of Fourier terms.
        t_b : tf.Tensor
            Input time index, 1-dimensional tensor with length t used for
            generating the backcast.
        t_f : tf.Tensor
            Output time index, 1-dimensional tensor with length H used for
            generating the forecast.

        Returns
        -------
        tuple
            backcast: tf.Tensor. Seasonality backcast, 2-dimensional tensor with
            shape (N, t) where N is the batch size and t is the length of the
            lookback period.
            forecast: tf.Tensor. Seasonality forecast, 2-dimensional tensor with
            shape (N, H) where N is the batch size and H is the length of the
            forecast period.
        """
        import tensorflow as tf

        if self.share_coefficients:
            theta = tf.keras.layers.Dense(units=2 * p, use_bias=False)(h)
            backcast = tf.keras.layers.Lambda(
                self._seasonality_model, arguments={"p": p, "t_": t_b}
            )(theta)
            forecast = tf.keras.layers.Lambda(
                self._seasonality_model, arguments={"p": p, "t_": t_f}
            )(theta)
        else:
            theta_b = tf.keras.layers.Dense(units=2 * p, use_bias=False)(h)
            theta_f = tf.keras.layers.Dense(units=2 * p, use_bias=False)(h)
            backcast = tf.keras.layers.Lambda(
                self._seasonality_model, arguments={"p": p, "t_": t_b}
            )(theta_b)
            forecast = tf.keras.layers.Lambda(
                self._seasonality_model, arguments={"p": p, "t_": t_f}
            )(theta_f)
        return backcast, forecast

    def _generic_block(self, h, p, t_b, t_f):
        """
        Derive the generic basis expansion coefficients.

        Parameters
        ----------
        h : tf.Tensor
            Output of 4-layer fully connected stack, 2-dimensional tensor with
            shape (N, k) where N is the batch size and k is the number of hidden
            units of each fully connected layer. Note that all fully connected
            layers have the same number of units.
        p : int
            Number of linear terms.
        t_b : tf.Tensor
            Input time index, 1-dimensional tensor with length t used for
            generating the backcast.
        t_f : tf.Tensor
            Output time index, 1-dimensional tensor with length H used for
            generating the forecast.

        Returns
        -------
        tuple
            backcast: tf.Tensor. Generic backcast, 2-dimensional tensor with
            shape (N, t) where N is the batch size and t is the length of the
            lookback period.
            forecast: tf.Tensor. Generic forecast, 2-dimensional tensor with
            shape (N, H) where N is the batch size and H is the length of the
            forecast period.
        """
        import tensorflow as tf

        if self.share_coefficients:
            theta = tf.keras.layers.Dense(units=p, use_bias=False)(h)
            backcast = tf.keras.layers.Dense(units=self.lookback_period)(theta)
            forecast = tf.keras.layers.Dense(units=self.horizon)(theta)
        else:
            theta_b = tf.keras.layers.Dense(units=p, use_bias=False)(h)
            theta_f = tf.keras.layers.Dense(units=p, use_bias=False)(h)
            backcast = tf.keras.layers.Dense(units=self.lookback_period)(theta_b)
            forecast = tf.keras.layers.Dense(units=self.horizon)(theta_f)
        return backcast, forecast

    def _get_block_output(self, stack_type, h, t_b, t_f):
        """
        Get the output of a block based on its type.

        Parameters
        ----------
        stack_type : str
            Type of stack ('trend', 'seasonality', or 'generic').
        h : tf.Tensor
            Output of 4-layer fully connected stack, 2-dimensional tensor with
            shape (N, k).
        t_b : tf.Tensor
            Input time index, 1-dimensional tensor with length t used for
            generating the backcast.
        t_f : tf.Tensor
            Output time index, 1-dimensional tensor with length H used for
            generating the forecast.

        Returns
        -------
        tuple
            backcast: tf.Tensor. Backcast tensor for the block.
            forecast: tf.Tensor. Forecast tensor for the block.
        """
        if stack_type == "trend":
            return self._trend_block(h, self.num_trend_coefficients, t_b, t_f)
        elif stack_type == "seasonality":
            return self._seasonality_block(h, self.num_seasonal_coefficients, t_b, t_f)
        else:
            return self._generic_block(h, self.num_generic_coefficients, t_b, t_f)

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

        if input_shape[-1] > 1:
            raise ValueError("NBeats only supports univariate series .")

        if self.stacks is None:
            self.stacks = ["trend", "seasonality"]
        self.lookback_period = input_shape[0]
        t_ = tf.cast(
            tf.range(0, self.lookback_period + self.horizon), dtype=tf.float32
        ) / (self.lookback_period + self.horizon)
        t_b = t_[: self.lookback_period]
        t_f = t_[self.lookback_period :]
        input_layer = tf.keras.layers.Input(shape=(self.lookback_period,))
        x = input_layer
        for s_idx, stack_type in enumerate(self.stacks):
            if self.share_weights:
                d1 = tf.keras.layers.Dense(self.units, activation="relu")
                d2 = tf.keras.layers.Dense(self.units, activation="relu")
                d3 = tf.keras.layers.Dense(self.units, activation="relu")
                d4 = tf.keras.layers.Dense(self.units, activation="relu")
            for b_idx in range(self.num_blocks_per_stack):
                if s_idx == 0 and b_idx == 0:
                    if self.share_weights:
                        h = d1(x)
                        h = d2(h)
                        h = d3(h)
                        h = d4(h)
                    else:
                        h = tf.keras.layers.Dense(self.units, activation="relu")(x)
                        h = tf.keras.layers.Dense(self.units, activation="relu")(h)
                        h = tf.keras.layers.Dense(self.units, activation="relu")(h)
                        h = tf.keras.layers.Dense(self.units, activation="relu")(h)
                    backcast_block, forecast_block = self._get_block_output(
                        stack_type,
                        h,
                        t_b,
                        t_f,
                    )
                    backcast_residual = tf.keras.layers.Subtract()([x, backcast_block])
                    forecast = forecast_block
                else:
                    if self.share_weights:
                        h = d1(backcast_residual)
                        h = d2(h)
                        h = d3(h)
                        h = d4(h)
                    else:
                        h = tf.keras.layers.Dense(self.units, activation="relu")(
                            backcast_residual
                        )
                        h = tf.keras.layers.Dense(self.units, activation="relu")(h)
                        h = tf.keras.layers.Dense(self.units, activation="relu")(h)
                        h = tf.keras.layers.Dense(self.units, activation="relu")(h)
                    backcast_block, forecast_block = self._get_block_output(
                        stack_type,
                        h,
                        t_b,
                        t_f,
                    )
                    backcast_residual = tf.keras.layers.Subtract()(
                        [backcast_residual, backcast_block]
                    )
                    forecast = tf.keras.layers.Add()([forecast, forecast_block])
        output_layer = [backcast_residual, forecast]
        return input_layer, output_layer
