"""DLinear Forecaster for time series forecasting.

Implementation of Decomposition-Linear (DLinear) model from the paper:
"Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)
https://ojs.aaai.org/index.php/AAAI/article/view/26317
"""

__maintainer__ = []
__all__ = ["DLinearForecaster"]

from aeon.forecasting.base import SeriesToSeriesForecastingMixin
from aeon.forecasting.deep_learning.base import BaseDeepForecaster


class MovingAverage:
    """
    Moving average block to highlight the trend of time series.

    Uses average pooling with front and end padding to extract the trend component.

    Parameters
    ----------
    kernel_size : int
        Size of the moving average window.
    stride : int
        Stride of the moving average operation.
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x):
        """
        Apply moving average to extract trend.

        Parameters
        ----------
        x : tensorflow tensor
            Input tensor of shape (batch, sequence_length, channels).

        Returns
        -------
        tensorflow tensor
            Trend component of same shape as input.
        """
        import tensorflow as tf

        # Padding on both ends of time series
        front = tf.repeat(x[:, 0:1, :], repeats=(self.kernel_size - 1) // 2, axis=1)
        end = tf.repeat(x[:, -1:, :], repeats=(self.kernel_size - 1) // 2, axis=1)
        x_padded = tf.concat([front, x, end], axis=1)

        # Transpose to (batch, channels, sequence) for 1D pooling
        x_padded = tf.transpose(x_padded, perm=[0, 2, 1])

        # Apply average pooling
        x_pooled = tf.nn.avg_pool1d(
            x_padded, ksize=self.kernel_size, strides=self.stride, padding="VALID"
        )

        # Transpose back to (batch, sequence, channels)
        x_pooled = tf.transpose(x_pooled, perm=[0, 2, 1])

        return x_pooled


class SeriesDecomposition:
    """
    Series decomposition block to split time series into trend and seasonal components.

    Parameters
    ----------
    kernel_size : int
        Kernel size for moving average used in trend extraction.
    """

    def __init__(self, kernel_size):
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def __call__(self, x):
        """
        Decompose time series into trend and seasonal components.

        Parameters
        ----------
        x : tensorflow tensor
            Input time series of shape (batch, sequence_length, channels).

        Returns
        -------
        tuple of tensorflow tensors
            (seasonal_component, trend_component) both of same shape as input.
        """
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class DLinearForecaster(BaseDeepForecaster, SeriesToSeriesForecastingMixin):
    """
    Decomposition-Linear (DLinear) Forecaster.

    DLinear decomposes the time series into trend and seasonal components using a
    moving average, then applies separate linear layers to each component before
    combining them for the final forecast. This explicit handling of trend and
    seasonality improves performance on data with clear patterns.

    From: "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)

    Parameters
    ----------
    window : int
        The lookback window size (number of past time steps to use as input).
    horizon : int, default=1
        The forecast horizon (number of future time steps to predict).
    kernel_size : int, default=25
        Kernel size for the moving average used in decomposition.
        Controls how smooth the extracted trend is.
    individual : bool, default=False
        If True, uses separate linear layers for each channel (channel-independent).
        If False, uses a single shared linear layer across all channels.
    epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=32
        Training batch size.
    learning_rate : float, default=0.001
        Learning rate for the Adam optimizer.
    verbose : int, default=0
        Verbosity mode (0, 1, or 2).
    callbacks : list, default=None
        List of Keras callbacks to apply during training.
    axis : int, default=0
        Axis along which to apply the forecaster.
        0: (n_timepoints, n_channels), 1: (n_channels, n_timepoints)
    last_file_name : str, default="last_dlinear_model"
        Filename for saving the last model.
    file_path : str, default="./"
        Directory path for saving models.

    Attributes
    ----------
    model_ : tf.keras.Model
        The fitted TensorFlow/Keras model.
    history_ : tf.keras.callbacks.History
        Training history containing loss and metrics.
    decomposition_ : SeriesDecomposition
        The series decomposition object.

    Examples
    --------
    >>> from aeon.forecasting.deep_learning import DLinearForecaster
    >>> import numpy as np
    >>> y_train = np.random.randn(1, 100)  # (channels, timepoints)
    >>> forecaster = DLinearForecaster(window=24, horizon=12, epochs=50)
    >>> forecaster.fit(y_train)  # doctest: +SKIP
    >>> predictions = forecaster.series_to_series_forecast(
    ...     y_train, prediction_horizon=12
    ... )  # doctest: +SKIP

    References
    ----------
    .. [1] Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023).
       "Are Transformers Effective for Time Series Forecasting?".
       Proceedings of the AAAI Conference on Artificial Intelligence,
       37(9), 11121-11128.
    """

    _tags = {
        "capability:horizon": True,
        "capability:exogenous": False,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "cant_pickle": True,
        "python_dependencies": "tensorflow",
        "capability:multivariate": True,
    }

    def __init__(
        self,
        window,
        horizon=1,
        kernel_size=25,
        individual=False,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        verbose=0,
        callbacks=None,
        axis=0,
        last_file_name="last_dlinear_model",
        file_path="./",
    ):
        self.kernel_size = kernel_size
        self.individual = individual
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.decomposition_ = None

        super().__init__(
            window=window,
            horizon=horizon,
            verbose=verbose,
            callbacks=callbacks,
            axis=axis,
            last_file_name=last_file_name,
            file_path=file_path,
        )

    def build_model(self, input_shape):
        """
        Construct the DLinear model architecture.

        Parameters
        ----------
        input_shape : tuple
            Shape of input data (batch_size, window, n_channels).

        Returns
        -------
        model : tf.keras.Model
            Compiled Keras model.
        """
        import tensorflow as tf

        _, seq_len, n_channels = input_shape
        pred_len = self.horizon

        # Model receives decomposed seasonal and trend components
        seasonal_input = tf.keras.layers.Input(shape=(seq_len, n_channels))
        trend_input = tf.keras.layers.Input(shape=(seq_len, n_channels))

        if self.individual:
            # Channel-independent mode: separate linear layer for each channel
            seasonal_outputs = []
            trend_outputs = []

            for i in range(n_channels):
                # Extract single channel
                seasonal_channel = seasonal_input[:, :, i : i + 1]
                trend_channel = trend_input[:, :, i : i + 1]

                # Apply dense layer (acts as linear transformation)
                seasonal_out = tf.keras.layers.Dense(
                    pred_len, name=f"seasonal_linear_{i}"
                )(tf.transpose(seasonal_channel, perm=[0, 2, 1]))

                trend_out = tf.keras.layers.Dense(pred_len, name=f"trend_linear_{i}")(
                    tf.transpose(trend_channel, perm=[0, 2, 1])
                )

                # Transpose back
                seasonal_outputs.append(tf.transpose(seasonal_out, perm=[0, 2, 1]))
                trend_outputs.append(tf.transpose(trend_out, perm=[0, 2, 1]))

            # Concatenate all channels
            seasonal_component = tf.concat(seasonal_outputs, axis=2)
            trend_component = tf.concat(trend_outputs, axis=2)
        else:
            # Channel-shared mode: single linear layer for all channels
            # Transpose: (batch, seq, channels) -> (batch, channels, seq)
            seasonal_transposed = tf.transpose(seasonal_input, perm=[0, 2, 1])
            trend_transposed = tf.transpose(trend_input, perm=[0, 2, 1])

            # Apply dense layer
            seasonal_out = tf.keras.layers.Dense(
                pred_len, name="seasonal_linear_shared"
            )(seasonal_transposed)
            trend_out = tf.keras.layers.Dense(pred_len, name="trend_linear_shared")(
                trend_transposed
            )

            # Transpose back: (batch, channels, pred_len) -> (batch, pred_len, channels)
            seasonal_component = tf.transpose(seasonal_out, perm=[0, 2, 1])
            trend_component = tf.transpose(trend_out, perm=[0, 2, 1])

        # Combine seasonal and trend
        output = seasonal_component + trend_component

        # Create model with decomposed inputs
        model = tf.keras.Model(
            inputs=[seasonal_input, trend_input], outputs=output, name="DLinear"
        )

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )

        return model

    def _prepare_data(self, y):
        """
        Prepare windowed sequences for training/prediction.

        Parameters
        ----------
        y : np.ndarray
            Input time series of shape (n_channels, n_timepoints).

        Returns
        -------
        X : np.ndarray
            Windowed input sequences of shape (n_samples, window, n_channels).
        y_target : np.ndarray or None
            Target sequences of shape (n_samples, horizon, n_channels) or None.
        """
        import tensorflow as tf

        # Convert to TensorFlow for decomposition
        y_tensor = tf.constant(y, dtype=tf.float32)

        # Transpose if needed: ensure (n_timepoints, n_channels)
        if self.axis == 1:
            y_tensor = tf.transpose(y_tensor)

        n_timepoints, n_channels = y_tensor.shape

        # Create sliding windows
        X_list = []
        y_list = []

        for i in range(n_timepoints - self.window - self.horizon + 1):
            X_list.append(y_tensor[i : i + self.window, :])
            y_list.append(y_tensor[i + self.window : i + self.window + self.horizon, :])

        if len(X_list) == 0:
            # Not enough data for even one window
            return None, None

        X = tf.stack(X_list, axis=0)
        y_target = tf.stack(y_list, axis=0) if len(y_list) > 0 else None

        return X.numpy(), y_target.numpy() if y_target is not None else None

    def _decompose_data(self, X):
        """
        Decompose input sequences into seasonal and trend components.

        Parameters
        ----------
        X : np.ndarray
            Input sequences of shape (n_samples, window, n_channels).

        Returns
        -------
        seasonal : np.ndarray
            Seasonal component.
        trend : np.ndarray
            Trend component.
        """
        import tensorflow as tf

        X_tensor = tf.constant(X, dtype=tf.float32)
        seasonal, trend = self.decomposition_(X_tensor)
        return seasonal.numpy(), trend.numpy()

    def _fit(self, y, exog=None):
        """
        Fit the DLinear model.

        Parameters
        ----------
        y : np.ndarray
            Training time series of shape (n_channels, n_timepoints).
        exog : np.ndarray, optional
            Exogenous variables (not supported).

        Returns
        -------
        self
        """
        # Initialize decomposition
        self.decomposition_ = SeriesDecomposition(kernel_size=self.kernel_size)

        # Prepare windowed data
        X, y_target = self._prepare_data(y)

        if X is None or y_target is None:
            raise ValueError(
                f"Not enough data to create training samples. "
                f"Need at least {self.window + self.horizon} time points, "
                f"but got {y.shape[1] if self.axis == 1 else y.shape[0]}."
            )

        # Decompose training data
        X_seasonal, X_trend = self._decompose_data(X)

        # Build model if not already built
        if self.model_ is None:
            self.model_ = self.build_model(input_shape=X.shape)

        # Train model
        callbacks = self._prepare_callbacks()
        self.history_ = self.model_.fit(
            [X_seasonal, X_trend],
            y_target,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=callbacks,
        )

        return self

    def _predict(self, y, exog=None):
        """
        Single-step prediction (horizon=1).

        Parameters
        ----------
        y : np.ndarray
            Input time series.
        exog : np.ndarray, optional
            Exogenous variables (not supported).

        Returns
        -------
        float
            Predicted value for next time step.
        """
        # For single-step, we can use series_to_series and take first value
        pred = self._series_to_series_forecast(y, prediction_horizon=1, exog=exog)
        return float(pred[0])

    def _series_to_series_forecast(self, y, prediction_horizon, exog=None):
        """
        Series-to-series forecasting for multiple steps ahead.

        Parameters
        ----------
        y : np.ndarray
            Input time series of shape (n_channels, n_timepoints).
        prediction_horizon : int
            Number of steps to forecast.
        exog : np.ndarray, optional
            Exogenous variables (not supported).

        Returns
        -------
        np.ndarray
            Forecasted values of shape (prediction_horizon,) for univariate
            or (prediction_horizon, n_channels) for multivariate.
        """
        import tensorflow as tf

        # Convert input
        y_tensor = tf.constant(y, dtype=tf.float32)

        # Transpose if needed
        if self.axis == 1:
            y_tensor = tf.transpose(y_tensor)

        # Extract last window
        last_window = y_tensor[-self.window :, :]
        last_window = tf.expand_dims(last_window, axis=0)  # Add batch dimension

        # Decompose
        seasonal, trend = self.decomposition_(last_window)

        # Predict
        predictions = self.model_.predict([seasonal.numpy(), trend.numpy()], verbose=0)

        # predictions shape: (1, prediction_horizon, n_channels)
        predictions = predictions[0]  # Remove batch dimension

        # If univariate, return 1D array
        if predictions.shape[1] == 1:
            return predictions[:, 0]
        else:
            return predictions

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        params1 = {
            "window": 10,
            "horizon": 3,
            "kernel_size": 5,
            "individual": False,
            "epochs": 1,
            "batch_size": 4,
        }
        params2 = {
            "window": 8,
            "horizon": 2,
            "kernel_size": 3,
            "individual": True,
            "epochs": 1,
            "batch_size": 2,
        }
        return [params1, params2]
