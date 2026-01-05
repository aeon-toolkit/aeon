"""LinearForecaster module for deep learning forecasting in aeon."""

from __future__ import annotations

__maintainer__ = []

__all__ = ["LinearForecaster"]

import os
import time
from copy import deepcopy
from typing import Any

import numpy as np
from sklearn.utils import check_random_state

from aeon.forecasting.base import SeriesToSeriesForecastingMixin
from aeon.forecasting.deep_learning.base import BaseDeepForecaster
from aeon.networks._mlp import MLPNetwork


class LinearForecaster(BaseDeepForecaster, SeriesToSeriesForecastingMixin):
    """A deep learning forecaster using Artificial Neural Network (ANN).

    Adapted from the implementation used in [1]_. Leverages the `MLPNetwork` from
    aeon's network module to build the architecture suitable for forecasting tasks.

    Parameters
    ----------
    horizon : int, default=1
        Forecasting horizon, the number of steps ahead to predict.
    batch_size : int, default=32
        Batch size for training the model.
    n_epochs : int, default=100
        Number of epochs to train the model.
    verbose : int, default=0
        Verbosity mode (0, 1, or 2).
    optimizer : str or tf.keras.optimizers.Optimizer, default=None
        Optimizer to use for training.
    metrics : str or list[str|function|keras.metrics.Metric], default="accuracy"
        The evaluation metrics to use during training. Each can be a string, function,
        or a keras.metrics.Metric instance (see https://keras.io/api/metrics/).
        If a single string metric is provided, it will be used as the only metric.
        If a list of metrics are provided, all will be used for evaluation.
    loss : str or tf.keras.losses.Loss, default='mse'
        Loss function for training.
    callbacks : list of tf.keras.callbacks.Callback or None, default=None
        List of Keras callbacks to be applied during training.
    random_state : int, default=None
        Seed for random number generators.
    axis : int, default=0
        Axis along which to apply the forecaster.
    last_file_name : str, default="last_model"
        The name of the file of the last model, used for saving models.
    save_best_model : bool, default=False
        Whether to save the best model during training based on validation loss.
    file_path : str, default="./"
        Directory path where models will be saved.
    save_last_model : bool, default=False
        Whether or not to save the last model, last epoch trained.
    save_init_model : bool, default=False
        Whether to save the initialization of the model.
    best_file_name : str, default="best_model"
        The name of the file of the best model.
    init_file_name : str, default="init_model"
        The name of the file of the init model.

    References
    ----------
    .. [1] Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers
       Effective for Time Series Forecasting?. Proceedings of the AAAI Conference
       on Artificial Intelligence, 37(9), 11121-11128.
       https://doi.org/10.1609/aaai.v37i9.26317.
    """

    _tags = {
        "python_dependencies": ["tensorflow"],
        "capability:horizon": True,
        "capability:multivariate": False,
        "capability:exogenous": False,
        "capability:univariate": True,
        "capability:is_series_to_series": True,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "cant_pickle": True,
    }

    def __init__(
        self,
        window,
        horizon=1,
        type="linear",
        batch_size=32,
        n_epochs=100,
        verbose=0,
        optimizer=None,
        metrics="accuracy",
        loss="mse",
        callbacks=None,
        random_state=None,
        axis=0,
        last_file_name="last_model",
        save_best_model=False,
        file_path="./",
        save_last_model=False,
        save_init_model=False,
        best_file_name="best_model",
        init_file_name="init_model",
    ):
        self.window = window
        self.horizon = horizon
        self.type = type
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.optimizer = optimizer
        self.metrics = metrics
        self.loss = loss
        self.callbacks = callbacks
        self.random_state = random_state
        self.axis = axis
        self.last_file_name = last_file_name
        self.save_best_model = save_best_model
        self.file_path = file_path
        self.save_last_model = save_last_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.init_file_name = init_file_name

        super().__init__(
            horizon=self.horizon,
            window=self.window,
            verbose=self.verbose,
            callbacks=self.callbacks,
            axis=self.axis,
            last_file_name=self.last_file_name,
            file_path=self.file_path,
        )

    def build_model(self, input_shape, type):
        """Build one-layered Linear model for forecasting.

        Parameters
        ----------
        input_shape : tuple
            Shape of input data.

        Returns
        -------
        model : tf.keras.Model
            Compiled Keras model with Linear architecture.
        """
        import tensorflow as tf

        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)

        tf.keras.utils.set_random_seed(self.random_state_)

        num_channels = input_shape[-1]
        network = MLPNetwork(
            n_layers=1,
            n_units=self.horizon * num_channels,
            activation="linear",
            dropout_rate=0.0,
            dropout_last=0.0,
        )
        if type == "linear":
            input_layer, output_layer = network.build_network(input_shape=input_shape)

        if type == "nlinear":
            from tensorflow.keras import layers

            input_layer = layers.Input(shape=input_shape)
            x_last = input_layer[:, -1:, :]
            x_norm = layers.Subtract(name="normalization")([input_layer, x_last])

            inner_input, inner_output = network.build_network(input_shape=input_shape)
            inner_model = tf.keras.Model(inputs=inner_input, outputs=inner_output)

            linear_preds = inner_model(x_norm)
            x = layers.Reshape((self.horizon, num_channels))(linear_preds)
            output_layer = layers.Add(name="denormalization")([x, x_last])

        if type == "dlinear":
            from tensorflow.keras import layers

            kernel_size = 25
            pad_len = (kernel_size - 1) // 2

            def _moving_avg_decomp(x):
                front = tf.gather(x[:, 0:1, :], repeats=pad_len, axis=1)
                end = tf.gather(x[:, -1:, :], repeats=pad_len, axis=1)
                x_padded = tf.concat([front, x, end], axis=1)

                trend = layers.AveragePooling1D(
                    pool_size=kernel_size, strides=1, padding="valid"
                )(x_padded)

                seasonal = layers.Subtract()([x, trend])
                return seasonal, trend

            seasonal_init, trend_init = _moving_avg_decomp(input_layer)

            seasonal_network = MLPNetwork(
                n_layers=1,
                n_units=self.horizon * num_channels,
                activation="linear",
                dropout_rate=0.0,
                dropout_last=0.0,
            )
            trend_network = MLPNetwork(
                n_layers=1,
                n_units=self.horizon * num_channels,
                activation="linear",
                dropout_rate=0.0,
                dropout_last=0.0,
            )

            s_in, s_out = seasonal_network.build_network(input_shape=input_shape)
            t_in, t_out = trend_network.build_network(input_shape=input_shape)

            linear_seasonal = tf.keras.Model(inputs=s_in, outputs=s_out)
            linear_trend = tf.keras.Model(inputs=t_in, outputs=t_out)

            seasonal_output = linear_seasonal(seasonal_init)
            trend_output = linear_trend(trend_init)

            combined = layers.Add()([seasonal_output, trend_output])
            output_layer = layers.Reshape((self.horizon, num_channels))(combined)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        self.optimizer_ = (
            tf.keras.optimizers.Adam() if self.optimizer is None else self.optimizer
        )

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=self._metrics,
        )

        return model

    def _fit(self, y, exog=None):
        """
        Fit the Linear Forecaster model to the training data.

        Parameters
        ----------
        y : np.ndarray or pd.Series
            Target time series to which to fit the forecaster.

        Returns
        -------
        self : object
        """
        import tensorflow as tf

        y_inner = y
        num_timepoints, num_channels = y_inner.shape
        num_sequences = num_timepoints - self.window - self.horizon + 1

        if y_inner.shape[0] < self.window + self.horizon:
            raise ValueError(
                f"Data length ({y_inner.shape}) is insufficient for window "
                f"({self.window}) and horizon ({self.horizon})."
            )

        if isinstance(self.metrics, list):
            self._metrics = self.metrics
        elif isinstance(self.metrics, str):
            self._metrics = [self.metrics]

        windows_full = np.lib.stride_tricks.sliding_window_view(
            y_inner, window_shape=(self.window, num_channels)
        )
        windows_full = np.squeeze(windows_full, axis=1)
        X_train = windows_full[:num_sequences]

        tail = y_inner[self.window :]
        y_windows = np.lib.stride_tricks.sliding_window_view(
            tail, window_shape=(self.horizon, num_channels)
        )
        y_windows = np.squeeze(y_windows, axis=1)
        y_train = y_windows[:num_sequences]

        input_shape = X_train.shape[1:]
        self.training_model_ = self.build_model(input_shape)

        if self.save_init_model:
            self.training_model_.save(self.file_path + self.init_file_name + ".keras")

        self.file_name_ = (
            self.best_file_name if self.save_best_model else str(time.time_ns())
        )

        callbacks_list = self._prepare_callbacks()
        callbacks_list.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=50, min_lr=0.0001
            )
        )

        self.history = self.training_model_.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=callbacks_list,
        )

        try:
            self.model_ = tf.keras.models.load_model(
                self.file_path + self.file_name_ + ".keras", compile=False
            )
            if not self.save_best_model:
                os.remove(self.file_path + self.file_name_ + ".keras")
        except ValueError:
            self.model_ = deepcopy(self.training_model_)

        if self.save_last_model:
            self.save_last_model_to_file(file_path=self.file_path)

        self.last_window_ = y_inner[-self.window :]

        return self

    def _predict(self, y=None, exog=None):
        """Make forecasts for y.

        Parameters
        ----------
        y : np.ndarray or pd.Series, default=None
            Series to predict from. If None, uses last fitted window.

        Returns
        -------
        predictions : np.ndarray
            Predicted values for the specified horizon.
        """
        if y is None:
            if not hasattr(self, "last_window_"):
                raise ValueError("No fitted data available for prediction.")
            y_inner = self.last_window_
        else:
            y_inner = y
            if y_inner.ndim == 1:
                y_inner = y_inner.reshape(-1, 1)
            if y_inner.shape[0] < self.window:
                raise ValueError(
                    f"Input data length ({y_inner.shape}) is less than the "
                    f"window size ({self.window})."
                )
        y_inner = y_inner[-self.window :]
        num_channels = y_inner.shape[-1]
        last_window = y_inner.reshape(1, self.window, num_channels)
        pred = self.model_.predict(last_window, verbose=0)
        prediction = np.squeeze(pred, axis=0)
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, num_channels)
        return prediction

    def _series_to_series_forecast(
        self, y, prediction_horizon, exog=None
    ) -> np.ndarray:
        """
        Forecasts the `prediction_horizon` using a single forward pass.

        Uses model's multi-input multi-output capability to predict up to
        `self.horizon` steps ahead in one go. If `prediction_horizon` exceeds
        `self.horizon`, recursively predicts in chunks.

        Parameters
        ----------
        y : np.ndarray
            The time series data used to form the last window for prediction.
        prediction_horizon : int
            The number of future time steps to forecast (ph).
        exog : np.ndarray, default=None
            Optional future exogenous data.

        Returns
        -------
        np.ndarray
            An array of shape (prediction_horizon,) containing the forecasts.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before forecasting.")

        if exog is not None:
            raise NotImplementedError(
                "Exogenous variables not supported in LinearForecaster."
            )

        if prediction_horizon <= self.horizon:
            full_preds = self._predict(y, exog)
            return full_preds[:prediction_horizon]

        else:
            num_channels = y.shape[-1]
            preds = np.zeros((prediction_horizon, num_channels))
            current_y = y.copy()
            current_idx = 0

            while current_idx < prediction_horizon:
                new_preds = self._predict(current_y, exog)
                remaining_steps = prediction_horizon - current_idx
                chunk_size = min(self.horizon, remaining_steps)
                preds[current_idx : current_idx + chunk_size] = new_preds[:chunk_size]
                current_y = np.append(current_y, new_preds)
                current_idx += chunk_size

            return preds

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
            special parameters are defined for a value, will return "default" set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
        """
        param = {"window": 10, "n_epochs": 10, "batch_size": 4}
        return [param]
