"""TCNForecaster module for deep learning forecasting in aeon."""

from __future__ import annotations

__maintainer__ = []

__all__ = ["TCNForecaster"]

import os
import time
from copy import deepcopy
from typing import Any

import numpy as np
from sklearn.utils import check_random_state

from aeon.forecasting.base import IterativeForecastingMixin
from aeon.forecasting.deep_learning.base import BaseDeepForecaster
from aeon.networks._tcn import TCNNetwork


class TCNForecaster(BaseDeepForecaster, IterativeForecastingMixin):
    """A deep learning forecaster using Temporal Convolutional Network (TCN).

    Adapted from the implementation used in [1]_. Leverages the `TCNNetwork` from
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
    n_blocks : list of int, default=[16, 16, 16]
        List specifying the number of output channels for each layer of the TCN.
        The length determines the depth of the network.
    kernel_size : int, default=2
        Size of the convolutional kernel in the TCN.
    dropout : float, default=0.2
        Dropout rate applied after each convolutional layer for regularization.
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
    .. [1] Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of
       generic convolutional and recurrent networks for sequence modeling.
       arXiv preprint arXiv:1803.01271.
    """

    _tags = {
        "python_dependencies": ["tensorflow"],
        "capability:horizon": True,
        "capability:multivariate": False,
        "capability:exogenous": False,
        "capability:univariate": True,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "cant_pickle": True,
    }

    def __init__(
        self,
        window,
        horizon=1,
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
        n_blocks=None,
        kernel_size=2,
        dropout=0.2,
        save_last_model=False,
        save_init_model=False,
        best_file_name="best_model",
        init_file_name="init_model",
    ):
        self.window = window
        self.horizon = horizon
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
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.dropout = dropout
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

    def build_model(self, input_shape):
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

        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)

        tf.keras.utils.set_random_seed(self.random_state_)
        network = TCNNetwork(
            n_blocks=self.n_blocks if self.n_blocks is not None else [16, 16, 16],
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )

        input_layer, output = network.build_network(input_shape=input_shape)

        model = tf.keras.Model(inputs=input_layer, outputs=output)

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
        Fit the TCN forecaster model to the training data.

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
            Predicted values for the specified horizon. Since TCN has single
            horizon capability, returns single step prediction.
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
        if num_channels == 1:
            prediction = pred.flatten()[0]
        else:
            prediction = pred[0, :]
        return float(prediction)

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
        param = {
            "window": 10,
            "n_epochs": 10,
            "batch_size": 4,
            "n_blocks": [8, 8],
            "kernel_size": 2,
            "dropout": 0.1,
        }
        return [param]
