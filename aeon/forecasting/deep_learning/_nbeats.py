"""N-BEATS Forecaster module for deep learning forecasting in aeon."""

from __future__ import annotations

__maintainer__ = []
__all__ = ["NBeatsForecaster"]

import numpy as np
import tensorflow as tf
from sklearn.utils import check_random_state

from aeon.forecasting.base import DirectForecastingMixin
from aeon.forecasting.deep_learning.base import BaseDeepForecaster


def smape_loss(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (SMAPE) loss."""
    epsilon = 0.1
    numerator = tf.abs(y_true - y_pred)
    denominator = tf.abs(y_true) + tf.abs(y_pred) + epsilon
    return 200.0 * tf.reduce_mean(numerator / denominator)


class _NBeatsDataGenerator(tf.keras.utils.Sequence):
    """Generates training data batches via random sampling."""

    def __init__(self, y, window, horizon, batch_size, steps_per_epoch, random_state):
        self.y = y
        self.window = window
        self.horizon = horizon
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.rng = check_random_state(random_state)
        self.valid_starts = np.arange(self.window, len(self.y) - self.horizon + 1)

        if len(self.valid_starts) == 0:
            raise ValueError(
                f"Data length ({len(y)}) is insufficient for window "
                f"({window}) and horizon ({horizon})."
            )

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        indices = self.rng.choice(self.valid_starts, size=self.batch_size)
        X_batch = []
        y_batch = []

        for idx in indices:
            X_batch.append(self.y[idx - self.window : idx])
            y_batch.append(self.y[idx : idx + self.horizon])
        return np.array(X_batch), np.array(y_batch)


class _NBeatsBlock(tf.keras.layers.Layer):
    """N-BEATS basic building block."""

    def __init__(
        self,
        stack_type,
        input_width,
        forecast_width,
        thetas_dim,
        n_layers,
        layer_width,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stack_type = stack_type
        self.input_width = input_width
        self.forecast_width = forecast_width
        self.thetas_dim = thetas_dim
        self.n_layers = n_layers
        self.layer_width = layer_width

        self.fc_stack = [
            tf.keras.layers.Dense(layer_width, activation="relu")
            for _ in range(n_layers)
        ]

        self.theta_b_fc = tf.keras.layers.Dense(thetas_dim, use_bias=False)
        self.theta_f_fc = tf.keras.layers.Dense(thetas_dim, use_bias=False)
        if self.stack_type == "generic":
            self.generic_backcast = tf.keras.layers.Dense(input_width)
            self.generic_forecast = tf.keras.layers.Dense(forecast_width)

    def call(self, x):
        h = x
        for layer in self.fc_stack:
            h = layer(h)

        theta_b = self.theta_b_fc(h)
        theta_f = self.theta_f_fc(h)
        return self._basis_function(theta_b, theta_f)

    def _basis_function(self, theta_b, theta_f):
        if self.stack_type == "generic":
            return self._basis_generic(theta_b, theta_f)
        elif self.stack_type == "trend":
            return self._basis_trend(theta_b, theta_f)
        elif self.stack_type == "seasonality":
            return self._basis_seasonality(theta_b, theta_f)
        else:
            raise ValueError(f"Unknown stack type: {self.stack_type}")

    def _basis_generic(self, theta_b, theta_f):
        backcast = self.generic_backcast(theta_b)
        forecast = self.generic_forecast(theta_f)
        return backcast, forecast

    def _basis_trend(self, theta_b, theta_f):
        t_b = np.linspace(0, 1, self.input_width)
        t_f = np.linspace(0, 1, self.forecast_width)
        degree = self.thetas_dim
        T_b = np.stack([t_b**i for i in range(degree)], axis=1)
        T_f = np.stack([t_f**i for i in range(degree)], axis=1)
        T_b = tf.cast(tf.constant(T_b), dtype=theta_b.dtype)
        T_f = tf.cast(tf.constant(T_f), dtype=theta_f.dtype)
        backcast = tf.matmul(theta_b, T_b, transpose_b=True)
        forecast = tf.matmul(theta_f, T_f, transpose_b=True)
        return backcast, forecast

    def _basis_seasonality(self, theta_b, theta_f):
        t_b = np.arange(self.input_width) / self.input_width
        t_f = np.arange(self.forecast_width) / self.forecast_width
        harmonics = self.thetas_dim // 2

        def get_s(t_vec):
            s_list = []
            for i in range(harmonics):
                s_list.append(tf.cos(2.0 * np.pi * (i + 1) * t_vec))
                s_list.append(tf.sin(2.0 * np.pi * (i + 1) * t_vec))
            return tf.stack(s_list, axis=1)

        S_b = get_s(t_b)
        S_f = get_s(t_f)

        S_b = tf.cast(S_b, dtype=theta_b.dtype)
        S_f = tf.cast(S_f, dtype=theta_f.dtype)

        backcast = tf.matmul(theta_b, S_b, transpose_b=True)
        forecast = tf.matmul(theta_f, S_f, transpose_b=True)
        return backcast, forecast

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stack_type": self.stack_type,
                "input_width": self.input_width,
                "forecast_width": self.forecast_width,
                "thetas_dim": self.thetas_dim,
                "n_layers": self.n_layers,
                "layer_width": self.layer_width,
            }
        )
        return config


class NBeatsForecaster(BaseDeepForecaster, DirectForecastingMixin):
    """N-BEATS: Neural Basis Expansion Analysis for Time Series.

    An interpretable deep learning architecture for time series forecasting.
    It decomposes the series into stacks (e.g. Trend, Seasonality) and aggregates
    their forecasts.

    Parameters
    ----------
    window : int, default=10
        The length of the input sequence (lookback period).
    horizon : int, default=1
        Forecasting horizon.
    stack_types : list of str, default=("trend", "seasonality")
        Ordered list of stack types. Options: "generic", "trend", "seasonality".
    nb_blocks_per_stack : int, default=3
        Number of blocks in each stack.
    trend_degree : int, default=2
        Polynomial degree for "trend" stacks (p in paper). Ignored for others.
    num_harmonics : int, default=None
        Number of harmonics for "seasonality" stacks. If None, defaults to
        horizon / 2 rounded up. Ignored for others.
    generic_dim : int, default=32
        Projection dimension for "generic" stacks. Ignored for others.
    share_weights_in_stack : bool, default=True
        If True, blocks within the same stack share weights.
    hidden_layer_units : int, default=256
        Width of FC layers in blocks.
    nb_layers_per_block : int, default=4
        Depth of FC layers in blocks.
    batch_size : int, default=1024
        Training batch size. Paper recommends large batches (1024).
    n_epochs : int, default=100
        Training epochs.
    steps_per_epoch : int, default=50
        Number of random batches to sample per epoch.
    optimizer : str, default="adam"
        Optimizer.
    loss : str, default="smape"
        Loss function. Options: "mse", "mae", "smape" (recommended).
    random_state : int, default=None
        Seed for reproducibility.
    file_path : str, default="./"
        Path to save models.

    References
    ----------
    .. [1] Oreshkin, Boris N., et al. "N-BEATS: Neural basis expansion analysis for
       interpretable time series forecasting." ICLR 2020.
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
        window=10,
        horizon=1,
        stack_types=("trend", "seasonality"),
        nb_blocks_per_stack=3,
        trend_degree=2,
        num_harmonics=None,
        generic_dim=32,
        share_weights_in_stack=True,
        hidden_layer_units=256,
        nb_layers_per_block=4,
        batch_size=1024,
        n_epochs=100,
        steps_per_epoch=50,
        optimizer="adam",
        loss="smape",
        verbose=0,
        callbacks=None,
        random_state=None,
        file_path="./",
        save_best_model=False,
        save_last_model=False,
    ):
        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.trend_degree = trend_degree
        self.num_harmonics = num_harmonics
        self.generic_dim = generic_dim
        self.share_weights_in_stack = share_weights_in_stack
        self.hidden_layer_units = hidden_layer_units
        self.nb_layers_per_block = nb_layers_per_block
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.optimizer = optimizer
        self.loss = loss
        self.random_state = random_state
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model

        super().__init__(
            window=window,
            horizon=horizon,
            verbose=verbose,
            callbacks=callbacks,
            file_path=file_path,
        )

    def _resolve_thetas_dim(self, stack_type):
        """Determine expansion coefficients dimension based on stack type."""
        if stack_type == "trend":
            return self.trend_degree
        elif stack_type == "seasonality":
            harmonics = self.num_harmonics
            if harmonics is None:
                harmonics = int(np.ceil(self.horizon / 2))
            return 2 * harmonics
        else:
            return self.generic_dim

    def build_model(self, input_shape):
        """Build the N-BEATS model."""
        inputs = tf.keras.Input(shape=input_shape)
        if len(input_shape) > 1:
            x_res = tf.keras.layers.Flatten()(inputs)
        else:
            x_res = inputs

        forecast_total = None

        for i, stack_type in enumerate(self.stack_types):
            thetas_dim = self._resolve_thetas_dim(stack_type)
            shared_block = None
            if self.share_weights_in_stack:
                shared_block = _NBeatsBlock(
                    stack_type=stack_type,
                    input_width=self.window,
                    forecast_width=self.horizon,
                    thetas_dim=thetas_dim,
                    n_layers=self.nb_layers_per_block,
                    layer_width=self.hidden_layer_units,
                    name=f"stack_{i}_{stack_type}_shared",
                )

            stack_forecast_acc = None

            for block_idx in range(self.nb_blocks_per_stack):
                if self.share_weights_in_stack:
                    block = shared_block
                else:
                    block = _NBeatsBlock(
                        stack_type=stack_type,
                        input_width=self.window,
                        forecast_width=self.horizon,
                        thetas_dim=thetas_dim,
                        n_layers=self.nb_layers_per_block,
                        layer_width=self.hidden_layer_units,
                        name=f"stack_{i}_{stack_type}_block_{block_idx}",
                    )

                backcast, forecast = block(x_res)
                x_res = tf.keras.layers.Subtract()([x_res, backcast])
                if stack_forecast_acc is None:
                    stack_forecast_acc = forecast
                else:
                    stack_forecast_acc = tf.keras.layers.Add()(
                        [stack_forecast_acc, forecast]
                    )

            stack_name = f"stack_{i}_{stack_type}_out"
            stack_output = tf.keras.layers.Identity(name=stack_name)(stack_forecast_acc)
            if forecast_total is None:
                forecast_total = stack_output
            else:
                forecast_total = tf.keras.layers.Add()([forecast_total, stack_output])

        outputs = tf.keras.layers.Reshape((self.horizon, 1), name="final_forecast")(
            forecast_total
        )
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        loss_fn = self.loss
        if loss_fn == "smape":
            loss_fn = smape_loss
        model.compile(optimizer=self.optimizer, loss=loss_fn)
        return model

    def _fit(self, y, exog=None):
        """Fit the model."""
        rng = check_random_state(self.random_state)
        seed = rng.randint(0, np.iinfo(np.int32).max)
        tf.keras.utils.set_random_seed(seed)

        y_inner = y
        if y_inner.ndim == 1:
            y_inner = y_inner.reshape(-1, 1)

        data_gen = _NBeatsDataGenerator(
            y=y_inner,
            window=self.window,
            horizon=self.horizon,
            batch_size=self.batch_size,
            steps_per_epoch=self.steps_per_epoch,
            random_state=seed,
        )

        input_shape = (self.window, y_inner.shape[1])
        self.model_ = self.build_model(input_shape)

        callbacks_list = self._prepare_callbacks()
        callbacks_list.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=5, min_lr=1e-5
            )
        )
        self.history_ = self.model_.fit(
            data_gen,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=callbacks_list,
        )
        if self.save_last_model:
            self.save_last_model_to_file(file_path=self.file_path)
        self.last_window_ = y_inner[-self.window :]
        return self

    def _predict(self, y=None, exog=None):
        """Predict standard forecast."""
        X_pred = self._prepare_input(y)
        y_pred = self.model_.predict(X_pred, verbose=0)
        return y_pred[0, :, 0]

    def predict_decomposition(self, y=None):
        """Predict and decompose the forecast into stack components."""
        X_pred = self._prepare_input(y)
        outputs = []
        output_names = []

        for layer in self.model_.layers:
            if layer.name.endswith("_out"):
                outputs.append(layer.output)
                output_names.append(layer.name.replace("_out", ""))

        outputs.append(self.model_.output)
        output_names.append("total_forecast")
        decomp_model = tf.keras.Model(inputs=self.model_.input, outputs=outputs)
        results = decomp_model.predict(X_pred, verbose=0)

        if len(outputs) == 1:
            results = [results]
        res_dict = {}
        for name, res in zip(output_names, results):
            if res.ndim == 3:
                res_dict[name] = res[0, :, 0]
            else:
                res_dict[name] = res[0, :]
        return res_dict

    def _prepare_input(self, y):
        if y is None:
            if self.last_window_ is None:
                raise ValueError("Model not fitted and no y provided.")
            X_pred = self.last_window_
        else:
            X_pred = y
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(1, -1, 1)
        elif X_pred.ndim == 2:
            X_pred = X_pred.reshape(1, X_pred.shape[0], X_pred.shape[1])
        if X_pred.shape[1] != self.window:
            X_pred = X_pred[:, -self.window :, :]
        return X_pred

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        params = {
            "window": 10,
            "horizon": 2,
            "n_epochs": 1,
            "batch_size": 4,
            "hidden_layer_units": 8,
            "stack_types": ["trend", "seasonality"],
            "nb_blocks_per_stack": 1,
            "steps_per_epoch": 2,
            "loss": "mse",
        }
        return [params]
