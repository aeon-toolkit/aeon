"""NBeatsForecaster module for deep learning forecasting in aeon."""

from __future__ import annotations

__maintainer__ = ["lucifer4073"]

__all__ = ["NBeatsForecaster"]

import os
import time
from copy import deepcopy
from typing import Any

import numpy as np
from sklearn.utils import check_random_state

from aeon.forecasting.base import SeriesToSeriesForecastingMixin
from aeon.forecasting.deep_learning.base import BaseDeepForecaster
from aeon.networks._nbeats import NBeatsNetwork


class NBeatsForecaster(BaseDeepForecaster, SeriesToSeriesForecastingMixin):
    """N-BEATS deep learning forecaster.

    Wraps ``NBeatsNetwork`` into the aeon forecasting API. Each block produces
    a backcast (input reconstruction) and a forecast (future prediction).
    Both heads are trained simultaneously; only the forecast is returned at
    inference time. Based on [1]_.

    Parameters
    ----------
    window : int
        Lookback window size.
    horizon : int, default=1
        Number of steps to forecast. Fixed at construction time.
    backcast_loss_weight : float, default=0.5
        Loss weight for the backcast head relative to the forecast head (1.0).
    batch_size : int, default=32
    n_epochs : int, default=100
    verbose : int, default=0
    optimizer : tf.keras.optimizers.Optimizer or None, default=None
        Defaults to Adam.
    metrics : str or list, default="mae"
        Metrics tracked on the forecast head during training.
    loss : str or tf.keras.losses.Loss, default="mse"
        Applied to both forecast and backcast outputs.
    callbacks : list or None, default=None
    random_state : int or None, default=None
    axis : int, default=0
    last_file_name : str, default="last_model"
    save_best_model : bool, default=False
    save_last_model : bool, default=False
    save_init_model : bool, default=False
    best_file_name : str, default="best_model"
    init_file_name : str, default="init_model"
    file_path : str, default="./"
    stacks : list of str or None, default=None
        Stack types: "trend", "seasonality", or "generic".
        Defaults to ["trend", "seasonality"].
    num_blocks_per_stack : int, default=2
    units : int, default=30
    num_trend_coefficients : int, default=3
    num_seasonal_coefficients : int, default=5
    num_generic_coefficients : int, default=7
    share_weights : bool, default=True
    share_coefficients : bool, default=True

    References
    ----------
    .. [1] Oreshkin et al. (2019). N-BEATS: Neural basis expansion analysis
       for interpretable time series forecasting. arXiv:1905.10437.

    Examples
    --------
    >>> from aeon.datasets import load_airline
    >>> from aeon.forecasting.deep_learning import NBeatsForecaster
    >>> y = load_airline()
    >>> forecaster = NBeatsForecaster(window=24, horizon=12, n_epochs=10)
    >>> forecaster.fit(y)
    NBeatsForecaster(...)
    >>> pred = forecaster.predict(y)
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
        backcast_loss_weight=0.5,
        batch_size=32,
        n_epochs=100,
        verbose=0,
        optimizer=None,
        metrics="mae",
        loss="mse",
        callbacks=None,
        random_state=None,
        axis=0,
        last_file_name="last_model",
        save_best_model=False,
        save_last_model=False,
        save_init_model=False,
        best_file_name="best_model",
        init_file_name="init_model",
        file_path="./",
        stacks=None,
        num_blocks_per_stack=2,
        units=30,
        num_trend_coefficients=3,
        num_seasonal_coefficients=5,
        num_generic_coefficients=7,
        share_weights=True,
        share_coefficients=True,
    ):
        self.window = window
        self.horizon = horizon
        self.backcast_loss_weight = backcast_loss_weight
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
        self.save_last_model = save_last_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.init_file_name = init_file_name
        self.file_path = file_path
        self.stacks = stacks
        self.num_blocks_per_stack = num_blocks_per_stack
        self.units = units
        self.num_trend_coefficients = num_trend_coefficients
        self.num_seasonal_coefficients = num_seasonal_coefficients
        self.num_generic_coefficients = num_generic_coefficients
        self.share_weights = share_weights
        self.share_coefficients = share_coefficients

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
        """Build and compile the N-BEATS Keras model.

        Declares both ``"forecast"`` (shape ``(batch, horizon)``) and
        ``"backcast"`` (shape ``(batch, window)``) as named outputs so the
        full graph is reachable and both heads receive direct gradients.

        Parameters
        ----------
        input_shape : tuple
            ``(window, 1)`` — channel dim satisfies NBeatsNetwork's guard.

        Returns
        -------
        model : tf.keras.Model
        """
        import tensorflow as tf

        self._metrics = (
            [self.metrics] if isinstance(self.metrics, str) else self.metrics
        )

        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)
        tf.keras.utils.set_random_seed(self.random_state_)

        network = NBeatsNetwork(
            horizon=self.horizon,
            stacks=self.stacks,
            num_blocks_per_stack=self.num_blocks_per_stack,
            units=self.units,
            num_trend_coefficients=self.num_trend_coefficients,
            num_seasonal_coefficients=self.num_seasonal_coefficients,
            num_generic_coefficients=self.num_generic_coefficients,
            share_weights=self.share_weights,
            share_coefficients=self.share_coefficients,
        )

        input_layer, (backcast_output, forecast_output) = network.build_network(
            input_shape=input_shape
        )

        model = tf.keras.Model(
            inputs=input_layer,
            outputs={"forecast": forecast_output, "backcast": backcast_output},
        )

        self.optimizer_ = (
            tf.keras.optimizers.Adam() if self.optimizer is None else self.optimizer
        )

        model.compile(
            loss={"forecast": self.loss, "backcast": self.loss},
            loss_weights={"forecast": 1.0, "backcast": self.backcast_loss_weight},
            optimizer=self.optimizer_,
            metrics={"forecast": self._metrics},
        )

        return model

    def _fit(self, y, exog=None):
        """Fit N-BEATS to training data.

        Parameters
        ----------
        y : np.ndarray of shape (n_timepoints, 1)
        exog : ignored

        Returns
        -------
        self
        """
        import tensorflow as tf

        num_timepoints, num_channels = y.shape

        if num_channels > 1:
            raise ValueError(
                f"NBeatsForecaster only supports univariate series, "
                f"got n_channels={num_channels}."
            )

        if num_timepoints < self.window + self.horizon:
            raise ValueError(
                f"Data length ({num_timepoints}) is insufficient for "
                f"window ({self.window}) + horizon ({self.horizon})."
            )

        num_sequences = num_timepoints - self.window - self.horizon + 1
        series_1d = y[:, 0]

        X_train = np.lib.stride_tricks.sliding_window_view(
            series_1d, window_shape=self.window
        )[
            :num_sequences
        ]  # (num_sequences, window)

        y_forecast = np.lib.stride_tricks.sliding_window_view(
            series_1d[self.window :], window_shape=self.horizon
        )[
            :num_sequences
        ]  # (num_sequences, horizon)

        # Backcast target is the input window itself (reconstruction objective)
        y_train_dict = {"forecast": y_forecast, "backcast": X_train}

        self.training_model_ = self.build_model(input_shape=(self.window, 1))

        if self.save_init_model:
            self.training_model_.save(self.file_path + self.init_file_name + ".keras")

        self.file_name_ = (
            self.best_file_name if self.save_best_model else str(time.time_ns())
        )

        callbacks_list = self._prepare_callbacks()
        callbacks_list.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=50, min_lr=1e-5
            )
        )

        self.history = self.training_model_.fit(
            X_train,
            y_train_dict,
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
        except (ValueError, OSError):
            self.model_ = deepcopy(self.training_model_)

        if self.save_last_model:
            self.save_last_model_to_file(file_path=self.file_path)

        self.last_window_ = series_1d[-self.window :]

        return self

    def _predict(self, y=None, exog=None):
        """Satisfy ``BaseForecaster._predict`` by delegating to the mixin.

        Parameters
        ----------
        y : np.ndarray or None, default=None
        exog : ignored

        Returns
        -------
        predictions : np.ndarray of shape (horizon,)
        """
        prediction = self._series_to_series_forecast(
            y=y, prediction_horizon=self.horizon, exog=exog
        )
        if len(prediction) == 1:
            prediction = float(prediction)
        return prediction

    def _series_to_series_forecast(self, y, prediction_horizon, exog=None):
        """Run one forward pass and return the forecast output.

        Parameters
        ----------
        y : np.ndarray or None
            Context series. Uses ``last_window_`` from fit if None.
            Must have at least ``window`` time steps if provided.
        prediction_horizon : int
            Must equal ``self.horizon`` (fixed at construction time).
        exog : ignored

        Returns
        -------
        predictions : np.ndarray of shape (prediction_horizon,)

        Raises
        ------
        ValueError
            If ``prediction_horizon != self.horizon``.
        """
        if prediction_horizon != self.horizon:
            raise ValueError(
                f"prediction_horizon={prediction_horizon} does not match "
                f"horizon={self.horizon}. Re-instantiate with the desired horizon."
            )

        if y is None:
            if not hasattr(self, "last_window_"):
                raise ValueError("No fitted data available for prediction.")
            window_data = self.last_window_
        else:
            y_inner = np.asarray(y)
            if y_inner.ndim == 2:
                if y_inner.shape[1] > 1:
                    raise ValueError(
                        "NBeatsForecaster only supports univariate series."
                    )
                y_inner = y_inner[:, 0]
            if y_inner.shape[0] < self.window:
                raise ValueError(
                    f"Input length ({y_inner.shape[0]}) < window ({self.window})."
                )
            window_data = y_inner[-self.window :]

        x_input = window_data.reshape(1, self.window)
        pred = self.model_.predict(x_input, verbose=0)
        # pred is a dict {"forecast": (1, horizon), "backcast": (1, window)}
        return pred["forecast"][0]

    @classmethod
    def _get_test_params(
        cls, parameter_set: str = "default"
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Return default test parameters."""
        return [
            {
                "window": 10,
                "horizon": 1,
                "n_epochs": 3,
                "batch_size": 4,
                "units": 16,
                "num_blocks_per_stack": 1,
                "stacks": ["generic"],
                "backcast_loss_weight": 0.5,
            }
        ]
