"""Specialized layers for N-BEATS model."""

import numpy as np

from aeon.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies(["tensorflow"], severity="none"):
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="aeon")
    class NbeatsSeasonalityLayer(tf.keras.layers.Layer):
        """Harmonic functions to model seasonality."""

        def __init__(self, harmonics: int, lookback: int, horizon: int, **kwargs):
            super().__init__(**kwargs)

            self.harmonics = harmonics
            self.lookback = lookback
            self.horizon = horizon

            self.theta_size = 4 * int(
                np.ceil(harmonics / 2 * horizon) - (harmonics - 1)
            )
            self.theta_layer = tf.keras.layers.Dense(self.theta_size)

            zero_freq = np.zeros(1, dtype=np.float32)
            other_freq = np.arange(harmonics, harmonics / 2 * horizon, dtype=np.float32)
            self.freq = np.concatenate([zero_freq, other_freq / harmonics], axis=0)

            time = np.arange(lookback, dtype=np.float32)[:, None] / horizon
            backcast_grid = -2 * np.pi * time * self.freq
            time = np.arange(horizon, dtype=np.float32)[:, None] / horizon
            forecast_grid = 2 * np.pi * time * self.freq

            self.b_proj_cos = tf.constant(np.cos(backcast_grid).T, dtype=tf.float32)
            self.b_proj_sin = tf.constant(np.sin(backcast_grid).T, dtype=tf.float32)
            self.f_proj_cos = tf.constant(np.cos(forecast_grid).T, dtype=tf.float32)
            self.f_proj_sin = tf.constant(np.sin(forecast_grid).T, dtype=tf.float32)

        def call(self, h):
            """Forward pass for the seasonality layer."""
            theta = self.theta_layer(h)
            s = self.theta_size // 4  # size per coefficients
            theta_bounds = [
                theta[..., 0 * s : 1 * s],  # Cosine coefficients for backcast
                theta[..., 1 * s : 2 * s],  # Sine coefficients for backcast
                theta[..., 2 * s : 3 * s],  # Cosine coefficients for forecast
                theta[..., 3 * s : 4 * s],  # Sine coefficients for forecast
            ]

            b_proj_cos = tf.matmul(theta_bounds[0], self.b_proj_cos)
            b_proj_sin = tf.matmul(theta_bounds[1], self.b_proj_sin)
            backcast = b_proj_cos + b_proj_sin

            f_proj_cos = tf.matmul(theta_bounds[2], self.f_proj_cos)
            f_proj_sin = tf.matmul(theta_bounds[3], self.f_proj_sin)
            forecast = f_proj_cos + f_proj_sin

            return backcast, forecast

        def get_config(self):
            """Get the configuration of the layer for serialization."""
            config = super().get_config()
            config.update(
                {
                    "harmonics": self.harmonics,
                    "lookback": self.lookback,
                    "horizon": self.horizon,
                }
            )
            return config

    @tf.keras.utils.register_keras_serializable(package="aeon")
    class NbeatsTrendLayer(tf.keras.layers.Layer):
        """Polynomial function to model trend."""

        def __init__(self, degree: int, lookback: int, horizon: int, **kwargs):
            super().__init__(**kwargs)

            self.degree = degree
            self.lookback = lookback
            self.horizon = horizon

            self.size = degree + 1  # +1 for the constant term

            self.theta_layer = tf.keras.layers.Dense(2 * self.size)

            t = np.arange(lookback, dtype=np.float32) / lookback
            forecast_time_np = [np.power(t, i)[None, :] for i in range(self.size)]
            self.backcast_time = tf.constant(
                np.concatenate(forecast_time_np, axis=0), dtype=tf.float32
            )

            t = np.arange(horizon, dtype=np.float32) / horizon
            forecast_time_np = [np.power(t, i)[None, :] for i in range(self.size)]
            self.forecast_time = tf.constant(
                np.concatenate(forecast_time_np, axis=0), dtype=tf.float32
            )

        def call(self, h):
            """Forward pass for the trend layer."""
            theta = self.theta_layer(h)
            backcast = tf.matmul(theta[:, : self.size], self.backcast_time)
            forecast = tf.matmul(theta[:, self.size :], self.forecast_time)
            return backcast, forecast

        def get_config(self):
            """Get the configuration of the layer for serialization."""
            config = super().get_config()
            config.update(
                {
                    "degree": self.degree,
                    "lookback": self.lookback,
                    "horizon": self.horizon,
                }
            )
            return config

    @tf.keras.utils.register_keras_serializable(package="aeon")
    class NbeatsGenericLayer(tf.keras.layers.Layer):
        """Generic layer for N-BEATS."""

        def __init__(self, lookback: int, horizon: int, **kwargs):
            super().__init__(**kwargs)

            self.lookback = lookback
            self.horizon = horizon

            self.lookback_layer = tf.keras.layers.Dense(lookback)
            self.forecast_layer = tf.keras.layers.Dense(horizon)

        def call(self, h):
            """Forward pass for the generic layer."""
            lookback = self.lookback_layer(h)
            forecast = self.forecast_layer(h)
            return lookback, forecast

        def get_config(self):
            """Get the configuration of the layer for serialization."""
            config = super().get_config()
            config.update(
                {
                    "lookback": self.lookback,
                    "horizon": self.horizon,
                }
            )
            return config
