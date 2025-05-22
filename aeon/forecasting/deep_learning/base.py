"""
BaseDeepForecaster class.

A simplified first base class for deep learning forecasting models.
This class is a subclass of BaseForecaster and inherits its methods and attributes.
It provides a base for deep learning models, including methods for training and
predicting.

"""

from abc import abstractmethod

import numpy as np
import pandas as pd
import tensorflow as tf

from aeon.forecasting.base import BaseForecaster


class BaseDeepForecaster(BaseForecaster):
    """Base class for deep learning forecasters in aeon.

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
    ):
        self.horizon = horizon
        self.window = window
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.optimizer = optimizer
        self.loss = loss
        self.random_state = random_state
        self.model_ = None
        super().__init__()

    def _fit(self, y, X=None):
        """Fit the forecaster to training data.

        Parameters
        ----------
        y : np.ndarray or pd.Series
            Target time series to which to fit the forecaster.
        X : np.ndarray or pd.DataFrame, default=None
            Exogenous variables.

        Returns
        -------
        self : returns an instance of self
        """
        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)
            tf.random.set_seed(self.random_state)

        # Convert input data to numpy array
        y_inner = self._convert_input(y)

        # Create sequences for training
        X_train, y_train = self._create_sequences(y_inner)

        # Build and compile the model
        self.model_ = self._build_model(X_train.shape[1:])
        self.model_.compile(optimizer=self.optimizer, loss=self.loss)

        # Train the model
        self.model_.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
        )

        return self

    def _predict(self, y=None, X=None):
        """Make forecasts for y.

        Parameters
        ----------
        y : np.ndarray or pd.Series, default=None
            Series to predict from.
        X : np.ndarray or pd.DataFrame, default=None
            Exogenous variables.

        Returns
        -------
        predictions : np.ndarray
            Predicted values.
        """
        if y is None:
            raise ValueError("y cannot be None for prediction")

        # Convert input data to numpy array
        y_inner = self._convert_input(y)

        # Use the last window of data for prediction
        last_window = y_inner[-self.window :].reshape(1, self.window, 1)

        # Make prediction
        prediction = self.model_.predict(last_window, verbose=0)

        return prediction.flatten()

    def _forecast(self, y, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        y : np.ndarray or pd.Series
            Time series to forecast from.
        X : np.ndarray or pd.DataFrame, default=None
            Exogenous variables.

        Returns
        -------
        forecasts : np.ndarray
            Forecasted values.
        """
        # Fit the model
        self._fit(y, X)

        # Make prediction
        return self._predict(y, X)

    def _convert_input(self, y):
        """Convert input data to numpy array.

        Parameters
        ----------
        y : np.ndarray or pd.Series
            Input time series.

        Returns
        -------
        y_inner : np.ndarray
            Converted numpy array.
        """
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y_inner = y.values
        else:
            y_inner = y

        # Ensure 1D array
        if len(y_inner.shape) > 1:
            y_inner = y_inner.flatten()

        return y_inner

    def _create_sequences(self, data):
        """Create input sequences and target values for training.

        Parameters
        ----------
        data : np.ndarray
            Time series data.

        Returns
        -------
        X : np.ndarray
            Input sequences.
        y : np.ndarray
            Target values.
        """
        X, y = [], []
        for i in range(len(data) - self.window - self.horizon + 1):
            X.append(data[i : (i + self.window)])
            y.append(data[i + self.window : (i + self.window + self.horizon)])

        return np.array(X).reshape(-1, self.window, 1), np.array(y)

    @abstractmethod
    def _build_model(self, input_shape):
        """Build the deep learning model.

        Parameters
        ----------
        input_shape : tuple
            Shape of input data.

        Returns
        -------
        model : tf.keras.Model
            Compiled Keras model.
        """
        pass
