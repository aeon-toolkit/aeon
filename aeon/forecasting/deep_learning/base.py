"""Base class module for deep learning forecasters in aeon.

This module defines the `BaseDeepForecaster` class, an abstract base class for
deep learning-based forecasting models within the aeon toolkit.
"""

from __future__ import annotations

__maintainer__ = []
__all__ = ["BaseDeepForecaster"]

from abc import abstractmethod

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from aeon.forecasting.base import BaseForecaster


class BaseDeepForecaster(BaseForecaster):
    """Base class for deep learning forecasters in aeon.

    This class provides a foundation for deep learning-based forecasting models,
    handling data preprocessing, model training, and prediction with enhanced
    capabilities for callbacks, model saving/loading, and efficiency.

    Parameters
    ----------
    horizon : int, default=1
        Forecasting horizon, the number of steps ahead to predict.
    window : int, default=10
        The window size for creating input sequences.
    batch_size : int, default=32
        Batch size for training the model.
    n_epochs : int, default=100
        Number of epochs to train the model.
    verbose : int, default=0
        Verbosity mode (0, 1, or 2).
    optimizer : str or tf.keras.optimizers.Optimizer, default='adam'
        Optimizer to use for training.
    loss : str or tf.keras.losses.Loss, default='mse'
        Loss function for training.
    callbacks : list of tf.keras.callbacks.Callback or None, default=None
        List of Keras callbacks to be applied during training.
    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.
    axis : int, default=0
        Axis along which to apply the forecaster.
    last_file_name : str, default="last_model"
        The name of the file of the last model, used for saving models.
    save_best_model : bool, default=False
        Whether to save the best model during training based on validation loss.
    file_path : str, default="./"
        Directory path where models will be saved.

    Attributes
    ----------
    model_ : tf.keras.Model or None
        The fitted Keras model.
    history_ : tf.keras.callbacks.History or None
        Training history containing loss and metrics.
    last_window_ : np.ndarray or None
        The last window of data used for prediction.
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
        horizon=1,
        window=10,
        batch_size=32, # remove
        n_epochs=100, # remove 
        verbose=0,
        optimizer="adam", # remove it 
        loss="mse", # remove it 
        callbacks=None,
        random_state=None, # remove it 
        axis=0,
        last_file_name="last_model",
        save_best_model=False,
        file_path="./",
    ):
        self.horizon = horizon
        self.window = window
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.optimizer = optimizer
        self.loss = loss
        self.callbacks = callbacks
        self.random_state = random_state
        self.axis = axis
        self.last_file_name = last_file_name
        self.save_best_model = save_best_model
        self.file_path = file_path
        
        # Initialize attributes
        self.model_ = None
        self.history_ = None
        self.last_window_ = None

        # Pass horizon and axis to BaseForecaster
        super().__init__(horizon=horizon, axis=axis)

    def _fit(self, y, X=None): # remove it 
        """Fit the forecaster to training data.

        Parameters
        ----------
        y : np.ndarray or pd.Series
            Target time series to which to fit the forecaster.
        X : np.ndarray or pd.DataFrame, default=None
            Exogenous variables.

        Returns
        -------
        self : BaseDeepForecaster
            Returns an instance of self.
        """
        import tensorflow as tf

        # Set random seed for reproducibility
        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)
        tf.keras.utils.set_random_seed(self.random_state_)

        # Convert input data to numpy array
        y_inner = self._convert_input(y)

        if y_inner.ndim == 1:
            y_inner = y_inner.reshape(-1, 1)  # Convert univariate to (timepoints, 1)

        if y_inner.shape[0] < self.window + self.horizon:
            raise ValueError(
                f"Data length ({y_inner.shape[0]}) is insufficient for window "
                f"({self.window}) and horizon ({self.horizon})."
            )

        # Create sequences for training
        X_train, y_train = self._create_sequences(y_inner)

        if X_train.shape[0] == 0:
            raise ValueError("No training sequences could be created.")

        # Build and compile the model
        input_shape = X_train.shape[1:]
        self.model_ = self.build_model(input_shape)
        self.model_.compile(optimizer=self.optimizer, loss=self.loss)

        # Prepare callbacks
        callbacks_list = self._prepare_callbacks()

        # Train the model
        self.history_ = self.model_.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=callbacks_list,
        )

        # Save the last window for prediction
        self.last_window_ = y_inner[-self.window:]

        return self

    def _predict(self, y=None, X=None): # remove it
        """Make forecasts for y.

        Parameters
        ----------
        y : np.ndarray or pd.Series, default=None
            Series to predict from. If None, uses last fitted window.
        X : np.ndarray or pd.DataFrame, default=None
            Exogenous variables (not supported by default).

        Returns
        -------
        predictions : np.ndarray
            Predicted values for the specified horizon. Shape: (horizon, channels) for multivariate
            data or (horizon,) for univariate data.
        """
        if y is None:
            if not hasattr(self, "last_window_"):
                raise ValueError("No fitted data available for prediction.")
            y_inner = self.last_window_
        else:
            y_inner = self._convert_input(y)
            if y_inner.ndim == 1:
                y_inner = y_inner.reshape(-1, 1)  # Convert univariate to (timepoints, 1)
            if y_inner.shape[0] < self.window:
                raise ValueError(
                    f"Input data length ({y_inner.shape[0]}) is less than the window size "
                    f"({self.window})."
                )
            y_inner = y_inner[-self.window:]
           
        # Get the number of channels from the input data
        num_channels = y_inner.shape[-1]
        last_window = y_inner.reshape(1, self.window, num_channels)
        predictions = []
        current_window = last_window

        for _ in range(self.horizon):
            pred = self.model_.predict(current_window, verbose=0)
            predictions.append(pred)  
            current_window = np.roll(current_window, -1, axis=1)
            current_window[0, -1, :] = pred[0, :]  # Update all channels

        predictions = np.array(predictions)  
        predictions = np.squeeze(predictions, axis=1) # Shape: (horizon, channels)
        if num_channels == 1:
            predictions = predictions.flatten()  # Convert to (horizon,) for univariate
        
        return predictions

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

        return y_inner

    def _create_sequences(self, data):
        """Create input sequences and target values for training.

        Parameters
        ----------
        data : np.ndarray
            Time series data. Assumes shape (timepoints, channels) for multivariate
            data or (timepoints,) for univariate.

        Returns
        -------
        X : np.ndarray
            Input sequences. Shape: (num_sequences, window, channels) for multivariate
            or (num_sequences, window, 1) for univariate.
        y : np.ndarray
            Target values. Shape: (num_sequences, horizon, channels) for multivariate
            or (num_sequences, horizon) for univariate (reshaped to (num_sequences, horizon, 1) if needed).
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)  # Convert univariate to (timepoints, 1)

        num_timepoints, num_channels = data.shape

        if num_timepoints < self.window + self.horizon:
            raise ValueError(
                f"Data length ({num_timepoints}) is insufficient for window "
                f"({self.window}) and horizon ({self.horizon})."
            )

        X, y = [], []
        for i in range(num_timepoints - self.window - self.horizon + 1):
            X.append(data[i : (i + self.window)])
            y.append(data[i + self.window : (i + self.window + self.horizon)])

        X = np.array(X)  # Shape: (num_sequences, window, channels)
        y = np.array(y)  # Shape: (num_sequences, horizon, channels)

        return X, y

    def _prepare_callbacks(self):
        """Prepare callbacks for training.

        Returns
        -------
        callbacks_list : list
            List of callbacks to be used during training.
        """
        callbacks_list = []

        # Add user-provided callbacks
        if self.callbacks is not None:
            if isinstance(self.callbacks, list):
                callbacks_list.extend(self.callbacks)
            else:
                callbacks_list.append(self.callbacks)

        # Add model checkpoint callback if save_best_model is True
        if self.save_best_model:
            callbacks_list = self._get_model_checkpoint_callback(
                callbacks_list, self.file_path, "best_model"
            )

        return callbacks_list

    def _get_model_checkpoint_callback(self, callbacks, file_path, file_name):
        """Add model checkpoint callback to save the best model.

        Parameters
        ----------
        callbacks : list
            Existing list of callbacks.
        file_path : str
            Directory path where the model will be saved.
        file_name : str
            Name of the model file.

        Returns
        -------
        callbacks : list
            Updated list of callbacks including ModelCheckpoint.
        """
        import tensorflow as tf

        model_checkpoint_ = tf.keras.callbacks.ModelCheckpoint(
            filepath=file_path + file_name + ".keras",
            monitor="loss",
            save_best_only=True,
            verbose=self.verbose,
        )

        if isinstance(callbacks, list):
            return callbacks + [model_checkpoint_]
        else:
            return [callbacks] + [model_checkpoint_]

    def summary(self):
        """Summary function to return the losses/metrics for model fit.

        Returns
        -------
        history : dict or None
            Dictionary containing model's train/validation losses and metrics.
        """
        return self.history_.history if self.history_ is not None else None

    def save_last_model_to_file(self, file_path="./"):
        """Save the last epoch of the trained deep learning model.

        Parameters
        ----------
        file_path : str, default="./"
            The directory where the model will be saved.

        Returns
        -------
        None
        """
        if self.model_ is None:
            raise ValueError("No model to save. Please fit the model first.")
        
        self.model_.save(file_path + self.last_file_name + ".keras")

    def load_model(self, model_path):
        """Load a pre-trained keras model instead of fitting.

        When calling this function, all functionalities can be used
        such as predict with the loaded model.

        Parameters
        ----------
        model_path : str
            Path to the saved model file including extension.
            Example: model_path="path/to/file/best_model.keras"

        Returns
        -------
        None
        """
        import tensorflow as tf

        self.model_ = tf.keras.models.load_model(model_path)
        self.is_fitted = True

    @abstractmethod
    def build_model(self, input_shape):
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
