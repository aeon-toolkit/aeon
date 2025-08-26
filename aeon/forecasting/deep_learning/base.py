"""Base class module for deep learning forecasters in aeon.

This module defines the `BaseDeepForecaster` class, an abstract base class for
deep learning-based forecasting models within the aeon toolkit.
"""

from __future__ import annotations

__maintainer__ = []
__all__ = ["BaseDeepForecaster"]

from abc import abstractmethod
from typing import Any

from aeon.forecasting.base import BaseForecaster


class BaseDeepForecaster(BaseForecaster):
    """Base class for deep learning forecasters in aeon.

    This class provides a foundation for deep learning-based forecasting models,
    handling data preprocessing, model training, and prediction with enhanced
    capabilities for callbacks, model saving/loading, and efficiency.

    Parameters
    ----------
    window : int,
        The window size for creating input sequences.
    horizon : int, default=1
        Forecasting horizon, the number of steps ahead to predict.
    verbose : int, default=0
        Verbosity mode (0, 1, or 2).
    callbacks : list of tf.keras.callbacks.Callback or None, default=None
        List of Keras callbacks to be applied during training.
    axis : int, default=0
        Axis along which to apply the forecaster.
    last_file_name : str, default="last_model"
        The name of the file of the last model, used for saving models.
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
        "capability:multivariate": False,
    }

    def __init__(
        self,
        window,
        horizon=1,
        verbose=0,
        callbacks=None,
        axis=0,
        last_file_name="last_model",
        file_path="./",
    ):
        self.horizon = horizon
        self.window = window
        self.verbose = verbose
        self.callbacks = callbacks
        self.axis = axis
        self.last_file_name = last_file_name
        self.file_path = file_path

        self.model_ = None
        self.history_ = None
        self.last_window_ = None

        super().__init__(horizon=horizon, axis=axis)

    def _prepare_callbacks(self):
        """Prepare callbacks for training.

        Returns
        -------
        callbacks_list : list
            List of callbacks to be used during training.
        """
        callbacks_list = []
        if self.callbacks is not None:
            if isinstance(self.callbacks, list):
                callbacks_list.extend(self.callbacks)
            else:
                callbacks_list.append(self.callbacks)

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
        import os

        if self.model_ is None:
            raise ValueError("No model to save. Please fit the model first.")
        self.model_.save(os.path.join(file_path, self.last_file_name + ".keras"))

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

    @classmethod
    def _get_test_params(
        cls, parameter_set: str = "default"
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
        """
        param = {
            "window": 10,
        }
        return [param]
