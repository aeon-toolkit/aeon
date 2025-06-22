"""Recurrent Neural Network (RNN) for regression."""

from __future__ import annotations

__maintainer__ = [""]
__all__ = ["RecurrentRegressor"]

import gc
import os
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.utils import check_random_state

from aeon.networks import RecurrentNetwork
from aeon.regression.deep_learning.base import BaseDeepRegressor

if TYPE_CHECKING:
    import tensorflow as tf
    from tensorflow.keras.callbacks import Callback


class RecurrentRegressor(BaseDeepRegressor):
    """
    Recurrent Neural Network (RNN) regressor.

    Adapted from the implementation used in sktime-dl for time series regression.

    Parameters
    ----------
    rnn_type : str, default = "lstm"
        Type of RNN layer to use. Options: "lstm", "gru", "simple_rnn"
    n_layers : int, default = 1
        Number of RNN layers
    n_units : int, default = 64
        Number of units in each RNN layer
    dropout_rate : float, default = 0.2
        Dropout rate for regularization
    bidirectional : bool, default = False
        Whether to use bidirectional RNN layers
    activation : str, default = "tanh"
        Activation function for RNN layers
    return_sequence_last : bool, default = None
        Whether RNN layers should return sequences. If None, automatically determined
    n_epochs : int, default = 100
        Number of epochs to train the model
    batch_size : int, default = 32
        Number of samples per gradient update
    use_mini_batch_size : bool, default = False
        Condition on using the mini batch size formula
    callbacks : keras callback or list of callbacks, default = None
        The default list of callbacks are set to ModelCheckpoint and ReduceLROnPlateau
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    file_path : str, default = './'
        File path when saving model_Checkpoint callback
    save_best_model : bool, default = False
        Whether or not to save the best model
    save_last_model : bool, default = False
        Whether or not to save the last model
    save_init_model : bool, default = False
        Whether to save the initialization of the model
    best_file_name : str, default = "best_model"
        The name of the file of the best model
    last_file_name : str, default = "last_model"
        The name of the file of the last model
    init_file_name : str, default = "init_model"
        The name of the file of the init model
    verbose : bool, default = False
        Whether to output extra information
    loss : str, default = "mean_squared_error"
        The name of the keras training loss
    optimizer : keras.optimizer, default = None
        The keras optimizer used for training. If None, uses Adam with lr=0.001
    metrics : str or list[str], default="mean_squared_error"
        The evaluation metrics to use during training
    output_activation : str, default = "linear"
        The output activation for the regressor

    Examples
    --------
    >>> from aeon.regression.deep_learning import RecurrentRegressor
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, regression_target=True,
    ...                              random_state=0)
    >>> rgs = RecurrentRegressor(n_epochs=20, batch_size=4)  # doctest: +SKIP
    >>> rgs.fit(X, y)  # doctest: +SKIP
    RecurrentRegressor(...)
    """

    def __init__(
        self,
        rnn_type: str = "lstm",
        n_layers: int = 1,
        n_units: int = 64,
        dropout_intermediate: float = 0.2,
        dropout_output: float = 0.2,
        bidirectional: bool = False,
        activation: str = "tanh",
        return_sequence_last: bool | None = None,
        n_epochs: int = 100,
        callbacks: Callback | list[Callback] | None = None,
        verbose: bool = False,
        loss: str = "mean_squared_error",
        output_activation: str = "linear",
        metrics: str | list[str] = "mean_squared_error",
        batch_size: int = 32,
        use_mini_batch_size: bool = False,
        random_state: int | np.random.RandomState | None = None,
        file_path: str = "./",
        save_best_model: bool = False,
        save_last_model: bool = False,
        save_init_model: bool = False,
        best_file_name: str = "best_model",
        last_file_name: str = "last_model",
        init_file_name: str = "init_model",
        optimizer: tf.keras.optimizers.Optimizer | None = None,
    ):
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.n_units = n_units
        self.dropout_intermediate = dropout_intermediate
        self.dropout_output = dropout_output
        self.bidirectional = bidirectional
        self.activation = activation
        self.return_sequence_last = return_sequence_last
        self.n_epochs = n_epochs
        self.callbacks = callbacks
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.use_mini_batch_size = use_mini_batch_size
        self.random_state = random_state
        self.output_activation = output_activation
        self.file_path = file_path
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.init_file_name = init_file_name
        self.optimizer = optimizer
        self.history = None

        super().__init__(batch_size=batch_size, last_file_name=last_file_name)

        self._network = RecurrentNetwork(
            rnn_type=self.rnn_type,
            n_layers=self.n_layers,
            n_units=self.n_units,
            dropout_intermediate=self.dropout_intermediate,
            dropout_output=self.dropout_output,
            bidirectional=self.bidirectional,
            activation=self.activation,
            return_sequence_last=self.return_sequence_last,
        )

    def build_model(
        self, input_shape: tuple[int, ...], **kwargs: Any
    ) -> tf.keras.Model:
        """
        Construct a compiled, un-trained, keras model that is ready for training.

        In aeon, time series are stored in numpy arrays of shape (d,m), where d
        is the number of dimensions, m is the series length. Keras/tensorflow assume
        data is in shape (m,d). This method also assumes (m,d). Transpose should
        happen in fit.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be (m,d)

        Returns
        -------
        output : a compiled Keras Model
        """
        import tensorflow as tf

        self.optimizer_ = (
            tf.keras.optimizers.Adam(learning_rate=0.001)
            if self.optimizer is None
            else self.optimizer
        )

        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)
        tf.keras.utils.set_random_seed(self.random_state_)

        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = tf.keras.layers.Dense(
            units=1,
            activation=self.output_activation,
        )(output_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=self._metrics,
        )

        return model

    def _fit(self, X: np.ndarray, y: np.ndarray) -> RecurrentRegressor:
        """
        Fit the regressor on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray
            The training input samples of shape (n_cases, n_channels, n_timepoints).
        y : np.ndarray
            The training data target values of shape (n_cases,).

        Returns
        -------
        self : object
        """
        import tensorflow as tf

        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        if isinstance(self.metrics, list):
            self._metrics = self.metrics
        elif isinstance(self.metrics, str):
            self._metrics = [self.metrics]

        self.input_shape = X.shape[1:]
        self.training_model_ = self.build_model(self.input_shape)

        if self.save_init_model:
            self.training_model_.save(self.file_path + self.init_file_name + ".keras")

        if self.verbose:
            self.training_model_.summary()

        self.file_name_ = (
            self.best_file_name if self.save_best_model else str(time.time_ns())
        )

        if self.callbacks is None:
            self.callbacks_ = [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=50, min_lr=0.0001
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.file_path + self.file_name_ + ".keras",
                    monitor="loss",
                    save_best_only=True,
                ),
            ]
        else:
            self.callbacks_ = self._get_model_checkpoint_callback(
                callbacks=self.callbacks,
                file_path=self.file_path,
                file_name=self.file_name_,
            )

        if self.use_mini_batch_size:
            mini_batch_size = min(self.batch_size, X.shape[0] // 10)
        else:
            mini_batch_size = self.batch_size

        self.history = self.training_model_.fit(
            X,
            y,
            batch_size=mini_batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks_,
        )

        try:
            self.model_ = tf.keras.models.load_model(
                self.file_path + self.file_name_ + ".keras", compile=False
            )
            if not self.save_best_model:
                os.remove(self.file_path + self.file_name_ + ".keras")
        except FileNotFoundError:
            self.model_ = deepcopy(self.training_model_)

        if self.save_last_model:
            self.save_last_model_to_file(file_path=self.file_path)

        gc.collect()
        return self

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
            special parameters are defined for a value, will return `"default"` set.
            For regressors, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        param = {
            "n_epochs": 10,
            "batch_size": 4,
            "n_layers": 1,
            "n_units": 6,
            "rnn_type": "lstm",
        }
        return [param]
