"""Time Transformer Classifier."""

__maintainer__ = []
__all__ = ["TimeTransformerClassifier"]

import gc
import os
from copy import deepcopy

from sklearn.utils import check_random_state

from aeon.classification.deep_learning.base import BaseDeepClassifier
from aeon.networks import TransformerNetwork


class TimeTransformerClassifier(BaseDeepClassifier):
    """Time Series Transformer Classifier.

    Parameters
    ----------
    n_layers : int, default = 2
        The number of transformer blocks.
    n_heads : int, default = 4
        The number of heads in the multi-head attention layer.
    d_model : int, default = 64
        The dimension of the embedding vector.
    d_inner : int, default = 128
        The dimension of the feed-forward network in the transformer block.
    activation : str, default = "relu"
        The activation function used in the feed-forward network.
    dropout : float, default = 0.1
        The dropout rate.
    n_epochs : int, default = 2000
        The number of epochs to train the model.
    batch_size : int, default = 64
        The number of samples per gradient update.
    callbacks : list of tf.keras.callbacks.Callback, default = None
        List of callbacks to apply during training.
    verbose : boolean, default = False
        Whether to output extra information.
    loss : str, default = "categorical_crossentropy"
        The name of the keras training loss.
    metrics : str or list[str], default = "accuracy"
        The evaluation metrics to use during training.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    optimizer : keras.optimizer, default = None
        The keras optimizer used for training. If None, Adam is used.
    file_path : str, default = "./"
        File path when saving model_Checkpoint callback.
    save_best_model : bool, default = False
        Whether or not to save the best model, if the
        modelcheckpoint callback is used by default,
        this condition, if True, will prevent the
        automatic deletion of the best saved model from
        file and the user can choose the file name.
    save_last_model : bool, default = False
        Whether or not to save the last model, last
        epoch trained, using the base class method
        save_last_model_to_file.
    best_file_name : str, default = "best_model"
        The name of the file of the best model, if
        save_best_model is set to False, this parameter
        is discarded.
    last_file_name : str, default = "last_model"
        The name of the file of the last model, if
        save_last_model is set to False, this parameter
        is discarded.
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
    }

    def __init__(
        self,
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_inner=128,
        activation="relu",
        dropout=0.1,
        n_epochs=2000,
        batch_size=64,
        callbacks=None,
        verbose=False,
        loss="categorical_crossentropy",
        metrics="accuracy",
        random_state=None,
        optimizer=None,
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_inner = d_inner
        self.activation = activation
        self.dropout = dropout

        self.n_epochs = n_epochs
        self.callbacks = callbacks
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.file_path = file_path
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.best_file_name = best_file_name
        self.last_file_name = last_file_name

        self.history = None

        super().__init__(batch_size=batch_size, random_state=random_state)

        self._network = TransformerNetwork(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_model=self.d_model,
            d_inner=self.d_inner,
            activation=self.activation,
            dropout=self.dropout,
        )

    def build_model(self, input_shape, n_classes, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        n_classes : int
            The number of classes, which shall become the size of the output
            layer

        Returns
        -------
        output : a compiled Keras Model
        """
        import tensorflow as tf

        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = tf.keras.layers.Dense(n_classes, activation="softmax")(
            output_layer
        )

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        self.optimizer_ = (
            tf.keras.optimizers.Adam(learning_rate=0.001)
            if self.optimizer is None
            else self.optimizer
        )

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=self._metrics,
        )

        return model

    def _fit(self, X, y):
        """Fit the classifier on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray
            The training input samples of shape (n_cases, n_channels, n_timepoints)
        y : np.ndarray
            The training data class labels of shape (n_cases,).

        Returns
        -------
        self : object
        """
        import tensorflow as tf

        y_onehot = self.convert_y_to_keras(y)
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        check_random_state(self.random_state)
        if isinstance(self.metrics, list):
            self._metrics = self.metrics
        elif isinstance(self.metrics, str):
            self._metrics = [self.metrics]

        self.input_shape = X.shape[1:]

        self.training_model_ = self.build_model(self.input_shape, self.n_classes_)

        if self.verbose:
            self.training_model_.summary()

        if self.callbacks is None:
            self.callbacks_ = [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=50, min_lr=0.0001
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.file_path + self.best_file_name + ".keras",
                    monitor="loss",
                    save_best_only=True,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss", patience=100, restore_best_weights=True
                ),
            ]
        else:
            self.callbacks_ = self.callbacks

        self.history = self.training_model_.fit(
            X,
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks_,
        )

        try:
            self.model_ = tf.keras.models.load_model(
                self.file_path + self.best_file_name + ".keras", compile=False
            )
            if not self.save_best_model:
                os.remove(self.file_path + self.best_file_name + ".keras")
        except FileNotFoundError:
            self.model_ = deepcopy(self.training_model_)

        if self.save_last_model:
            self.save_last_model_to_file(file_path=self.file_path)

        gc.collect()
        return self

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
        """
        params = {
            "n_epochs": 2,
            "batch_size": 4,
            "n_layers": 1,
            "n_heads": 2,
            "d_model": 16,
            "d_inner": 32,
        }
        return params
