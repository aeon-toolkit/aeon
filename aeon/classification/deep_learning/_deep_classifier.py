"""Generic deep learning classifier for time series."""

__maintainer__ = ["melpiro"]
__all__ = ["DeepClassifier"]

import gc
import os
import time
from copy import deepcopy

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier


class DeepClassifier(BaseClassifier):
    """Generic deep learning classifier using an aeon Network.

    Parameters
    ----------
    network : BaseDeepNetwork
        Network used to extract features from the input time series.
        The network must implement ``build_network(input_shape)``.
    n_epochs : int, default=32
        Number of training epochs.
    batch_size : int, default=64
        Training batch size.
    loss : str, default="categorical_crossentropy"
        Loss function used for training.
    optimizer : str or keras.optimizers.Optimizer, default=None
        Optimizer used for training. Defaults to Adam.
    metrics : str or list of str, default="accuracy"
        Metrics used during training.
    output_activation : str, default="softmax"
        Activation function of the classification output layer.
    verbose : bool or int, default=False
        Whether to display training information.
    random_state : int, RandomState instance or None, default=None
        Random seed for reproducibility.
    ensemble_size : int, default=1
        Number of networks to train. Predictions are averaged across networks.
    callbacks : list or keras.callbacks.Callback, default=None
        Additional training callbacks.
    use_mini_batch_size : bool, default=False
        Whether to adapt the batch size to the number of training cases.
    file_path : str, default="./"
        Directory used to save models.
    save_best_model : bool, default=False
        Whether to save the best model during training.
    save_last_model : bool, default=False
        Whether to save the final trained model.
    save_init_model : bool, default=False
        Whether to save the model before training.
    best_file_name : str, default="best_model"
        Base filename used for the best model.
    last_file_name : str, default="last_model"
        Base filename used for the final model.
    init_file_name : str, default="init_model"
        Base filename used for the initial model.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "cant_pickle": True,
        "python_dependencies": "tensorflow",
    }

    def __init__(
        self,
        network,
        n_epochs=32,
        batch_size=64,
        loss="categorical_crossentropy",
        optimizer=None,
        metrics="accuracy",
        output_activation="softmax",
        verbose=False,
        random_state=None,
        ensemble_size=1,
        callbacks=None,
        use_mini_batch_size=False,
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        save_init_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        init_file_name="init_model",
    ):
        self.network = network
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.output_activation = output_activation
        self.verbose = verbose
        self.random_state = random_state
        self.ensemble_size = ensemble_size
        self.callbacks = callbacks
        self.use_mini_batch_size = use_mini_batch_size
        self.file_path = file_path
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.last_file_name = last_file_name
        self.init_file_name = init_file_name

        super().__init__()

    def build_model(self, input_shape, n_classes):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In aeon, time series are stored in numpy arrays of shape (d,m), where d
        is the number of channels, m is the series length. Keras/tensorflow assume
        data is in shape (m,d). This method also assumes (m,d). Transpose should
        happen in fit.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be (m, d).
        n_classes : int
            The number of classes, which becomes the size of the output layer.

        Returns
        -------
        output : a compiled Keras Model
        """
        import tensorflow as tf

        input_layer, output_layer = self.network_.build_network(input_shape)

        output_layer = tf.keras.layers.Dense(
            units=n_classes,
            activation=self.output_activation,
        )(output_layer)

        model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=output_layer,
        )

        optimizer = (
            tf.keras.optimizers.Adam()
            if self.optimizer is None
            else deepcopy(self.optimizer)
        )

        metrics = (
            [self.metrics] if isinstance(self.metrics, str) else list(self.metrics)
        )

        model.compile(
            loss=self.loss,
            optimizer=optimizer,
            metrics=metrics,
        )

        return model

    def _get_callbacks(self, file_name):
        import tensorflow as tf

        callbacks = deepcopy(self.callbacks) if self.callbacks is not None else []

        # Always add checkpointing internally if required for best-model selection.
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self._get_model_path(file_name),
                monitor="loss",
                save_best_only=True,
            )
        )

        return callbacks

    def _get_file_name(self, base_name, index):
        """Get a filename for a network."""
        if self.ensemble_size == 1:
            return base_name

        return f"{base_name}{index}"

    def _get_model_path(self, file_name):
        """Get the full path of a model."""
        return os.path.join(self.file_path, file_name + ".keras")

    def _fit(self, X, y):
        """Fit the classifier."""
        import tensorflow as tf

        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

        self.label_encoder_ = encoder
        self.classes_ = encoder.classes_
        self.n_classes_ = len(self.classes_)

        y_onehot = tf.keras.utils.to_categorical(
            y,
            num_classes=self.n_classes_,
        )

        # aeon uses (n_cases, n_channels, n_timepoints).
        # Networks expect (n_cases, n_timepoints, n_channels).
        X = X.transpose(0, 2, 1)

        self.input_shape_ = X.shape[1:]

        if self.use_mini_batch_size:
            batch_size = min(self.batch_size, max(1, X.shape[0] // 10))
        else:
            batch_size = self.batch_size

        rng = check_random_state(self.random_state)

        self.models_ = []
        self.histories_ = []
        self.training_models_ = []

        for i in range(self.ensemble_size):
            # Each network receives an independent random initialization.
            tf.keras.backend.clear_session()

            random_state = rng.randint(0, np.iinfo(np.int32).max)
            tf.keras.utils.set_random_seed(random_state)

            # Avoid sharing mutable network state between ensemble members.
            self.network_ = deepcopy(self.network)

            self.training_model_ = self.build_model(
                self.input_shape_,
                self.n_classes_,
            )

            if self.save_init_model:
                init_name = self._get_file_name(self.init_file_name, i)
                self.training_model_.save(self._get_model_path(init_name))

            if self.verbose:
                self.training_model_.summary()

            file_name = (
                self._get_file_name(self.best_file_name, i)
                if self.save_best_model
                else f"model_{time.time_ns()}_{i}"
            )

            callbacks = self._get_callbacks(file_name)

            history = self.training_model_.fit(
                X,
                y_onehot,
                batch_size=batch_size,
                epochs=self.n_epochs,
                verbose=self.verbose,
                callbacks=callbacks,
            )

            model_path = self._get_model_path(file_name)

            try:
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False,
                )

                if not self.save_best_model:
                    os.remove(model_path)

            except FileNotFoundError:
                # No checkpoint was produced, e.g. if the checkpoint
                # callback was not able to save the model.
                model = deepcopy(self.training_model_)

            self.models_.append(model)
            self.histories_.append(history)
            self.training_models_.append(self.training_model_)

            if self.save_last_model:
                last_name = self._get_file_name(self.last_file_name, i)
                model.save(self._get_model_path(last_name))

            gc.collect()

        return self

    def _predict_proba(self, X):
        """Predict class probabilities."""
        X = X.transpose(0, 2, 1)

        probabilities = []

        for model in self.models_:
            probs = model.predict(
                X,
                batch_size=self.batch_size,
                verbose=0,
            )

            if probs.shape[1] == 1:
                probs = np.hstack([1 - probs, probs])

            # Normalize probabilities, including sigmoid outputs.
            probs = probs / probs.sum(axis=1, keepdims=True)

            probabilities.append(probs)

        return np.mean(probabilities, axis=0)

    def _predict(self, X):
        """Predict class labels."""
        probabilities = self._predict_proba(X)
        rng = check_random_state(self.random_state)

        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in probabilities
            ]
        )

    def summary(self):
        """Return the training history."""
        if not hasattr(self, "histories_"):
            return None

        if self.ensemble_size == 1:
            return self.history.history

        return [history.history for history in self.histories_]

    def save_last_model_to_file(self, file_path=None):
        """Save the trained model(s)."""
        if file_path is None:
            file_path = self.file_path

        for i, model in enumerate(self.models_):
            file_name = self._get_file_name(self.last_file_name, i)
            model.save(os.path.join(file_path, file_name + ".keras"))

    @classmethod
    def load_model(cls, model_path, classes):
        """Load a previously trained model or ensemble."""
        import tensorflow as tf

        if isinstance(model_path, (str, os.PathLike)):
            model_path = [model_path]

        model = cls()
        model.models_ = [
            tf.keras.models.load_model(path, compile=False) for path in model_path
        ]

        model.ensemble_size = len(model.models_)
        model.model_ = model.models_[0]
        model.classes_ = np.asarray(classes)
        model.n_classes_ = len(model.classes_)
        model.is_fitted = True

        return model

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        from aeon.networks import FCNNetwork

        return [
            {
                "network": FCNNetwork(
                    n_layers=1,
                    n_filters=4,
                    kernel_size=3,
                ),
                "n_epochs": 2,
                "batch_size": 4,
            },
            {
                "network": FCNNetwork(
                    n_layers=1,
                    n_filters=4,
                    kernel_size=3,
                ),
                "n_epochs": 2,
                "batch_size": 4,
                "use_mini_batch_size": True,
                "verbose": True,
                "metrics": ["accuracy"],
                "ensemble_size": 2,
            },
        ]
