"""Base class for deep clustering."""

__maintainer__ = []

import gc
import os
import sys
import time
from abc import abstractmethod
from copy import deepcopy

import numpy as np
from sklearn.utils import check_random_state

from aeon.base._base import _clone_estimator
from aeon.clustering._k_means import TimeSeriesKMeans
from aeon.clustering.base import BaseClusterer


class BaseDeepClusterer(BaseClusterer):
    """Abstract base class for deep learning time series clusterers.

    Parameters
    ----------
    estimator : aeon clusterer, default=None
        An aeon estimator to be built using the transformed data.
        Defaults to aeon TimeSeriesKMeans() with euclidean distance
        and mean averaging method and n_clusters set to 2.
    batch_size : int, default = 32
        training batch size for the model
    last_file_name : str, default = "last_model"
        The name of the file of the last model, used
        only if save_last_model_to_file is used in
        child class.

    Attributes
    ----------
    estimator_ : aeon clusterer
        The fitted clustering estimator used to assign cluster labels
        from the model's latent space representation.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "cant_pickle": True,
        "python_dependencies": "tensorflow",
    }

    @abstractmethod
    def __init__(
        self,
        estimator=None,
        n_epochs=2000,
        batch_size=32,
        validation_split=0,
        use_mini_batch_size=False,
        random_state=None,
        verbose=False,
        loss="mse",
        metrics=None,
        optimizer=None,
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        save_init_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        init_file_name="init_model",
        callbacks=None,
    ):
        super().__init__()

        self.estimator = estimator
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.use_mini_batch_size = use_mini_batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.file_path = file_path
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.last_file_name = last_file_name
        self.init_file_name = init_file_name
        self.callbacks = callbacks
        self._model = None
        self._network = None
        self.can_multi_rec = False

    def _check_params(self):
        import tensorflow as tf

        # metrics
        if self.metrics is None:
            self._metrics = []
        elif isinstance(self.metrics, str):
            self._metrics = [self.metrics]
        else:
            self._metrics = self.metrics

        # optimizer
        self._optimizer = self.optimizer
        if self._optimizer is None:
            self._optimizer = tf.keras.optimizers.Adam()
        elif isinstance(self._optimizer, str):
            self._optimizer = tf.keras.optimizers.get(self._optimizer)

        # random state
        rng = check_random_state(self.random_state)
        self._random_state = rng.randint(0, np.iinfo(np.int32).max)

        # file name
        self._file_name = (
            self.best_file_name if self.save_best_model else str(time.time_ns())
        )

        # callbacks
        self._callbacks = self.callbacks
        if self._callbacks is None:
            self._callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=50, min_lr=0.0001
                )
            ]
        self._callbacks = self._add_model_checkpoint_callback(
            callbacks=self._callbacks,
            file_path=self.file_path,
            file_name=self._file_name,
        )

        # validation split
        self._validation_split = self.validation_split
        if self.validation_split is None:
            self._validation_split = 0

        # check can_multi_rec
        if self.loss == "multi_rec" and not self.can_multi_rec:
            raise ValueError(
                "The loss function 'multi_rec' is not supported for this model."
            )

    def build_model(self, input_shape, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In aeon, time series are stored in numpy arrays of shape
        (n_channels,n_timepoints). Keras/tensorflow assume
        data is in shape (n_timepoints,n_channels). This method also assumes
        (n_timepoints,n_channels). Transpose should happen in fit.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be
            (n_timepoints,n_channels).

        Returns
        -------
        output : a compiled Keras Model.
        """
        import tensorflow as tf

        self._check_params()

        tf.keras.utils.set_random_seed(self._random_state)
        self._network.build_network(input_shape, **kwargs)
        model = self._network.get_autoencoder_model()
        model.compile(optimizer=self._optimizer, loss=self.loss, metrics=self._metrics)

        return model

    def _fit(self, X):
        """Fit the Clusterer on the training set X.

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases (n), n_channels (d), n_timepoints (m))
            The training input samples.

        Returns
        -------
        self : object
        """
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)
        if X.dtype != np.float32:
            # cast input data if it is not float32, as required by Keras/Tensorflow
            X = X.astype(np.float32)

        self.input_shape = X.shape[1:]
        self._training_model = self.build_model(self.input_shape)

        if self.save_init_model:
            self._training_model.save(self.file_path + self.init_file_name + ".keras")

        if self.verbose:
            self._training_model.summary()

        if self.use_mini_batch_size:
            self._batch_size = min(self.batch_size, X.shape[0] // 10)
        else:
            self._batch_size = self.batch_size

        if self.loss != "multi_rec":
            self.history = self._training_model.fit(
                X,
                X,
                batch_size=self._batch_size,
                validation_split=self.validation_split,
                epochs=self.n_epochs,
                verbose=self.verbose,
                callbacks=self._callbacks,
            )
        else:
            self.history = self._fit_multi_rec_model(
                autoencoder=self._training_model,
                inputs=X,
                batch_size=self._batch_size,
                validation_split=self.validation_split,
                epochs=self.n_epochs,
                verbose=self.verbose,
            )

        self._load_best_model_and_save()

        self._fit_clustering(X=X)

        gc.collect()
        return self

    def _split_validation_data(self, inputs, batch_size, validation_split):
        import tensorflow as tf

        train_size = int(len(inputs) * (1 - validation_split))

        train_dataset = (
            tf.data.Dataset.from_tensor_slices(inputs[:train_size])
            .shuffle(buffer_size=1024)
            .batch(batch_size)
        )

        val_dataset = tf.data.Dataset.from_tensor_slices(inputs[train_size:]).batch(
            batch_size
        )
        return train_dataset, val_dataset

    def _build_layerwise_model(self, autoencoder):
        import tensorflow as tf

        encoder = autoencoder.get_layer("encoder")
        decoder = autoencoder.get_layer("decoder")

        encoder_layers = [
            layer
            for layer in encoder.layers
            if layer.name.startswith("__act_encoder_block_")
        ]
        decoder_layers = [
            layer
            for layer in decoder.layers
            if layer.name.startswith("__act_decoder_block_")
        ]

        if len(encoder_layers) != len(decoder_layers):
            raise ValueError("The Auto-Encoder must be symmetric in nature.")

        outputs = [
            *[layer.output for layer in encoder_layers],
            *[layer.output for layer in decoder_layers],
            autoencoder.output,
        ]

        return tf.keras.Model(
            inputs=autoencoder.input,
            outputs=outputs,
        )

    def _layerwise_mse_loss(self, layerwise_model, inputs):
        import tensorflow as tf

        outputs = layerwise_model(inputs, training=True)
        outputs = [inputs, *outputs]

        n_layers = len(outputs) // 2
        encoder_outputs = outputs[:n_layers]
        decoder_outputs = outputs[n_layers:][::-1]

        return sum(
            tf.reduce_mean(tf.square(x - y))
            for x, y in zip(encoder_outputs, decoder_outputs)
        )

    def _fit_multi_rec_model_batch(self, step, layerwise_model, autoencoder, x_batch):
        import tensorflow as tf

        with tf.GradientTape() as tape:
            loss_value = self._layerwise_mse_loss(layerwise_model, x_batch)

        grads = tape.gradient(loss_value, autoencoder.trainable_weights)
        self._optimizer.apply_gradients(zip(grads, autoencoder.trainable_weights))

        # Update callbacks on batch end
        for callback in self._callbacks:
            callback.on_batch_end(step, {"loss": float(loss_value)})

        return float(loss_value)

    def _fit_multi_rec_model_epoch(
        self,
        epoch,
        layerwise_model,
        autoencoder,
        train_dataset,
        val_dataset,
        history,
        verbose,
    ):
        epoch_loss, val_loss = 0, 0
        num_batches, val_batches = len(train_dataset), len(val_dataset)

        # training
        for step, x_batch_train in enumerate(train_dataset):
            epoch_loss += self._fit_multi_rec_model_batch(
                step, layerwise_model, autoencoder, x_batch_train
            )
        epoch_loss /= num_batches
        history["loss"].append(epoch_loss)

        # validation
        for x_batch_val in val_dataset:
            val_loss += self._layerwise_mse_loss(layerwise_model, x_batch_val)
        if val_batches > 0:
            val_loss /= val_batches
            history["val_loss"].append(val_loss)

        if verbose:
            message = f"Training loss at epoch {epoch}: {epoch_loss:.4f}"
            if val_batches > 0:
                message += f", Validation loss: {val_loss:.4f}"
            sys.stdout.write(message + "\n")

        logs = {"loss": float(epoch_loss)}
        if val_batches > 0:
            logs["val_loss"] = float(val_loss)

        for callback in self._callbacks:
            callback.on_epoch_end(epoch, logs)

    def _fit_multi_rec_model(
        self,
        autoencoder,
        inputs,
        batch_size,
        validation_split,
        epochs,
        verbose,
    ):
        train_dataset, val_dataset = self._split_validation_data(
            inputs, batch_size, validation_split
        )
        layerwise_model = self._build_layerwise_model(autoencoder)

        history = {"loss": []}
        if len(val_dataset) > 0:
            history["val_loss"] = []

        # Initialize callbacks
        for callback in self._callbacks:
            callback.set_model(autoencoder)
            callback.on_train_begin()

        for epoch in range(epochs):
            self._fit_multi_rec_model_epoch(
                epoch,
                layerwise_model,
                autoencoder,
                train_dataset,
                val_dataset,
                history,
                verbose,
            )

        # Finalize callbacks
        for callback in self._callbacks:
            callback.on_train_end()

        return history

    def _load_best_model_and_save(self):
        import tensorflow as tf

        try:
            self._model = tf.keras.models.load_model(
                self.file_path + self._file_name + ".keras", compile=False
            )
            if not self.save_best_model:
                os.remove(self.file_path + self._file_name + ".keras")
        except FileNotFoundError:
            self._model = deepcopy(self._training_model)

        if self.save_last_model:
            self.save_last_model_to_file(file_path=self.file_path)

    def summary(self):
        """
        Summary function to return the losses/metrics for model fit.

        Returns
        -------
        history : dict or None,
            Dictionary containing model's train/validation losses and metrics

        """
        return self.history.history if self.history is not None else None

    def save_last_model_to_file(self, file_path="./"):
        """Save the last epoch of the trained deep learning model.

        Parameters
        ----------
        file_path : str, default = "./"
            The directory where the model will be saved

        Returns
        -------
        None
        """
        self._model.save(file_path + self.last_file_name + ".keras")

    def _fit_clustering(self, X):
        """Train the clustering algorithm in the latent space.

        Parameters
        ----------
        X : np.ndarray, shape=(n_cases, n_timepoints, n_channels)
            The input time series.
        """
        self.estimator_ = (
            TimeSeriesKMeans(
                n_clusters=2, distance="euclidean", averaging_method="mean"
            )
            if self.estimator is None
            else _clone_estimator(self.estimator)
        )
        latent_space = self._model.get_layer("encoder").predict(X)
        self.estimator_.fit(X=latent_space)
        if hasattr(self.estimator_, "labels_"):
            self.labels_ = self.estimator_.labels_
        else:
            self.labels_ = self.estimator_.predict(X=latent_space)

        return self

    def _predict(self, X):
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)
        latent_space = self._model.get_layer("encoder").predict(X)
        clusters = self.estimator_.predict(latent_space)

        return clusters

    def _predict_proba(self, X):
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)
        latent_space = self._model.get_layer("encoder").predict(X)
        clusters_proba = self.estimator_.predict_proba(latent_space)

        return clusters_proba

    def predict_encoder(self, X):
        """Predict the latent space representation of the input data.

        Parameters
        ----------
        X : np.ndarray, shape=(n_cases, n_timepoints, n_channels)
            The input time series.

        Returns
        -------
        latent_space : np.ndarray, shape=(n_cases, latent_space_dim)
            The latent space representation of the input data.
        """
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)
        latent_space = self._model.get_layer("encoder").predict(X)

        return latent_space

    def predict_decoder(self, latent_space):
        """Predict the reconstructed data from the latent space representation.

        Parameters
        ----------
        latent_space : np.ndarray, shape=(n_cases, latent_space_dim)
            The latent space representation of the input data.

        Returns
        -------
        reconstructed_data : np.ndarray, shape=(n_cases, n_timepoints, n_channels)
            The reconstructed data from the latent space representation.
        """
        reconstructed_data = self._model.get_layer("decoder").predict(latent_space)
        return reconstructed_data

    def load_model(self, model_path, estimator):
        """Load a pre-trained keras model instead of fitting.

        When calling this function, all functionalities can be used
        such as predict, predict_proba etc. with the loaded model.

        Parameters
        ----------
        model_path : str (path including model name and extension)
            The directory where the model will be saved including the model
            name with a ".keras" extension.
            Example: model_path="path/to/file/best_model.keras"
        estimator : estimator : aeon clusterer
            Pre-trained clusterer needed for loading model.

        Returns
        -------
        None
        """
        import tensorflow as tf

        self._model = tf.keras.models.load_model(model_path)
        self.is_fitted = True

        # use deep copy to preserve fit state
        self.estimator_ = deepcopy(estimator)

    def _add_model_checkpoint_callback(self, callbacks, file_path, file_name):
        import tensorflow as tf

        _model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=file_path + file_name + ".keras",
            monitor="val_loss" if self.validation_split > 0 else "loss",
            save_best_only=True,
        )

        if isinstance(callbacks, list):
            return callbacks + [_model_checkpoint]
        else:
            return [callbacks] + [_model_checkpoint]
