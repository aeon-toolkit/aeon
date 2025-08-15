"""TimeMCL SSL transformer."""

from __future__ import annotations

__maintainer__ = ["hadifawaz1999"]
__all__ = ["TimeMCL"]

import gc
import os
import sys
import time
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from sklearn.utils import check_random_state

from aeon.networks import BaseDeepLearningNetwork
from aeon.transformations.collection import BaseCollectionTransformer

if TYPE_CHECKING:
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.optimizers import Optimizer


class TimeMCL(BaseCollectionTransformer):
    """Time Mixup Contrastive Learning (TimeMCL).

    TimeMCL [1]_ learns from time series without
    labels by mixing up samples and trying to predict
    how much each original contributed. Given two inputs
    x1 and x2, it creates a new sample
    x̃ = λ * x1 + (1 - λ) * x2, where λ ∈ [0, 1].
    By learning from the mixing ratio λ, it builds strong,
    general features for other tasks.

    Parameters
    ----------
    alpha : float, default = 0.2
        The alpha value for the Beta distribution. In the Beta distribution,
        alpha controls how strongly the distribution is pulled toward 1,
        with larger alpha pushing probability mass toward higher values of x
        and smaller alpha concentrating it near 0.
        More info here:
        https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta
    mixup_temperature : float, default = 0.5
        The value that controls the logits smoothness.
    backbone_network : aeon Network, default = None
        The backbone network used for the SSL model, it can be any network from
        the aeon.networks module on condition for it's structure to be configured
        as "encoder", see _config attribute. For TimeMCL, the default network
        used is FCNNetwork(n_layers=3,
                           n_filters=[128, 256, 128],
                           kernel_size=[7, 5, 3],
                           dilation_rate=[2, 4, 8]).
    latent_space_dim : int, default = 128
        The size of the latent space, applied using a fully connected layer
        at the end of the network's output.
    latent_space_activation : str, default = "linear"
        The activation to control the range of values
        of the latent space.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
        Seeded random number generation can only be guaranteed on CPU processing,
        GPU processing will be non-deterministic.
    verbose : boolean, default = False
        Whether to output extra information.
    optimizer : keras.optimizer, default = tf.keras.optimizers.Adam()
        The keras optimizer used for training.
    file_path : str, default = "./"
        File path to save best model.
    save_best_model : bool, default = False
        Whether or not to save the best model, if the modelcheckpoint callback
        is used by default, this condition, if True, will prevent theautomatic
        deletion of the best saved model from file and the user can choose the
        file name.
    save_last_model : bool, default = False
        Whether or not to save the last model, last epoch trained, using the
        base class method save_last_model_to_file.
    save_init_model : bool, default = False
        Whether to save the initialization of the model.
    best_file_name : str, default = "best_model"
        The name of the file of the best model, if save_best_model is set to
        False, this parameter is discarded.
    last_file_name : str, default = "last_model"
        The name of the file of the last model, if save_last_model is set to
        False, this parameter is discarded.
    init_file_name : str, default = "init_model"
        The name of the file of the init model, if save_init_model is set to False,
        this parameter is discarded.
    callbacks : keras callback or list of callbacks,
        default = None
        The default list of callbacks are set to ModelCheckpoint and ReduceLROnPlateau.
        ModelCheckpoint will ensure the best model over the training loss is being
        saved, which will then be loaded when fitting is finished, the file will be
        deleted unless ``save_best_model`` is set to ``True``. ReduceLROnPlateau
        reduces the learning rate during trainining using a schedualar.
        More info on these two callbacks are available on the keras docs.
    batch_size : int, default = 64
        The number of samples per gradient update.
    use_mini_batch_size : bool, default = False
        Whether or not to use the mini batch size formula.
    n_epochs : int, default = 1000
        The number of epochs to train the model.

    Notes
    -----
    Adapted from the implementation from Wickstrøm et. al
    https://github.com/Wickstrom/MixupContrastiveLearning

    References
    ----------
    .. [1] Wickstrøm, Kristoffer, Michael Kampffmeyer,
    Karl Øyvind Mikalsen, and Robert Jenssen. "Mixing up
    contrastive learning: Self-supervised representation
    learning for time series." Pattern Recognition
    Letters 155 (2022): 54-61.

    Examples
    --------
    >>> from aeon.transformations.collection.self_supervised import TimeMCL
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> ssl = TimeMCL(latent_space_dim=2, n_epochs=5) # doctest: +SKIP
    >>> ssl.fit(X_train) # doctest: +SKIP
    TimeMCL(...)
    >>> X_train_transformed = ssl.transform(X_train) # doctest: +SKIP
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "algorithm_type": "deeplearning",
        "python_dependencies": "tensorflow",
        "non_deterministic": True,
        "cant_pickle": True,
    }

    def __init__(
        self,
        alpha: float = 0.2,
        mixup_temperature: float = 0.5,
        backbone_network: BaseDeepLearningNetwork = None,
        latent_space_dim: int = 128,
        latent_space_activation: str = "linear",
        random_state: int | np.random.RandomState | None = None,
        verbose: bool = False,
        optimizer: Optimizer | None = None,
        file_path: str = "./",
        save_best_model: bool = False,
        save_last_model: bool = False,
        save_init_model: bool = False,
        best_file_name: str = "best_model",
        last_file_name: str = "last_model",
        init_file_name: str = "init_model",
        callbacks: Callback | list[Callback] | None = None,
        batch_size: int = 64,
        use_mini_batch_size: bool = False,
        n_epochs: int = 1000,
    ):

        self.alpha = alpha
        self.mixup_temperature = mixup_temperature
        self.backbone_network = backbone_network
        self.latent_space_dim = latent_space_dim
        self.latent_space_activation = latent_space_activation
        self.random_state = random_state
        self.verbose = verbose
        self.optimizer = optimizer
        self.file_path = file_path
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.last_file_name = last_file_name
        self.init_file_name = init_file_name
        self.callbacks = callbacks
        self.batch_size = batch_size
        self.use_mini_batch_size = use_mini_batch_size
        self.n_epochs = n_epochs

        super().__init__()

    def _fit(self, X: np.ndarray, y=None):
        """Fit the SSL model on X, y is ignored.

        Parameters
        ----------
        X : np.ndarray
            The training input samples of shape (n_cases, n_channels, n_timepoints)
        y : ignored argument for interface compatibility

        Returns
        -------
        self : object
        """
        import tensorflow as tf

        from aeon.networks import FCNNetwork

        assert self.alpha > 0, "alpha must be greater than 0"
        assert self.mixup_temperature > 0, "mixup_temperature must be greater than 0"

        if isinstance(self.backbone_network, BaseDeepLearningNetwork):
            self._backbone_network = deepcopy(self.backbone_network)
        elif self.backbone_network is None:
            self._backbone_network = FCNNetwork(
                n_layers=3,
                n_filters=[128, 256, 128],
                kernel_size=[7, 5, 3],
                dilation_rate=[2, 4, 8],
            )
        else:
            raise ValueError(
                "The parameter backbone_network", "should be an aeon network."
            )

        X = X.transpose(0, 2, 1)

        self.input_shape = X.shape[1:]
        self.training_model_ = self.build_model(self.input_shape)

        if self.save_init_model:
            self.training_model_.save(
                os.path.join(self.file_path, self.init_file_name + ".keras")
            )

        if self.verbose:
            self.training_model_.summary()

        if self.use_mini_batch_size:
            mini_batch_size = min(self.batch_size, X.shape[0] // 10)
        else:
            mini_batch_size = self.batch_size

        self.file_name_ = (
            self.best_file_name if self.save_best_model else str(time.time_ns())
        )

        if self.callbacks is None:
            self.callbacks_ = [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=50, min_lr=0.0001
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.file_path, self.file_name_ + ".keras"),
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

        fake_y = np.zeros(shape=len(X))

        train_dataset = tf.data.Dataset.from_tensor_slices((X, fake_y))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(mini_batch_size)

        history = {"loss": []}

        for callback in self.callbacks_:
            callback.set_model(self.training_model_)
            callback.on_train_begin()

        for epoch in range(self.n_epochs):
            epoch_loss = 0
            num_batches = 0

            for step, (x_batch_train, _) in enumerate(train_dataset):

                x_batch_train2 = tf.random.shuffle(x_batch_train)

                _contrastive_weight = np.random.beta(self.alpha, self.alpha)

                x_batch_augmented = (
                    _contrastive_weight * x_batch_train
                    + (1 - _contrastive_weight) * x_batch_train2
                )

                with tf.GradientTape() as tape:
                    z1 = self.training_model_(x_batch_train, training=True)
                    z2 = self.training_model_(x_batch_train2, training=True)
                    z_augmented = self.training_model_(x_batch_augmented, training=True)

                    loss_batch = self._mixup_loss(
                        z1=z1,
                        z2=z2,
                        z_augmented=z_augmented,
                        contrastive_weight=_contrastive_weight,
                    )
                    loss_mean = tf.reduce_mean(loss_batch)

                gradients = tape.gradient(
                    loss_mean, self.training_model_.trainable_weights
                )
                self.optimizer_.apply_gradients(
                    zip(gradients, self.training_model_.trainable_weights)
                )

                epoch_loss += float(loss_mean)
                num_batches += 1

                for callback in self.callbacks_:
                    callback.on_batch_end(step, {"loss": float(loss_mean)})

            epoch_loss /= num_batches
            history["loss"].append(epoch_loss)

            if self.verbose:
                sys.stdout.write(
                    "Training loss at epoch %d: %.4f\n" % (epoch, float(epoch_loss))
                )

            for callback in self.callbacks_:
                callback.on_epoch_end(epoch, {"loss": float(epoch_loss)})

        for callback in self.callbacks_:
            callback.on_train_end()

        self.history = history

        try:
            self.model_ = tf.keras.models.load_model(
                os.path.join(self.file_path, self.file_name_ + ".keras"), compile=False
            )
            if not self.save_best_model:
                os.remove(os.path.join(self.file_path, self.file_name_ + ".keras"))
        except FileNotFoundError:
            self.model_ = deepcopy(self.training_model_)

        if self.save_last_model:
            self.save_last_model_to_file(file_path=self.file_path)

        gc.collect()
        return self

    def _transform(self, X, y=None):
        """Transform input time series using TimeMCL.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        np.ndarray (n_cases, latent_space_dim), transformed features
        """
        X = X.transpose(0, 2, 1)
        X_transformed = self.model_.predict(X, self.batch_size)

        return X_transformed

    def build_model(self, input_shape):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In aeon, time series are stored in numpy arrays of shape (d,m), where d
        is the number of dimensions, m is the series length. Keras/tensorflow assume
        data is in shape (m,d). This method also assumes (m,d). Transpose should
        happen in fit.

        Parameters
        ----------
        input_shape : tuple[int, int]
            The shape of the data fed into the input layer, should be (m, d).

        Returns
        -------
        output : a compiled Keras Model
        """
        import tensorflow as tf

        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)
        tf.keras.utils.set_random_seed(self.random_state_)

        input_layer, gap_layer = self._backbone_network.build_network(
            input_shape=input_shape
        )
        dense1 = tf.keras.layers.Dense(
            units=self.latent_space_dim, activation=self.latent_space_activation
        )(gap_layer)
        batch_norm1 = tf.keras.layers.BatchNormalization()(dense1)
        activation1 = tf.keras.layers.Activation(activation="relu")(batch_norm1)
        output_layer = tf.keras.layers.Dense(
            units=self.latent_space_dim, activation=self.latent_space_activation
        )(activation1)

        model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=output_layer,
        )

        self.optimizer_ = (
            tf.keras.optimizers.Adam() if self.optimizer is None else self.optimizer
        )

        # compile but won't be used
        model.compile(loss="mse", optimizer=self.optimizer_)

        return model

    def _mixup_loss(self, z1, z2, z_augmented, contrastive_weight):
        import tensorflow as tf

        batch_size = tf.shape(z_augmented)[0]

        z1 = tf.nn.l2_normalize(z1, axis=1)
        z2 = tf.nn.l2_normalize(z2, axis=1)
        z_augmented = tf.nn.l2_normalize(z_augmented, axis=1)

        eye = tf.eye(batch_size)
        labels_lamda_0 = contrastive_weight * eye
        labels_lamda_1 = (1 - contrastive_weight) * eye
        labels = tf.concat([labels_lamda_0, labels_lamda_1], axis=1)

        logits_1 = tf.matmul(z_augmented, z1, transpose_b=True)
        logits_2 = tf.matmul(z_augmented, z2, transpose_b=True)
        logits = tf.concat([logits_1, logits_2], axis=1)

        logits /= self.mixup_temperature

        log_probs = tf.nn.log_softmax(logits, axis=1)
        loss = -tf.reduce_sum(labels * log_probs, axis=1)

        return loss

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
        self.model_.save(os.path.join(file_path, self.last_file_name + ".keras"))

    def load_model(self, model_path):
        """Load a pre-trained keras model instead of fitting.

        When calling this function, all functionalities can be used
        such as predict, predict_proba etc. with the loaded model.

        Parameters
        ----------
        model_path : str (path including model name and extension)
            The directory where the model will be saved including the model
            name with a ".keras" extension.
            Example: model_path="path/to/file/best_model.keras"

        Returns
        -------
        None
        """
        import tensorflow as tf

        self.model_ = tf.keras.models.load_model(model_path)
        self.is_fitted = True

    def _get_model_checkpoint_callback(self, callbacks, file_path, file_name):
        import tensorflow as tf

        model_checkpoint_ = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(file_path, file_name + ".keras"),
            monitor="loss",
            save_best_only=True,
        )

        if isinstance(callbacks, list):
            return callbacks + [model_checkpoint_]
        else:
            return [callbacks] + [model_checkpoint_]

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the transformer.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        from aeon.networks import FCNNetwork

        params = {
            "latent_space_dim": 2,
            "backbone_network": FCNNetwork(n_layers=1, n_filters=2, kernel_size=2),
            "n_epochs": 3,
        }

        return [params]
