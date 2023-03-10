# -*- coding: utf-8 -*-
"""Encoder Classifier."""

__author__ = ["hadifawaz1999"]
__all__ = ["EncoderClassifier"]

from copy import deepcopy

from sklearn.utils import check_random_state

from aeon.classification.deep_learning.base import BaseDeepClassifier
from aeon.networks.encoder import EncoderNetwork
from aeon.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class EncoderClassifier(BaseDeepClassifier):
    """
    Establish the network structure for an Encoder.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    kernel_size    : array of int, default = [5, 11, 21]
        specifying the length of the 1D convolution windows
    n_filters       : array of int, default = [128, 256, 512]
        specifying the number of 1D convolution filters used for each layer,
        the shape of this array should be the same as kernel_size
    max_pool_size   : int, default = 2
        size of the max pooling windows
    activation      : string, default = sigmoid
        keras activation function
    dropout_proba   : float, default = 0.2
        specifying the dropout layer probability
    padding         : string, default = same
        specifying the type of padding used for the 1D convolution
    strides         : int, default = 1
        specifying the sliding rate of the 1D convolution filter
    fc_units        : int, default = 256
        specifying the number of units in the hiddent fully
        connected layer used in the EncoderNetwork
    random_state    : int, default = 0
        seed to any needed random actions

    Notes
    -----
    Adapted from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/encoder.py

    References
    ----------
    .. [1] Serrà et al. Towards a Universal Neural Network Encoder for Time Series
    In proceedings International Conference of the Catalan Association
    for Artificial Intelligence, 120--129 2018.


    """

    _tags = {"python_dependencies": ["tensorflow", "tensorflow_addons"]}

    def __init__(
        self,
        n_epochs=100,
        batch_size=12,
        kernel_size=None,
        n_filters=None,
        dropout_proba=0.2,
        activation="sigmoid",
        max_pool_size=2,
        padding="same",
        strides=1,
        fc_units=256,
        callbacks=None,
        file_path="./",
        verbose=False,
        loss="categorical_crossentropy",
        metrics=None,
        random_state=None,
        use_bias=True,
        optimizer=None,
    ):
        _check_dl_dependencies(severity="error")
        super(EncoderClassifier, self).__init__()

        self.n_filters = n_filters
        self.max_pool_size = max_pool_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.padding = padding
        self.dropout_proba = dropout_proba
        self.fc_units = fc_units

        self.callbacks = callbacks
        self.file_path = file_path
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.random_state = random_state
        self.use_bias = use_bias
        self.optimizer = optimizer
        self.history = None

        self._network = EncoderNetwork(
            kernel_size=self.kernel_size,
            max_pool_size=self.max_pool_size,
            n_filters=self.n_filters,
            fc_units=self.fc_units,
            strides=self.strides,
            padding=self.padding,
            dropout_proba=self.dropout_proba,
            activation=self.activation,
            random_state=self.random_state,
        )

    def build_model(self, input_shape, n_classes, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In sktime, time series are stored in numpy arrays of shape (d,m), where d
        is the number of dimensions, m is the series length. Keras/tensorflow assume
        data is in shape (m,d). This method also assumes (m,d). Transpose should
        happen in fit.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be (m,d)
        n_classes: int
            The number of classes, which becomes the size of the output layer

        Returns
        -------
        output : a compiled Keras Model
        """
        import tensorflow as tf

        tf.random.set_seed(self.random_state)

        if self.metrics is None:
            metrics = ["accuracy"]
        else:
            metrics = self.metrics
        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = tf.keras.layers.Dense(
            units=n_classes, activation=self.activation, use_bias=self.use_bias
        )(output_layer)

        self.optimizer_ = (
            tf.keras.optimizers.Adam(learning_rate=0.00001)
            if self.optimizer is None
            else self.optimizer
        )

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=metrics,
        )

        # self.callbacks = [
        #     tf.keras.callbacks.ModelCheckpoint(
        #         filepath=self.file_path + "best_model.hdf5",
        #         monitor="loss",
        #         save_best_only=True,
        #     )
        #     if self.callbacks is None
        #     else self.callbacks
        # ]

        return model

    def _fit(self, X, y):
        """Fit the classifier on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray of shape = (n_instances (n), n_dimensions (d), series_length (m))
            The training input samples.
        y : np.ndarray of shape n
            The training data class labels.

        Returns
        -------
        self : object
        """
        y_onehot = self.convert_y_to_keras(y)
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        check_random_state(self.random_state)
        self.input_shape = X.shape[1:]
        self.model_ = self.build_model(self.input_shape, self.n_classes_)
        if self.verbose:
            self.model_.summary()
        self.history = self.model_.fit(
            X,
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=deepcopy(self.callbacks) if self.callbacks else [],
        )

        return self

        # try:
        #     import os

        #     import tensorflow as tf

        #     self.model_ = tf.keras.models.load_model(
        #         self.file_path + "best_model.hdf5", compile=False
        #     )
        #     os.remove(self.file_path + "best_model.hdf5")

        #     return self
        # except FileNotFoundError:
        #     return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        param1 = {
            "n_epochs": 10,
            "batch_size": 4,
        }

        # param2 = {
        #     "n_epochs": 12,
        #     "batch_size": 6,
        #     "max_pool_size": 1,
        # }
        test_params = [param1]

        return test_params
