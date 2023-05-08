# -*- coding: utf-8 -*-
"""Time Convolutional Neural Network (CNN) for classification."""

__author__ = ["James-Large", "TonyBagnall", "hadifawaz1999"]
__all__ = ["CNNClassifier"]

import os
import time
from copy import deepcopy

from sklearn.utils import check_random_state

from aeon.classification.deep_learning.base import BaseDeepClassifier
from aeon.networks.cnn import CNNNetwork
from aeon.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class CNNClassifier(BaseDeepClassifier):
    """Time Convolutional Neural Network (CNN), as described in [1]_.

    Parameters
    ----------
    n_layers        : int, default = 2,
        the number of convolution layers in the network
    kernel_size    : int or list of int, default = 7,
        kernel size of convolution layers, if not a list, the same kernel size
        is used for all layer, len(list) should be n_layers
    n_filters       : int or list of int, default = [6, 12],
        number of filters for each convolution layer, if not a list, the same n_filters
        is used in all layers.
    avg_pool_size   : int or list of int, default = 3,
        the size of the average pooling layer, if not a list, the same
        max pooling size is used
        for all convolution layer
    activation      : str or list of str, default = "sigmoid",
        keras activation function used in the model for each layer,
        if not a list, the same
        activation is used for all layers
    padding         : str or list of str, default = 'valid',
        the method of padding in convolution layers, if not a list,
        the same padding used
        for all convolution layers
    strides         : int or list of int, default = 1,
        the strides of kernels in the convolution and max pooling layers,
        if not a list, the same strides are used for all layers
    dilation_rate   : int or list of int, default = 1,
        the dilation rate of the convolution layers, if not a list,
        the same dilation rate is used all over the network
    use_bias        : bool or list of bool, default = True,
        condition on wether or not to use bias values for convolution layers,
        if not a list, the same condition is used for all layers
    random_state    : int, default = 0
        seed to any needed random actions
    n_epochs       : int, default = 2000
        the number of epochs to train the model
    batch_size      : int, default = 16
        the number of samples per gradient update.
    verbose         : boolean, default = False
        whether to output extra information
    loss            : string, default="mean_squared_error"
        fit parameter for the keras model
    optimizer       : keras.optimizer, default=keras.optimizers.Adam(),
    metrics         : list of strings, default=["accuracy"],
    callbacks       : keras.callbacks, default=model_checkpoint to save best
                      model on training loss
    file_path       : file_path for the best model (if checkpoint is used as callback)


    Notes
    -----
    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/cnn.py

    References
    ----------
    .. [1] Zhao et. al, Convolutional neural networks for time series classification,
    Journal of Systems Engineering and Electronics, 28(1):2017.

    Examples
    --------
    >>> from aeon.classification.deep_learning.cnn import CNNClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> cnn = CNNClassifier(n_epochs=20,batch_size=4)  # doctest: +SKIP
    >>> cnn.fit(X_train, y_train)  # doctest: +SKIP
    CNNClassifier(...)
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        n_layers=2,
        kernel_size=7,
        n_filters=None,
        avg_pool_size=3,
        activation="sigmoid",
        padding="valid",
        strides=1,
        dilation_rate=1,
        n_epochs=2000,
        batch_size=16,
        callbacks=None,
        file_path="./",
        verbose=False,
        loss="mean_squared_error",
        metrics=None,
        random_state=None,
        use_bias=True,
        optimizer=None,
    ):
        _check_dl_dependencies(severity="error")
        super(CNNClassifier, self).__init__()

        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.avg_pool_size = avg_pool_size
        self.activation = activation
        self.use_bias = use_bias
        self.random_state = random_state

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.file_path = file_path
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.history = None
        self._network = CNNNetwork(
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            n_filters=self.n_filters,
            avg_pool_size=self.avg_pool_size,
            activation=self.activation,
            padding=self.padding,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            use_bias=self.use_bias,
            random_state=self.random_state,
        )

    def build_model(self, input_shape, n_classes, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In aeon, time series are stored in numpy arrays of shape (d,m), where d
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
            tf.keras.optimizers.Adam() if self.optimizer is None else self.optimizer
        )

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=metrics,
        )

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
        import tensorflow as tf

        y_onehot = self.convert_y_to_keras(y)
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        check_random_state(self.random_state)

        self.input_shape = X.shape[1:]
        self.training_model_ = self.build_model(self.input_shape, self.n_classes_)

        if self.verbose:
            self.training_model_.summary()

        self.file_name_ = str(time.time_ns())

        self.callbacks_ = (
            [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.file_path + self.file_name_ + ".hdf5",
                    monitor="loss",
                    save_best_only=True,
                ),
            ]
            if self.callbacks is None
            else self.callbacks
        )

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
                self.file_path + self.file_name_ + ".hdf5", compile=False
            )
            os.remove(self.file_path + self.file_name_ + ".hdf5")
        except FileNotFoundError:
            self.model_ = deepcopy(self.training_model_)

        return self

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
            "avg_pool_size": 4,
        }

        test_params = [param1]

        return test_params
