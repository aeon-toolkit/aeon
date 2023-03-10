# -*- coding: utf-8 -*-
"""Fully Connected Neural Network (CNN) for classification."""

__author__ = ["James-Large", "AurumnPegasus", "hadifawaz1999"]
__all__ = ["FCNClassifier"]

from copy import deepcopy

from sklearn.utils import check_random_state

from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.networks.fcn import FCNNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class FCNClassifier(BaseDeepClassifier):
    """Fully Connected Neural Network (FCN), as described in [1]_.

    Parameters
    ----------
    n_layers        : int, default = 3
        number of convolution layers
    n_filters       : int or list of int, default = [128,256,128]
        number of filters used in convolution layers
    kernel_size     : int or list of int, default = [8,5,3]
        size of convolution kernel
    dilation_rate   : int or list of int, default = 1
        the dilation rate for convolution
    strides         : int or list of int, default = 1
        the strides of the convolution filter
    padding         : str or list of str, default = "same"
        the type of padding used for convolution
    activation      : str or list of str, default = "relu"
        activation used after the convolution
    use_bias        : bool or list of bool, default = True
        whether or not ot use bias in convolution
    n_epochs        : int, default = 2000
        the number of epochs to train the model
    batch_size      : int, default = 16
        the number of samples per gradient update.
    use_mini_batch_size : bool, default = True,
        whether or not to use the mini batch size formula
    random_state    : int or None, default=None
        Seed for random number generation.
    verbose         : boolean, default = False
        whether to output extra information
    loss            : string, default="mean_squared_error"
        fit parameter for the keras model
    metrics         : list of strings, default=["accuracy"],
    optimizer       : keras.optimizers object, default = Adam(lr=0.01)
        specify the optimizer and the learning rate to be used.
    file_path       : str, default = "./"
        file path to save best model
    callbacks       : keras.callbacks, default = None
    Notes
    -----
    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py

    References
    ----------
    .. [1] Zhao et. al, Convolutional neural networks for time series classification,
    Journal of Systems Engineering and Electronics, 28(1):2017.

    Examples
    --------
    >>> from sktime.classification.deep_learning.fcn import FCNClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> fcn = FCNClassifier(n_epochs=20,batch_size=4)  # doctest: +SKIP
    >>> fcn.fit(X_train, y_train)  # doctest: +SKIP
    FCNClassifier(...)
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        n_layers=3,
        n_filters=None,
        kernel_size=None,
        dilation_rate=1,
        strides=1,
        padding="same",
        activation="relu",
        file_path="./",
        n_epochs=2000,
        batch_size=16,
        use_mini_batch_size=True,
        callbacks=None,
        verbose=False,
        loss="categorical_crossentropy",
        metrics=None,
        random_state=None,
        use_bias=True,
        optimizer=None,
    ):
        _check_dl_dependencies(severity="error")
        super(FCNClassifier, self).__init__()

        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.strides = strides
        self.activation = activation
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.use_bias = use_bias

        self.callbacks = callbacks
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.use_mini_batch_size = use_mini_batch_size
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.random_state = random_state
        self.optimizer = optimizer
        self.history = None
        self.file_path = file_path
        self._network = FCNNetwork(
            random_state=self.random_state,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            n_filters=self.n_filters,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
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
            units=n_classes, activation="softmax", use_bias=self.use_bias
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

        # self.callbacks = [
        #     tf.keras.callbacks.ModelCheckpoint(
        #         filepath=self.file_path + "best_model.hdf5",
        #         monitor="loss",
        #         save_best_only=True,
        #     ),
        #     tf.keras.callbacks.ReduceLROnPlateau(
        #         monitor="loss", factor=0.5, patience=50, min_lr=0.0001
        #     )
        #     if self.callbacks is None
        #     else self.callbacks,
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

        if self.use_mini_batch_size:
            mini_batch_size = min(self.batch_size, X.shape[0] // 10)
        else:
            mini_batch_size = self.batch_size

        self.history = self.model_.fit(
            X,
            y_onehot,
            batch_size=mini_batch_size,
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
            "use_bias": False,
        }

        param2 = {
            "n_epochs": 12,
            "batch_size": 6,
            "use_bias": True,
        }
        test_params = [param1, param2]

        return test_params
