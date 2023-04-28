# -*- coding: utf-8 -*-
"""Residual Network (ResNet) for classification."""

__author__ = ["James-Large", "AurumnPegasus", "nilesh05apr"]
__all__ = ["ResNetClassifier"]

from sklearn.utils import check_random_state

from aeon.classification.deep_learning.base import BaseDeepClassifier
from aeon.networks.resnet import ResNetNetwork
from aeon.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class ResNetClassifier(BaseDeepClassifier):
    """
    Residual Neural Network as described in [1].

    Parameters
    ----------
        n_residual_blocks           : int, default = 3,
            the number of residual blocks of ResNet's model
        n_conv_per_residual_block   : int, default = 3,
            the number of convolution blocks in each residual block
        n_filters                   : int or list of int, default = [128, 64, 64],
            the number of convolution filters for all the convolution layers in the same
            residual block, if not a list, the same number of filters is used in all
            convolutions of all residual blocks.
        kernel_sizes                : int or list of int, default = [8, 5, 3],
            the kernel size of all the convolution layers in one residual block, if not
            a list, the same kernel size is used in all convolution layers
        strides                     : int or list of int, default = 1,
            the strides of convolution kernels in each of the
            convolution layers in one residual block, if not
            a list, the same kernel size is used in all convolution layers
        dilation_rate               : int or list of int, default = 1,
            the dilation rate of the convolution layers in one residual block, if not
            a list, the same kernel size is used in all convolution layers
        padding                     : str or list of str, default = 'padding',
            the type of padding used in the convolution layers
            in one residual block, if not
            a list, the same kernel size is used in all convolution layers
        activation                  : str or list of str, default = 'relu',
            keras activation used in the convolution layers
            in one residual block, if not
            a list, the same kernel size is used in all convolution layers
        use_bias                    : bool or list of bool, default = True,
            condition on wether or not to use bias values in
            the convolution layers in one residual block, if not
            a list, the same kernel size is used in all convolution layers

        n_epochs                   : int, default = 1500
            the number of epochs to train the model
        batch_size                  : int, default = 16
            the number of samples per gradient update.
        use_mini_batch_size         : bool, default = False
            condition on using the mini batch size formula Wang et al.
        callbacks                   : callable or None, default
        ReduceOnPlateau and ModelCheckpoint
            list of tf.keras.callbacks.Callback objects.
        file_path                   : str, default = './'
            file_path when saving model_Checkpoint callback
        verbose                     : boolean, default = False
            whether to output extra information
        loss                        : string, default="mean_squared_error"
            fit parameter for the keras model
        optimizer                   : keras.optimizer, default=keras.optimizers.Adam(),
        metrics                     : list of strings, default=["accuracy"],

    Notes
    -----
    Adapted from the implementation from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py

    References
    ----------
        .. [1] Wang et. al, Time series classification from
    scratch with deep neural networks: A strong baseline,
    International joint conference on neural networks (IJCNN), 2017.

    Examples
    --------
    >>> from aeon.classification.deep_learning.resnet import ResNetClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> clf = ResNetClassifier(n_epochs=20, bacth_size=4) # doctest: +SKIP
    >>> clf.fit(X_train, Y_train) # doctest: +SKIP
    ResNetClassifier(...)
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        n_epochs=1500,
        callbacks=None,
        verbose=False,
        loss="categorical_crossentropy",
        metrics=None,
        batch_size=16,
        random_state=None,
        activation="sigmoid",
        use_bias=True,
        optimizer=None,
    ):
        _check_dl_dependencies(severity="error")
        super(ResNetClassifier, self).__init__()
        self.n_epochs = n_epochs
        self.callbacks = callbacks
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.random_state = random_state
        self.activation = activation
        self.use_bias = use_bias
        self.optimizer = optimizer
        self.history = None
        self._network = ResNetNetwork(random_state=random_state)

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
        from tensorflow import keras

        tf.random.set_seed(self.random_state)

        self.optimizer_ = (
            keras.optimizers.Adam(learning_rate=0.01)
            if self.optimizer is None
            else self.optimizer
        )

        if self.metrics is None:
            metrics = ["accuracy"]
        else:
            metrics = self.metrics

        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(
            units=n_classes, activation=self.activation, use_bias=self.use_bias
        )(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
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
        self.model_ = self.build_model(self.input_shape, self.n_classes_)
        if self.verbose:
            self.model_.summary()

        self.callbacks_ = (
            [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=50, min_lr=0.0001
                )
            ]
            if self.callbacks is None
            else self.callbacks
        )

        self.history = self.model_.fit(
            X,
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks_,
        )

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
            "use_bias": False,
        }

        # param2 = {
        #     "n_epochs": 12,
        #     "batch_size": 6,
        #     "use_bias": True,
        # }
        test_params = [param1]

        return test_params
