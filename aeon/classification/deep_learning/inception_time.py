# -*- coding: utf-8 -*-
"""InceptionTime classifier."""

__author__ = ["James-Large", "TonyBagnall", "MatthewMiddlehurst", "hadifawaz1999"]
__all__ = ["InceptionTimeClassifier"]

import os
from copy import deepcopy

import numpy as np
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.classification.deep_learning.base import BaseDeepClassifier
from aeon.networks.inception import InceptionNetwork
from aeon.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class InceptionTimeClassifier(BaseClassifier):
    """InceptionTime ensemble classifier.

    Ensemble of IndividualInceptionTimeClassifiers, as described in [1].

    Parameters
    ----------
        n_classifiers       : int, default = 5,
            the number of Inception models used for the
            Ensemble in order to create
            InceptionTime.
        depth               : int, default = 6,
            the number of inception modules used
        nb_filters          : int or list of int32, default = 32,
            the number of filters used in one inception
            module, if not a list,
            the same number of filters is used in
            all inception modules
        nb_conv_per_layer   : int or list of int, default = 3,
            the number of convolution layers in each inception
            module, if not a list,
            the same number of convolution layers is used
            in all inception modules
        kernel_size         : int or list of int, default = 40,
            the head kernel size used for each inception
            module, if not a list,
            the same is used in all inception modules
        use_max_pooling     : bool or list of bool, default = True,
            conditioning wether or not to use max pooling layer
            in inception modules,if not a list,
            the same is used in all inception modules
        max_pool_size       : int or list of int, default = 3,
            the size of the max pooling layer, if not a list,
            the same is used in all inception modules
        strides             : int or list of int, default = 1,
            the strides of kernels in convolution layers for each
            inception module, if not a list,
            the same is used in all inception modules
        dilation_rate       : int or list of int, default = 1,
            the dilation rate of convolutions in each inception
            module, if not a list,
            the same is used in all inception modules
        padding             : str or list of str, default = "same",
            the type of padding used for convoltuon for each
            inception module, if not a list,
            the same is used in all inception modules
        activation          : str or list of str, default = "relu",
            the activation function used in each inception
            module, if not a list,
            the same is used in all inception modules
        use_bias            : bool or list of bool, default = False,
            conditioning wether or not convolutions should
            use bias values in each inception
            module, if not a list,
            the same is used in all inception modules
        use_residual        : bool, default = True,
            condition wether or not to use residual
            connections all over Inception
        use_bottleneck      : bool, default = True,
            confition wether or not to use bottlesnecks
            all over Inception
        bottleneck_size     : int, default = 32,
            the bottleneck size in case use_bottleneck = True
        use_custom_filters  : bool, default = True,
            condition on wether or not to use custom
            filters in the first inception module
        batch_size          : int, default = 64
            the number of samples per gradient update.
        use_mini_batch_size : bool, default = False
            condition on using the mini batch size
            formula Wang et al.
        n_epochs           : int, default = 1500
            the number of epochs to train the model.
        callbacks           : callable or None, default
        ReduceOnPlateau and ModelCheckpoint
            list of tf.keras.callbacks.Callback objects.
        file_path           : str, default = "./"
            file_path when saving model_Checkpoint callback
        random_state        : int, default = 0
            seed to any needed random actions.
        verbose             : boolean, default = False
            whether to output extra information
        optimizer           : keras optimizer, default = Adam
        loss                : keras loss,
                              default = categorical_crossentropy
        metrics             : keras metrics, default = None,
        will be set to accuracy as default if None

    Notes
    -----
    ..[1] Fawaz et al. InceptionTime: Finding AlexNet for Time Series
    Classification, Data Mining and Knowledge Discovery, 34, 2020

    ..[2] Ismail-Fawaz et al. Deep Learning For Time Series
    Classification Using New
    Hand-Crafted Convolution Filters, 2022 IEEE International
    Conference on Big Data.

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py

    and Ismail-Fawaz et al.
    https://github.com/MSD-IRIMAS/CF-4-TSC

    """

    _tags = {
        "python_dependencies": "tensorflow",
        "capability:multivariate": True,
        "algorithm_type": "deeplearning",
    }

    def __init__(
        self,
        n_classifiers=5,
        nb_filters=32,
        nb_conv_per_layer=3,
        kernel_size=40,
        use_max_pooling=True,
        max_pool_size=3,
        strides=1,
        dilation_rate=1,
        padding="same",
        activation="relu",
        use_bias=False,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        use_custom_filters=True,
        file_path="./",
        batch_size=64,
        use_mini_batch_size=False,
        n_epochs=1500,
        callbacks=None,
        random_state=None,
        verbose=False,
        loss="categorical_crossentropy",
        metrics=None,
        optimizer=None,
    ):
        self.n_classifiers = n_classifiers

        self.nb_filters = nb_filters
        self.nb_conv_per_layer = nb_conv_per_layer
        self.use_max_pooling = use_max_pooling
        self.max_pool_size = max_pool_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size
        self.depth = depth
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.use_custom_filters = use_custom_filters

        self.file_path = file_path

        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose
        self.use_mini_batch_size = use_mini_batch_size
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        self.classifers_ = []

        super(InceptionTimeClassifier, self).__init__()

    def _fit(self, X, y):
        """Fit each of the Individual Inception models.

        Arguments:
        ----------

        X : np.ndarray of shape = (n_instances (n), n_channels (c), series_length (m))
            The training input samples.
        y : np.ndarray of shape n
            The training data class labels.

        Returns
        -------
        self : object
        """
        self.classifers_ = []
        rng = check_random_state(self.random_state)

        for _ in range(0, self.n_classifiers):
            cls = IndividualInceptionClassifier(
                nb_filters=self.nb_filters,
                nb_conv_per_layer=self.nb_conv_per_layer,
                kernel_size=self.kernel_size,
                use_max_pooling=self.use_max_pooling,
                max_pool_size=self.max_pool_size,
                strides=self.strides,
                dilation_rate=self.dilation_rate,
                padding=self.padding,
                activation=self.activation,
                use_bias=self.use_bias,
                use_residual=self.use_residual,
                use_bottleneck=self.use_bottleneck,
                depth=self.depth,
                use_custom_filters=self.use_custom_filters,
                file_path=self.file_path,
                batch_size=self.batch_size,
                use_mini_batch_size=self.use_mini_batch_size,
                n_epochs=self.n_epochs,
                callbacks=self.callbacks,
                loss=self.loss,
                metrics=self.metrics,
                optimizer=self.optimizer,
                random_state=rng.randint(0, np.iinfo(np.int32).max),
                verbose=self.verbose,
            )
            cls.fit(X, y)
            self.classifers_.append(cls)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict the labels of the test set using InceptionTime.

        Arguments:
        ---------

        X : np.ndarray of shape = (n_instances (n), n_channels (c), series_length (m))
            The testing input samples.

        Returns
        -------
        Y : np.ndarray of shape = (n_instances (n)), the predicted labels

        """
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def _predict_proba(self, X) -> np.ndarray:
        """Predict the proba of labels of the test set using InceptionTime.

        Arguments:
        ---------

        X : np.ndarray of shape = (n_instances (n), n_channels (c), series_length (m))
            The testing input samples.

        Returns
        -------
        Y : np.ndarray of shape = (n_instances (n), n_classes (c)), the predicted probs

        """
        probs = np.zeros((X.shape[0], self.n_classes_))

        for cls in self.classifers_:
            probs += cls._predict_proba(X)

        probs = probs / self.n_classifiers

        return probs

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
            "n_classifiers": 1,
            "n_epochs": 10,
            "batch_size": 4,
            "kernel_size": 4,
            "use_residual": False,
            "use_bottleneck": True,
        }

        return [param1]


class IndividualInceptionClassifier(BaseDeepClassifier):
    """Single InceptionTime classifier.

    Parameters
    ----------
        depth               : int, default = 6,
            the number of inception modules used
        nb_filters          : int or list of int32, default = 32,
            the number of filters used in one inception module, if not a list,
            the same number of filters is used in all inception modules
        nb_conv_per_layer   : int or list of int, default = 3,
            the number of convolution layers in each inception module, if not a list,
            the same number of convolution layers is used in all inception modules
        kernel_size         : int or list of int, default = 40,
            the head kernel size used for each inception module, if not a list,
            the same is used in all inception modules
        use_max_pooling     : bool or list of bool, default = True,
            conditioning wether or not to use max pooling layer
            in inception modules,if not a list,
            the same is used in all inception modules
        max_pool_size       : int or list of int, default = 3,
            the size of the max pooling layer, if not a list,
            the same is used in all inception modules
        strides             : int or list of int, default = 1,
            the strides of kernels in convolution layers for
            each inception module, if not a list,
            the same is used in all inception modules
        dilation_rate       : int or list of int, default = 1,
            the dilation rate of convolutions in each inception module, if not a list,
            the same is used in all inception modules
        padding             : str or list of str, default = "same",
            the type of padding used for convoltuon for each
            inception module, if not a list,
            the same is used in all inception modules
        activation          : str or list of str, default = "relu",
            the activation function used in each inception module, if not a list,
            the same is used in all inception modules
        use_bias            : bool or list of bool, default = False,
            conditioning wether or not convolutions should
            use bias values in each inception
            module, if not a list,
            the same is used in all inception modules
        use_residual        : bool, default = True,
            condition wether or not to use residual connections all over Inception
        use_bottleneck      : bool, default = True,
            confition wether or not to use bottlesnecks all over Inception
        bottleneck_size     : int, default = 32,
            the bottleneck size in case use_bottleneck = True
        use_custom_filters  : bool, default = True,
            condition on wether or not to use custom filters
            in the first inception module

        batch_size          : int, default = 64
            the number of samples per gradient update.
        use_mini_batch_size : bool, default = False
            condition on using the mini batch size formula Wang et al.
        n_epochs           : int, default = 1500
            the number of epochs to train the model.
        callbacks           : callable or None, default
        ReduceOnPlateau and ModelCheckpoint
            list of tf.keras.callbacks.Callback objects.
        file_path           : str, default = "./"
            file_path when saving model_Checkpoint callback
        random_state        : int, default = 0
            seed to any needed random actions.
        verbose             : boolean, default = False
            whether to output extra information
        optimizer           : keras optimizer, default = Adam
        loss                : keras loss, default = categorical_crossentropy
        metrics             : keras metrics, default = None, will be set
        to accuracy as default if None

    Notes
    -----
    ..[1] Fawaz et al. InceptionTime: Finding AlexNet for Time Series
    Classification, Data Mining and Knowledge Discovery, 34, 2020

    ..[2] Ismail-Fawaz et al. Deep Learning For Time Series Classification Using New
    Hand-Crafted Convolution Filters, 2022 IEEE International Conference on Big Data.

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py

    and Ismail-Fawaz et al.
    https://github.com/MSD-IRIMAS/CF-4-TSC
    """

    def __init__(
        self,
        nb_filters=32,
        nb_conv_per_layer=3,
        kernel_size=40,
        use_max_pooling=True,
        max_pool_size=3,
        strides=1,
        dilation_rate=1,
        padding="same",
        activation="relu",
        use_bias=False,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        use_custom_filters=True,
        file_path="./",
        batch_size=64,
        use_mini_batch_size=False,
        n_epochs=1500,
        callbacks=None,
        random_state=None,
        verbose=False,
        loss="categorical_crossentropy",
        metrics=None,
        optimizer=None,
    ):
        _check_dl_dependencies(severity="error")
        super(IndividualInceptionClassifier, self).__init__()
        # predefined
        self.nb_filters = nb_filters
        self.nb_conv_per_layer = nb_conv_per_layer
        self.use_max_pooling = use_max_pooling
        self.max_pool_size = max_pool_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size
        self.depth = depth
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.use_custom_filters = use_custom_filters

        self.file_path = file_path

        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose
        self.use_mini_batch_size = use_mini_batch_size
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        self._network = InceptionNetwork(
            nb_filters=self.nb_filters,
            nb_conv_per_layer=self.nb_conv_per_layer,
            kernel_size=self.kernel_size,
            use_max_pooling=self.use_max_pooling,
            max_pool_size=self.max_pool_size,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            padding=self.padding,
            activation=self.activation,
            use_bias=self.use_bias,
            use_residual=self.use_residual,
            use_bottleneck=self.use_bottleneck,
            bottleneck_size=self.bottleneck_size,
            depth=self.depth,
            use_custom_filters=self.use_custom_filters,
            random_state=self.random_state,
        )

    def build_model(self, input_shape, n_classes, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        n_classes: int
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

        tf.random.set_seed(self.random_state)

        if self.metrics is None:
            metrics = ["accuracy"]
        else:
            metrics = self.metrics

        self.optimizer_ = (
            tf.keras.optimizers.Adam() if self.optimizer is None else self.optimizer
        )

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=metrics,
        )

        return model

    def _fit(self, X, y):
        """
        Fit the classifier on the training set (X, y).

        Parameters
        ----------
        X : array-like of shape = (n_instances, n_channels, series_length)
            The training input samples. If a 2D array-like is passed,
            n_channels is assumed to be 1.
        y : array-like, shape = (n_instances)
            The training data class labels.

        Returns
        -------
        self : object
        """
        import tensorflow as tf

        y_onehot = self.convert_y_to_keras(y)
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        # ignore the number of instances, X.shape[0],
        # just want the shape of each instance
        self.input_shape = X.shape[1:]

        if self.use_mini_batch_size:
            mini_batch_size = int(min(X.shape[0] // 10, self.batch_size))
        else:
            mini_batch_size = self.batch_size
        self.training_model_ = self.build_model(self.input_shape, self.n_classes_)

        if self.verbose:
            self.training_model_.summary()

        self.file_name_ = str(np.random.randint(low=0, high=1e9, size=1)[0])

        self.callbacks_ = (
            [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=50, min_lr=0.0001
                ),
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
            batch_size=mini_batch_size,
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
            "kernel_size": 4,
            "use_residual": False,
            "use_bottleneck": True,
        }

        return [param1]
