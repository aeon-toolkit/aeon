# -*- coding: utf-8 -*-
"""InceptionTime classifier."""

__author__ = ["James-Large", "TonyBagnall", "MatthewMiddlehurst"]

import numpy as np
from sklearn.utils import check_random_state
# from sktime.classification.base import BaseClassifier
from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.utils.validation._dependencies import _check_dl_dependencies

from sktime.networks.inception import InceptionNetwork

_check_dl_dependencies(severity="warning")


class InceptionTimeClassifier:
    """InceptionTime ensemble classifier.

    Ensemble of IndividualInceptionTimeClassifiers, as desribed in [1].

    Parameters
    ----------
    n_classifiers=5,
    nb_filters: int,
    use_residual: boolean,
    use_bottleneck: boolean,
    depth: int
    kernel_size: int, specifying the length of the 1D convolution
     window
    batch_size: int, the number of samples per gradient update.
    bottleneck_size: int,
    nb_epochs: int, the number of epochs to train the model
    callbacks: list of tf.keras.callbacks.Callback objects
    random_state: int, seed to any needed random actions
    verbose: boolean, whether to output extra information
    model_name: string, the name of this model for printing and
     file writing purposes
    model_save_directory: string, if not None; location to save
     the trained keras model in hdf5 format

    Notes
    -----
    ..[1] Fawaz et. al, InceptionTime: Finding AlexNet for Time Series
    Classification, Data Mining and Knowledge Discovery, 34, 2020

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
    """

    _tags = {"capability:multivariate": True}

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
        padding='same',
        activation='relu',
        use_bias=False,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        file_path='./',
        
        batch_size=64,
        use_mini_batch_size=False,
        nb_epochs=1500,
        callbacks=None,
        random_state=None,
        verbose=False,
        loss='categorical_crossentropy',
        metrics=None,
        optimizer=None
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
        self.nb_epochs = nb_epochs

        self.file_path = file_path

        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose
        self.use_mini_batch_size = use_mini_batch_size
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        self.classifers_ = []

        # super(InceptionTimeClassifier, self).__init__()

    def fit(self, X, y):
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
                file_path=self.file_path,
                batch_size=self.batch_size,
                use_mini_batch_size=self.use_mini_batch_size,
                nb_epochs=self.nb_epochs,
                callbacks=self.callbacks,
                loss=self.loss,
                metrics=self.metrics,
                optimizer=self.optimizer,
                random_state=rng.randint(0, np.iinfo(np.int32).max),
                verbose=self.verbose,
            )
            cls.fit(X, y)
            self.classifers_.append(cls)

            self.classes_ = cls.classes_
            self.n_classes_ = cls.n_classes_

        return self

    def predict(self, X) -> np.ndarray:
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def predict_proba(self, X) -> np.ndarray:
        probs = np.zeros((X.shape[0], self.n_classes_))

        for cls in self.classifers_:
            probs += cls._predict_proba(X)

        probs = probs / self.n_classifiers

        return probs

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
            "n_classifiers": 2,
            "n_epochs": 10,
            "batch_size": 4,
            "kernel_size": 4,
            "use_residual": False,
            "use_bottleneck": True,
        }

        param2 = {
            "n_classifiers": 3,
            "n_epochs": 12,
            "batch_size": 6,
            "use_bias": True,
        }

        return [param1, param2]


class IndividualInceptionClassifier(BaseDeepClassifier):
    """Single InceptionTime classifier.

    Parameters
    ----------
    nb_filters: int, default = 32
    use_residual: boolean, default = True
    use_bottleneck: boolean, default = True
    bottleneck_size: int, default = 32
    depth: int, default = 6
    kernel_size: int, default = 40
        specifies the length of the 1D convolution window.
    batch_size: int, default = 64
        the number of samples per gradient update.
    nb_epochs: int, default = 1500
        the number of epochs to train the model.
    callbacks: callable or None, default None
        list of tf.keras.callbacks.Callback objects.
    random_state: int, default = 0
        seed to any needed random actions.
    verbose: boolean, default = False
        whether to output extra information

    Notes
    -----
    ..[1] Fawaz et. al, InceptionTime: Finding AlexNet for Time Series
    Classification, Data Mining and Knowledge Discovery, 34, 2020

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
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
        padding='same',
        activation='relu',
        use_bias=False,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        file_path='./',
        
        batch_size=64,
        use_mini_batch_size=False,
        nb_epochs=1500,
        callbacks=None,
        random_state=None,
        verbose=False,
        loss='categorical_crossentropy',
        metrics=None,
        optimizer=None
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
        self.nb_epochs = nb_epochs

        self.file_path = file_path

        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose
        self.use_mini_batch_size = use_mini_batch_size
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        self._inception_network = InceptionNetwork(
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
            random_state=self.random_state
        )

    def build_model(self, input_shape, n_classes, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        nb_classes: int
            The number of classes, which shall become the size of the output
             layer

        Returns
        -------
        output : a compiled Keras Model
        """
        import tensorflow as tf

        input_layer, output_layer = self._inception_network.build_network(input_shape, **kwargs)

        output_layer = tf.keras.layers.Dense(n_classes, activation="softmax")(output_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        tf.random.set_seed(self.random_state)

        if self.metrics is None:
            metrics = ["accuracy"]
        else:
            metrics = self.metrics
        
        self.optimizer_ = (
            tf.keras.optimizers.Adam()
            if self.optimizer is None
            else self.optimizer
        )

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=metrics,
        )

        # if user hasn't provided a custom ReduceLROnPlateau via init already,
        # add the default from literature
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=self.file_path+'best_model.hdf5',
                                        monitor='loss', save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                                 patience=50, min_lr=0.0001)
            if self.callbacks is None
            else self.callbacks
        ]

        # if not any(
        #     isinstance(callback, keras.callbacks.ReduceLROnPlateau)
        #     for callback in self.callbacks
        # ):
        #     reduce_lr = keras.callbacks.ReduceLROnPlateau(
        #         monitor="loss", factor=0.5, patience=50, min_lr=0.0001
        #     )
        #     self.callbacks.append(reduce_lr)

        return model

    def _fit(self, X, y):
        """
        Fit the classifier on the training set (X, y).

        Parameters
        ----------
        X : array-like of shape = (n_instances, n_dimensions, series_length)
            The training input samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
        y : array-like, shape = [n_instances]
            The training data class labels.
        input_checks : boolean
            whether to check the X and y parameters
        validation_X : a nested pd.Dataframe, or array-like of shape =
        (n_instances, series_length, n_dimensions)
            The validation samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
            Unless strictly defined by the user via callbacks (such as
            EarlyStopping), the presence or state of the validation
            data does not alter training in any way. Predictions at each epoch
            are stored in the model's fit history.
        validation_y : array-like, shape = [n_instances]
            The validation class labels.

        Returns
        -------
        self : object
        """
        # self.random_state = check_random_state(self.random_state)
        
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
        self.model_ = self.build_model(self.input_shape, self.n_classes_)

        if self.verbose:
            self.model_.summary()

        self.history = self.model_.fit(
            X,
            y_onehot,
            batch_size=mini_batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
        )

        #        self.save_trained_model()
        #        self._is_fitted = True

        try:
            import tensorflow as tf
            import os

            self.model_ = tf.keras.models.load_model(self.file_path+'best_model.hdf5', compile=False)
            os.remove(self.file_path+'best_model.hdf5')

            return self
        except:
            return self

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

        param2 = {
            "n_epochs": 12,
            "batch_size": 6,
            "use_bias": True,
        }

        return [param1, param2]
