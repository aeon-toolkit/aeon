"""InceptionTime and Inception classifiers."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["InceptionTimeClassifier"]

from aeon.classification.deep_learning._deep_classifier import DeepClassifier
from aeon.networks import InceptionNetwork


class InceptionTimeClassifier(DeepClassifier):
    """InceptionTime ensemble classifier.

    Ensemble of IndividualInceptionTimeClassifier objects, as described in [1]_.

    Parameters
    ----------
    n_classifiers : int, default = 5,
        the number of Inception models used for the
        Ensemble in order to create
        InceptionTime.
    depth : int, default = 6,
            the number of inception modules used
    n_filters : int or list of int32, default = 32,
        the number of filters used in one inception
        module, if not a list,
        the same number of filters is used in
        all inception modules
    n_conv_per_layer : int or list of int, default = 3,
        the number of convolution layers in each inception
        module, if not a list,
        the same number of convolution layers is used
        in all inception modules
    kernel_size : int or list of int, default = 40,
        the head kernel size used for each inception
        module, if not a list,
        the same is used in all inception modules
    use_max_pooling : bool or list of bool, default = True,
        conditioning whether or not to use max pooling layer
        in inception modules,if not a list,
        the same is used in all inception modules
    max_pool_size : int or list of int, default = 3,
        the size of the max pooling layer, if not a list,
        the same is used in all inception modules
    strides : int or list of int, default = 1,
        the strides of kernels in convolution layers for each
        inception module, if not a list,
        the same is used in all inception modules
    dilation_rate : int or list of int, default = 1,
        the dilation rate of convolutions in each inception
        module, if not a list,
        the same is used in all inception modules
    padding : str or list of str, default = "same",
        the type of padding used for convolution for each
        inception module, if not a list,
        the same is used in all inception modules
    activation : str or list of str, default = "relu",
        the activation function used in each inception
        module, if not a list,
        the same is used in all inception modules
    use_bias : bool or list of bool, default = False,
        conditioning whether or not convolutions should
        use bias values in each inception
        module, if not a list,
        the same is used in all inception modules
    use_residual : bool, default = True,
        condition whether or not to use residual
        connections all over Inception
    use_bottleneck : bool, default = True,
        condition whether or not to use bottlenecks
        all over Inception
    bottleneck_size : int, default = 32,
        the bottleneck size in case use_bottleneck = True
    use_custom_filters : bool, default = False,
        condition on whether or not to use custom
        filters in the first inception module
    batch_size : int, default = 64
        the number of samples per gradient update.
    use_mini_batch_size : bool, default = False
        condition on using the mini batch size
        formula Wang et al.
    n_epochs : int, default = 1500
        the number of epochs to train the model.
    callbacks : keras callback or list of callbacks,
        default = None
        The default list of callbacks are set to
        ModelCheckpoint and ReduceLROnPlateau.
    file_path : str, default = "./"
        file_path when saving model_Checkpoint callback
    save_best_model : bool, default = False
        Whether or not to save the best model, if the
        modelcheckpoint callback is used by default,
        this condition, if True, will prevent the
        automatic deletion of the best saved model from
        file and the user can choose the file name
    save_last_model : bool, default = False
        Whether or not to save the last model, last
        epoch trained, using the base class method
        save_last_model_to_file
    save_init_model : bool, default = False
        Whether to save the initialization of the  model.
    best_file_name : str, default = "best_model"
        The name of the file of the best model, if
        save_best_model is set to False, this parameter
        is discarded
    last_file_name : str, default = "last_model"
        The name of the file of the last model, if
        save_last_model is set to False, this parameter
        is discarded
    init_file_name : str, default = "init_model"
        The name of the file of the init model, if save_init_model is set to False,
        this parameter is discarded.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
        Seeded random number generation can only be guaranteed on CPU processing,
        GPU processing will be non-deterministic.
    verbose : boolean, default = False
        whether to output extra information
    optimizer : keras.optimizer, default = tf.keras.optimizers.Adam()
        The keras optimizer used for training.
    loss : str, default = "categorical_crossentropy"
        The name of the keras training loss.
    metrics : str or list[str], default="accuracy"
        The evaluation metrics to use during training. If
        a single string metric is provided, it will be
        used as the only metric. If a list of metrics are
        provided, all will be used for evaluation.

    Notes
    -----
    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py

    and Ismail-Fawaz et al.
    https://github.com/MSD-IRIMAS/CF-4-TSC

    References
    ----------
    ..[1] Fawaz et al. InceptionTime: Finding AlexNet for Time Series
    Classification, Data Mining and Knowledge Discovery, 34, 2020

    ..[2] Ismail-Fawaz et al. Deep Learning For Time Series
    Classification Using New
    Hand-Crafted Convolution Filters, 2022 IEEE International
    Conference on Big Data.

    Examples
    --------
    >>> from aeon.classification.deep_learning import InceptionTimeClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> inctime = InceptionTimeClassifier(n_epochs=20,batch_size=4)  # doctest: +SKIP
    >>> inctime.fit(X_train, y_train)  # doctest: +SKIP
    InceptionTimeClassifier(...)
    """

    _tags = {
        "python_dependencies": "tensorflow",
        "capability:multivariate": True,
        "non_deterministic": True,
        "cant_pickle": True,
        "algorithm_type": "deeplearning",
    }

    def __init__(
        self,
        n_classifiers: int = 5,
        n_filters: int | list[int] = 32,
        n_conv_per_layer: int | list[int] = 3,
        kernel_size: int | list[int] = 40,
        use_max_pooling: bool | list[bool] = True,
        max_pool_size: int | list[int] = 3,
        strides: int | list[int] = 1,
        dilation_rate: int | list[int] = 1,
        padding: str | list[str] = "same",
        activation: str | list[str] = "relu",
        use_bias: bool | list[bool] = False,
        use_residual: bool = True,
        use_bottleneck: bool = True,
        bottleneck_size: int = 32,
        depth: int = 6,
        use_custom_filters: bool = False,
        file_path: str = "./",
        save_last_model: bool = False,
        save_best_model: bool = False,
        save_init_model: bool = False,
        best_file_name: str = "best_model",
        last_file_name: str = "last_model",
        init_file_name: str = "init_model",
        batch_size: int = 64,
        use_mini_batch_size: bool = False,
        n_epochs: int = 1500,
        callbacks=None,
        random_state=None,
        verbose: bool = False,
        loss: str = "categorical_crossentropy",
        metrics: str | list[str] = "accuracy",
        optimizer=None,
    ):
        self.n_classifiers = n_classifiers
        self.n_filters = n_filters
        self.n_conv_per_layer = n_conv_per_layer
        self.kernel_size = kernel_size
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
        self.use_custom_filters = use_custom_filters

        network = InceptionNetwork(
            n_filters=n_filters,
            n_conv_per_layer=n_conv_per_layer,
            kernel_size=kernel_size,
            use_max_pooling=use_max_pooling,
            max_pool_size=max_pool_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            use_residual=use_residual,
            use_bottleneck=use_bottleneck,
            bottleneck_size=bottleneck_size,
            depth=depth,
            use_custom_filters=use_custom_filters,
        )

        super().__init__(
            network=network,
            n_epochs=n_epochs,
            batch_size=batch_size,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            ensemble_size=n_classifiers,
            callbacks=callbacks,
            use_mini_batch_size=use_mini_batch_size,
            file_path=file_path,
            save_best_model=save_best_model,
            save_last_model=save_last_model,
            save_init_model=save_init_model,
            best_file_name=best_file_name,
            last_file_name=last_file_name,
            init_file_name=init_file_name,
            random_state=random_state,
            verbose=verbose,
        )

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        return [
            {
                "n_classifiers": 1,
                "n_epochs": 10,
                "batch_size": 4,
                "kernel_size": 4,
                "use_residual": False,
                "depth": 1,
                "use_custom_filters": False,
            }
        ]


class IndividualInceptionClassifier(InceptionTimeClassifier):
    """Single InceptionTime classifier.

    Parameters
    ----------
        depth : int, default = 6,
            the number of inception modules used
        n_filters : int or list of int32, default = 32,
            the number of filters used in one inception module, if not a list,
            the same number of filters is used in all inception modules
        n_conv_per_layer : int or list of int, default = 3,
            the number of convolution layers in each inception module, if not a list,
            the same number of convolution layers is used in all inception modules
        kernel_size : int or list of int, default = 40,
            the head kernel size used for each inception module, if not a list,
            the same is used in all inception modules
        use_max_pooling : bool or list of bool, default = True,
            conditioning whether or not to use max pooling layer
            in inception modules,if not a list,
            the same is used in all inception modules
        max_pool_size : int or list of int, default = 3,
            the size of the max pooling layer, if not a list,
            the same is used in all inception modules
        strides : int or list of int, default = 1,
            the strides of kernels in convolution layers for
            each inception module, if not a list,
            the same is used in all inception modules
        dilation_rate : int or list of int, default = 1,
            the dilation rate of convolutions in each inception module, if not a list,
            the same is used in all inception modules
        padding : str or list of str, default = "same",
            the type of padding used for convolution for each
            inception module, if not a list,
            the same is used in all inception modules
        activation : str or list of str, default = "relu",
            the activation function used in each inception module, if not a list,
            the same is used in all inception modules
        use_bias : bool or list of bool, default = False,
            conditioning whether or not convolutions should
            use bias values in each inception
            module, if not a list,
            the same is used in all inception modules
        use_residual : bool, default = True,
            condition whether or not to use residual connections all over Inception
        use_bottleneck : bool, default = True,
            condition whether or not to use bottlenecks all over Inception
        bottleneck_size : int, default = 32,
            the bottleneck size in case use_bottleneck = True
        use_custom_filters : bool, default = False,
            condition on whether or not to use custom filters
            in the first inception module
        batch_size : int, default = 64
            the number of samples per gradient update.
        use_mini_batch_size : bool, default = False
            condition on using the mini batch size formula Wang et al.
        n_epochs : int, default = 1500
            the number of epochs to train the model.
        callbacks : keras callback or list of callbacks,
            default = None
            The default list of callbacks are set to
            ModelCheckpoint and ReduceLROnPlateau.
        file_path : str, default = "./"
            file_path when saving model_Checkpoint callback
        save_best_model : bool, default = False
            Whether or not to save the best model, if the
            modelcheckpoint callback is used by default,
            this condition, if True, will prevent the
            automatic deletion of the best saved model from
            file and the user can choose the file name
        save_last_model : bool, default = False
            Whether or not to save the last model, last
            epoch trained, using the base class method
            save_last_model_to_file
        save_init_model : bool, default = False
            Whether to save the initialization of the  model.
        best_file_name : str, default = "best_model"
            The name of the file of the best model, if
            save_best_model is set to False, this parameter
            is discarded.
        last_file_name : str, default = "last_model"
            The name of the file of the last model, if
            save_last_model is set to False, this parameter
            is discarded.
        init_file_name : str, default = "init_model"
            The name of the file of the init model, if save_init_model is set to False,
            this parameter is discarded.
        random_state : int, RandomState instance or None, default=None
            If `int`, random_state is the seed used by the random number generator;
            If `RandomState` instance, random_state is the random number generator;
            If `None`, the random number generator is the `RandomState` instance used
            by `np.random`.
            Seeded random number generation can only be guaranteed on CPU processing,
            GPU processing will be non-deterministic.
        verbose : boolean, default = False
            whether to output extra information
        optimizer : keras.optimizer, default = tf.keras.optimizers.Adam()
            The keras optimizer used for training.
        loss : str, default = "categorical_crossentropy"
            The name of the keras training loss.
        metrics : str or list[str], default="accuracy"
            The evaluation metrics to use during training. If
            a single string metric is provided, it will be
            used as the only metric. If a list of metrics are
            provided, all will be used for evaluation.

    Notes
    -----
    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py

    and Ismail-Fawaz et al.
    https://github.com/MSD-IRIMAS/CF-4-TSC

    References
    ----------
    ..[1] Fawaz et al. InceptionTime: Finding AlexNet for Time Series
    Classification, Data Mining and Knowledge Discovery, 34, 2020

    ..[2] Ismail-Fawaz et al. Deep Learning For Time Series Classification Using New
    Hand-Crafted Convolution Filters, 2022 IEEE International Conference on Big Data.

    Examples
    --------
    >>> from aeon.classification.deep_learning import IndividualInceptionClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> inc = IndividualInceptionClassifier(n_epochs=20,batch_size=4)  # doctest: +SKIP
    >>> inc.fit(X_train, y_train)  # doctest: +SKIP
    IndividualInceptionClassifier(...)
    """

    def __init__(
        self,
        n_filters: int | list[int] = 32,
        n_conv_per_layer: int | list[int] = 3,
        kernel_size: int | list[int] = 40,
        use_max_pooling: bool | list[bool] = True,
        max_pool_size: int | list[int] = 3,
        strides: int | list[int] = 1,
        dilation_rate: int | list[int] = 1,
        padding: str | list[str] = "same",
        activation: str | list[str] = "relu",
        use_bias: bool | list[bool] = False,
        use_residual: bool = True,
        use_bottleneck: bool = True,
        bottleneck_size: int = 32,
        depth: int = 6,
        use_custom_filters: bool = False,
        file_path: str = "./",
        save_best_model: bool = False,
        save_last_model: bool = False,
        save_init_model: bool = False,
        best_file_name: str = "best_model",
        last_file_name: str = "last_model",
        init_file_name: str = "init_model",
        batch_size: int = 64,
        use_mini_batch_size: bool = False,
        n_epochs: int = 1500,
        callbacks=None,
        random_state=None,
        verbose: bool = False,
        loss: str = "categorical_crossentropy",
        metrics: str | list[str] = "accuracy",
        optimizer=None,
    ):
        super().__init__(
            n_classifiers=1,
            n_filters=n_filters,
            n_conv_per_layer=n_conv_per_layer,
            kernel_size=kernel_size,
            use_max_pooling=use_max_pooling,
            max_pool_size=max_pool_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            use_residual=use_residual,
            use_bottleneck=use_bottleneck,
            bottleneck_size=bottleneck_size,
            depth=depth,
            use_custom_filters=use_custom_filters,
            file_path=file_path,
            save_last_model=save_last_model,
            save_best_model=save_best_model,
            save_init_model=save_init_model,
            best_file_name=best_file_name,
            last_file_name=last_file_name,
            init_file_name=init_file_name,
            batch_size=batch_size,
            use_mini_batch_size=use_mini_batch_size,
            n_epochs=n_epochs,
            callbacks=callbacks,
            random_state=random_state,
            verbose=verbose,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
        )

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        return [
            {
                "n_epochs": 10,
                "batch_size": 4,
                "kernel_size": 4,
                "use_residual": False,
                "use_bottleneck": True,
                "depth": 1,
                "use_custom_filters": False,
            }
        ]
