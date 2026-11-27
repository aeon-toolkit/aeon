"""LITETime and LITE classifiers."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["LITETimeClassifier"]


from aeon.classification.deep_learning._deep_classifier import DeepClassifier
from aeon.networks import LITENetwork


class LITETimeClassifier(DeepClassifier):
    """LITETime or LITEMVTime ensemble classifier.

    Ensemble of IndividualLITETimeClassifier objects, as described in [1]_
    and [2]_. For using LITEMV, simply set the `use_litemv`
    bool parameter to True.

    Parameters
    ----------
    n_classifiers : int, default = 5,
        the number of LITE or LITEMV models used for the
        Ensemble in order to create
        LITETime or LITEMVTime.
    use_litemv : bool, default = False
        The boolean value to control which version of the
        network to use. If set to `False`, then LITE is used,
        if set to `True` then LITEMV is used. LITEMV is the
        same architecture as LITE but specifically designed
        to better handle multivariate time series.
    n_filters : int, default = 32
        The number of filters used in one lite layer.
    kernel_size : int, default = 40
        The head kernel size used for each lite layer.
    strides : int or list of int, default = 1
        The strides of kernels in convolution layers for each lite layer,
        if not a list, the same is used in all lite layers.
    activation : str or list of str, default = 'relu'
        The activation function used in each lite layer, if not a list,
        the same is used in all lite layers.
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
        model checkpoint callback is used by default,
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
    metrics : str or list[str|function|keras.metrics.Metric], default="accuracy"
        the evaluation metrics to use during training. each of this can be a
        string, function or a keras.metrics.metric instance (for details, see
        https://keras.io/api/metrics/).
        if a single string metric is provided, it will be used as the only
        metric. if a list of metrics are provided, all will be used for
        evaluation.

    References
    ----------
    ..[1] Ismail-Fawaz et al. LITE: Light Inception with boosTing
    tEchniques for Time Series Classification, IEEE International
    Conference on Data Science and Advanced Analytics, 2023.
    ..[2] Ismail-Fawaz, Ali, et al. "Look Into the LITE
    in Deep Learning for Time Series Classification."
    arXiv preprint arXiv:2409.02869 (2024).

    Notes
    -----
    Adapted from the implementation from Ismail-Fawaz et. al
    https://github.com/MSD-IRIMAS/LITE

    Examples
    --------
    >>> from aeon.classification.deep_learning import LITETimeClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> ltime = LITETimeClassifier(n_epochs=20,batch_size=4)  # doctest: +SKIP
    >>> ltime.fit(X_train, y_train)  # doctest: +SKIP
    LITETimeClassifier(...)
    """

    def __init__(
        self,
        n_classifiers: int = 5,
        use_litemv: bool = False,
        n_filters: int = 32,
        kernel_size: int = 40,
        strides: int | list[int] = 1,
        activation: str | list[str] = "relu",
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
        self.use_litemv = use_litemv
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation

        network = LITENetwork(
            use_litemv=use_litemv,
            n_filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
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
                "n_epochs": 2,
                "batch_size": 4,
                "kernel_size": 4,
            },
            {
                "n_classifiers": 1,
                "use_litemv": True,
                "n_epochs": 2,
                "batch_size": 4,
                "kernel_size": 4,
                "metrics": ["accuracy"],
                "verbose": True,
                "use_mini_batch_size": True,
            },
        ]


class IndividualLITEClassifier(LITETimeClassifier):
    """Single LITE or LITEMV classifier.

    One LITE or LITEMV deep model, as described in [1]_
    and [2]_. For using LITEMV, simply set the `use_litemv`
    bool parameter to True.

    Parameters
    ----------
    use_litemv : bool, default = False
        The boolean value to control which version of the
        network to use. If set to `False`, then LITE is used,
        if set to `True` then LITEMV is used. LITEMV is the
        same architecture as LITE but specifically designed
        to better handle multivariate time series.
    n_filters : int, default = 32
        The number of filters used in one lite layer.
    kernel_size : int, default = 40
        The head kernel size used for each lite layer.
    strides : int or list of int, default = 1
        The strides of kernels in convolution layers for each lite layer,
        if not a list, the same is used in all lite layers.
    activation : str or list of str, default = 'relu'
        The activation function used in each lite layer, if not a list,
        the same is used in all lite layers.
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
        model checkpoint callback is used by default,
        this condition, if True, will prevent the
        automatic deletion of the best saved model from
        file and the user can choose the file name
    save_last_model     : bool, default = False
        Whether or not to save the last model, last
        epoch trained, using the base class method
        save_last_model_to_file
    save_init_model : bool, default = False
        Whether to save the initialization of the  model.
    best_file_name      : str, default = "best_model"
        The name of the file of the best model, if
        save_best_model is set to False, this parameter
        is discarded
    last_file_name      : str, default = "last_model"
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
    metrics : str or list[str|function|keras.metrics.Metric], default="accuracy"
        the evaluation metrics to use during training. each of this can be a
        string, function or a keras.metrics.metric instance (for details, see
        https://keras.io/api/metrics/).
        if a single string metric is provided, it will be used as the only
        metric. if a list of metrics are provided, all will be used for
        evaluation.

    References
    ----------
    ..[1] Ismail-Fawaz et al. LITE: Light Inception with boosTing
    tEchniques for Time Series Classification, IEEE International
    Conference on Data Science and Advanced Analytics, 2023.
    ..[2] Ismail-Fawaz, Ali, et al. "Look Into the LITE
    in Deep Learning for Time Series Classification."
    arXiv preprint arXiv:2409.02869 (2024).

    Notes
    -----
    Adapted from the implementation from Ismail-Fawaz et. al
    https://github.com/MSD-IRIMAS/LITE

    Examples
    --------
    >>> from aeon.classification.deep_learning import IndividualLITEClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> lite = IndividualLITEClassifier(n_epochs=20,batch_size=4)  # doctest: +SKIP
    >>> lite.fit(X_train, y_train)  # doctest: +SKIP
    IndividualLITEClassifier(...)
    """

    def __init__(
        self,
        use_litemv: bool = False,
        n_filters: int = 32,
        kernel_size: int = 40,
        strides: int | list[int] = 1,
        activation: str | list[str] = "relu",
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
            use_litemv=use_litemv,
            n_filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
            file_path=file_path,
            save_best_model=save_best_model,
            save_last_model=save_last_model,
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
    def _get_test_params(cls, parameter_set: str = "default") -> dict | list[dict]:
        return [
            {
                "n_epochs": 2,
                "batch_size": 4,
                "kernel_size": 4,
            },
            {
                "use_litemv": True,
                "n_epochs": 2,
                "batch_size": 4,
                "kernel_size": 4,
                "metrics": ["accuracy"],
                "verbose": True,
                "use_mini_batch_size": True,
            },
        ]
