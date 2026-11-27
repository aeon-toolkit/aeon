"""DisjointCNN classifier."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["DisjointCNNClassifier"]

from aeon.classification.deep_learning._deep_classifier import DeepClassifier
from aeon.networks import DisjointCNNNetwork


class DisjointCNNClassifier(DeepClassifier):
    """Disjoint Convolutional Neural Network classifier.

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    n_layers : int, default = 4
        Number of 1+1D Convolution layers.
    n_filters : int or list of int, default = 64
        Number of filters used in convolution layers. If
        input is set to a list, the length should be the same
        as `n_layers`, if input is int the a list of the same
        element is created of length `n_layers`.
    kernel_size : int or list of int, default = None
        Size of convolution kernel. If
        input is set to a list, the length should be the same
        as `n_layers`, if input is int the a list of the same
        element is created of length `n_layers`. If set to None,
        defaults to [8, 5, 5, 3].
    dilation_rate : int or list of int, default = 1
        The dilation rate for convolution. If
        input is set to a list, the length should be the same
        as `n_layers`, if input is int the a list of the same
        element is created of length `n_layers`.
    strides : int or list of int, default = 1
        The strides of the convolution filter. If
        input is set to a list, the length should be the same
        as `n_layers`, if input is int the a list of the same
        element is created of length `n_layers`.
    padding : str or list of str, default = "same"
        The type of padding used for convolution. If
        input is set to a list, the length should be the same
        as `n_layers`, if input is int the a list of the same
        element is created of length `n_layers`.
    activation : str or list of str, default = "elu"
        Activation used after the convolution. If
        input is set to a list, the length should be the same
        as `n_layers`, if input is int the a list of the same
        element is created of length `n_layers`.
    use_bias : bool or list of bool, default = True
        Whether or not to use bias in convolution. If
        input is set to a list, the length should be the same
        as `n_layers`, if input is int the a list of the same
        element is created of length `n_layers`.
    kernel_initializer: str or list of str, default = "he_uniform"
        The initialization method of convolution layers. If
        input is set to a list, the length should be the same
        as `n_layers`, if input is int the a list of the same
        element is created of length `n_layers`.
    pool_size: int, default = 5
        The size of the one max pool layer at the end of
        the model, default = 5.
    pool_strides: int, default = None
        The strides used for the one max pool layer at
        the end of the model, default = None.
    pool_padding: str, default = "valid"
        The padding method for the one max pool layer at
        the end of the model, default = "valid".
    hidden_fc_units: int, default = 128
        The number of fully connected units.
    activation_fc: str, default = "relu"
        The activation of the fully connected layer.
    n_epochs : int, default = 2000
        The number of epochs to train the model.
    batch_size : int, default = 16
        The number of samples per gradient update.
    use_mini_batch_size : bool, default = False
        Whether or not to use the mini batch size formula.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
        Seeded random number generation can only be guaranteed on CPU processing,
        GPU processing will be non-deterministic.
    verbose : boolean, default = False
        Whether to output extra information.
    loss : str, default = "categorical_crossentropy"
        The name of the keras training loss.
    metrics : str or list[str], default="accuracy"
        The evaluation metrics to use during training. If
        a single string metric is provided, it will be
        used as the only metric. If a list of metrics are
        provided, all will be used for evaluation.
    optimizer : keras.optimizer, default = tf.keras.optimizers.Adam()
        The keras optimizer used for training.
    file_path : str, default = "./"
        File path to save best model.
    save_best_model : bool, default = False
        Whether or not to save the best model, if the
        modelcheckpoint callback is used by default,
        this condition, if True, will prevent the
        automatic deletion of the best saved model from
        file and the user can choose the file name.
    save_last_model : bool, default = False
        Whether or not to save the last model, last
        epoch trained, using the base class method
        save_last_model_to_file.
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
        The name of the file of the init model, if
        save_init_model is set to False,
        this parameter is discarded.
    callbacks : keras callback or list of callbacks,
        default = None
        The default list of callbacks are set to
        ModelCheckpoint and ReduceLROnPlateau.

    Notes
    -----
    Adapted from the implementation from:
    https://github.com/Navidfoumani/Disjoint-CNN

    References
    ----------
    .. [1] Foumani, Seyed Navid Mohammadi, Chang Wei Tan, and Mahsa Salehi.
    "Disjoint-cnn for multivariate time series classification."
    2021 International Conference on Data Mining Workshops
    (ICDMW). IEEE, 2021.

    Examples
    --------
    >>> from aeon.classification.deep_learning import DisjointCNNClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> disjoint_cnn = DisjointCNNClassifier(n_epochs=20,
    ... batch_size=4)  # doctest: +SKIP
    >>> disjoint_cnn.fit(X_train, y_train)  # doctest: +SKIP
    DisjointCNNClassifier(...)
    """

    def __init__(
        self,
        n_layers: int = 4,
        n_filters: int | list[int] = 64,
        kernel_size: int | list[int] = None,
        dilation_rate: int | list[int] = 1,
        strides: int | list[int] = 1,
        padding: str | list[str] = "same",
        activation: str | list[str] = "elu",
        use_bias: bool | list[bool] = True,
        kernel_initializer: str | list[str] = "he_uniform",
        pool_size: int = 5,
        pool_strides: int = None,
        pool_padding: str = "valid",
        hidden_fc_units: int = 128,
        activation_fc: str = "relu",
        n_epochs: int = 2000,
        batch_size: int = 16,
        use_mini_batch_size: bool = False,
        random_state=None,
        verbose: bool = False,
        loss: str = "categorical_crossentropy",
        metrics: str | list[str] = "accuracy",
        optimizer=None,
        file_path: str = "./",
        save_best_model: bool = False,
        save_last_model: bool = False,
        save_init_model: bool = False,
        best_file_name: str = "best_model",
        last_file_name: str = "last_model",
        init_file_name: str = "init_model",
        callbacks=None,
    ):
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.pool_padding = pool_padding
        self.hidden_fc_units = hidden_fc_units
        self.activation_fc = activation_fc

        network = DisjointCNNNetwork(
            n_layers=n_layers,
            n_filters=n_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            strides=strides,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            pool_size=pool_size,
            pool_strides=pool_strides,
            pool_padding=pool_padding,
            hidden_fc_units=hidden_fc_units,
            activation_fc=activation_fc,
        )

        _optimizer = optimizer
        if _optimizer is None:
            import tensorflow as tf

            _optimizer = tf.keras.optimizers.Adam()

        super().__init__(
            network=network,
            n_epochs=n_epochs,
            batch_size=batch_size,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            verbose=verbose,
            random_state=random_state,
            callbacks=callbacks,
            use_mini_batch_size=use_mini_batch_size,
            file_path=file_path,
            save_best_model=save_best_model,
            save_last_model=save_last_model,
            save_init_model=save_init_model,
            best_file_name=best_file_name,
            last_file_name=last_file_name,
            init_file_name=init_file_name,
        )

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> dict | list[dict]:
        return [
            {
                "n_epochs": 3,
                "batch_size": 4,
                "use_bias": False,
                "n_layers": 2,
                "n_filters": 2,
                "kernel_size": [2, 2],
            },
            {
                "n_epochs": 3,
                "batch_size": 4,
                "use_bias": False,
                "n_layers": 2,
                "n_filters": 2,
                "kernel_size": [2, 2],
                "verbose": True,
                "metrics": ["accuracy"],
                "use_mini_batch_size": True,
            },
        ]
