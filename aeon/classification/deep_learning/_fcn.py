"""Fully Convolutional Network (FCN) classifier."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["FCNClassifier"]

from aeon.classification.deep_learning._deep_classifier import DeepClassifier
from aeon.networks import FCNNetwork


class FCNClassifier(DeepClassifier):
    """Fully Convolutional Network (FCN).

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    n_layers : int, default = 3
        Number of convolution layers.
    n_filters : int or list of int, default = None
        Number of filters used in convolution layers.
        If set to None, defaults to [128, 256, 128].
    kernel_size : int or list of int, default = None
        Size of convolution kernel. If set to None,
        defaults to [8,5,3].
    dilation_rate : int or list of int, default = 1
        The dilation rate for convolution.
    strides : int or list of int, default = 1
        The strides of the convolution filter.
    padding : str or list of str, default = "same"
        The type of padding used for convolution.
    activation : str or list of str, default = "relu"
        Activation used after the convolution.
    use_bias : bool or list of bool, default = True
        Whether or not to use bias in convolution.
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
    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py

    References
    ----------
    .. [1] Z. Wang, W. Yan and T. Oates, "Time series classification from scratch
    with deep neural networks: A strong baseline," 2017 International Joint Conference
    on Neural Networks (IJCNN), Anchorage, AK, USA, 2017, pp. 1578-1585.

    Examples
    --------
    >>> from aeon.classification.deep_learning import FCNClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> fcn = FCNClassifier(n_epochs=20, batch_size=4)  # doctest: +SKIP
    >>> fcn.fit(X_train, y_train)  # doctest: +SKIP
    FCNClassifier(...)
    """

    def __init__(
        self,
        n_layers: int = 3,
        n_filters: int | list[int] = None,
        kernel_size: int | list[int] = None,
        dilation_rate: int | list[int] = 1,
        strides: int | list[int] = 1,
        padding: str | list[str] = "same",
        activation: str | list[str] = "relu",
        file_path: str = "./",
        save_best_model: bool = False,
        save_last_model: bool = False,
        save_init_model: bool = False,
        best_file_name: str = "best_model",
        last_file_name: str = "last_model",
        init_file_name: str = "init_model",
        n_epochs: int = 2000,
        batch_size: int = 16,
        use_mini_batch_size: bool = False,
        callbacks=None,
        verbose: bool = False,
        loss: str = "categorical_crossentropy",
        metrics: str | list[str] = "accuracy",
        random_state=None,
        use_bias: bool | list[bool] = True,
        optimizer=None,
    ):
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias

        network = FCNNetwork(
            n_layers=n_layers,
            n_filters=n_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            strides=strides,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
        )

        super().__init__(
            network=network,
            n_epochs=n_epochs,
            batch_size=batch_size,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
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
                "n_epochs": 10,
                "batch_size": 4,
                "use_bias": False,
                "n_layers": 1,
                "n_filters": 4,
                "kernel_size": 3,
                "padding": "valid",
                "strides": 2,
            }
        ]
