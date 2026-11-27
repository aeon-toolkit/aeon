"""Time Convolutional Neural Network (CNN) classifier."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["TimeCNNClassifier"]

from aeon.classification.deep_learning._deep_classifier import DeepClassifier
from aeon.networks import TimeCNNNetwork


class TimeCNNClassifier(DeepClassifier):
    """
    Time Convolutional Neural Network (CNN).

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    n_layers : int, default = 2
        The number of convolution layers in the network.
    kernel_size : int or list of int, default = 7
        Kernel size of convolution layers, if not a list, the same kernel size
        is used for all layer, len(list) should be n_layers.
    n_filters : int or list of int, default = None
        Number of filters for each convolution layer, if not a list, the same n_filters
        is used in all layers. If set to None, defaults to [6, 12].
    avg_pool_size : int or list of int, default = 3
        The size of the average pooling layer, if not a list, the same
        max pooling size is used for all convolution layer.
    activation : str or list of str, default = "sigmoid"
        Keras activation function used in the model for each layer, if not a list,
        the same activation is used for all layers.
    padding : str or list of str, default = 'valid'
        The method of padding in convolution layers, if not a list, the same padding
        used for all convolution layers.
    strides : int or list of int, default = 1
        The strides of kernels in the convolution and max pooling layers, if not a
        list, the same strides are used for all layers.
    strides_pooling : int or list of int, default = None
        Strides for the pooling layers. If None, defaults to pool_size.
        If not a list, the same strides are used for all pooling layers.
    dilation_rate : int or list of int, default = 1
        The dilation rate of the convolution layers, if not a list, the same dilation
        rate is used all over the network.
    use_bias : bool or list of bool, default = True
        Condition on whether to use bias values for convolution layers,
        if not a list, the same condition is used for all layers.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
        Seeded random number generation can only be guaranteed on CPU processing,
        GPU processing will be non-deterministic.
    n_epochs : int, default = 2000
        The number of epochs to train the model.
    batch_size : int, default = 16
        The number of samples per gradient update.
    verbose : boolean, default = False
        Whether to output extra information.
    loss : str, default = "mean_squared_error"
        The name of the keras training loss.
    optimizer : keras.optimizer, default = tf.keras.optimizers.Adam()
        The keras optimizer used for training.
    metrics : str or list[str], default="accuracy"
        The evaluation metrics to use during training. If
        a single string metric is provided, it will be
        used as the only metric. If a list of metrics are
        provided, all will be used for evaluation.
    callbacks : keras callback or list of callbacks,
        default = None
        The default list of callbacks are set to
        ModelCheckpoint.
    file_path : file_path for the best model
        Only used if checkpoint is used as callback.
    save_best_model : bool, default = False
        Whether to save the best model, if the modelcheckpoint callback is used by
        default, this condition, if True, will prevent the automatic deletion of the
        best saved model from file and the user can choose the file name.
    save_last_model : bool, default = False
        Whether to save the last model, last epoch trained, using the base class method
        save_last_model_to_file.
    save_init_model : bool, default = False
        Whether to save the initialization of the  model.
    best_file_name : str, default = "best_model"
        The name of the file of the best model, if save_best_model is set to False,
        this parameter is discarded.
    last_file_name : str, default = "last_model"
        The name of the file of the last model, if save_last_model is set to False,
        this parameter is discarded.
    init_file_name : str, default = "init_model"
        The name of the file of the init model, if save_init_model is set to False,
        this parameter is discarded.

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
    >>> from aeon.classification.deep_learning import TimeCNNClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> cnn = TimeCNNClassifier(n_epochs=20, batch_size=4)  # doctest: +SKIP
    >>> cnn.fit(X_train, y_train)  # doctest: +SKIP
    TimeCNNClassifier(...)
    """

    def __init__(
        self,
        n_layers: int = 2,
        kernel_size: int | list[int] = 7,
        n_filters: int | list[int] = None,
        avg_pool_size: int | list[int] = 3,
        activation: str | list[str] = "sigmoid",
        padding: str | list[str] = "valid",
        strides: int | list[int] = 1,
        strides_pooling: int | list[int] = None,
        dilation_rate: int | list[int] = 1,
        n_epochs: int = 2000,
        batch_size: int = 16,
        callbacks=None,
        file_path="./",
        save_best_model: bool = False,
        save_last_model: bool = False,
        save_init_model: bool = False,
        best_file_name: str = "best_model",
        last_file_name: str = "last_model",
        init_file_name: str = "init_model",
        verbose: bool = False,
        loss: str = "mean_squared_error",
        metrics: str | list[str] = "accuracy",
        random_state=None,
        use_bias: bool | list[bool] = True,
        optimizer=None,
    ):
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.padding = padding
        self.strides = strides
        self.strides_pooling = strides_pooling
        self.dilation_rate = dilation_rate
        self.avg_pool_size = avg_pool_size
        self.activation = activation
        self.use_bias = use_bias

        network = TimeCNNNetwork(
            n_layers=n_layers,
            kernel_size=kernel_size,
            n_filters=n_filters,
            avg_pool_size=avg_pool_size,
            activation=activation,
            padding=padding,
            strides=strides,
            strides_pooling=strides_pooling,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
        )

        super().__init__(
            network=network,
            n_epochs=n_epochs,
            batch_size=batch_size,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            # sigmoid used in inner layers seems bad, and
            # output activation should be softmax for classification
            # implemented the same way as in the old code, but is this correct ?
            output_activation=activation,
            verbose=verbose,
            random_state=random_state,
            callbacks=callbacks,
            use_mini_batch_size=False,
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
                "n_epochs": 10,
                "batch_size": 4,
                "avg_pool_size": 4,
            }
        ]
