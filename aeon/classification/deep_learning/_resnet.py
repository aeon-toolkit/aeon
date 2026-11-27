"""Residual Network (ResNet) classifier."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["ResNetClassifier"]

from aeon.classification.deep_learning._deep_classifier import DeepClassifier
from aeon.networks import ResNetNetwork


class ResNetClassifier(DeepClassifier):
    """
    Residual Neural Network (RNN).

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    n_residual_blocks : int, default = 3
        The number of residual blocks of ResNet's model.
    n_conv_per_residual_block : int, default = 3
        The number of convolution blocks in each residual block.
    n_filters : int or list of int, default = None
        The number of convolution filters for all the convolution layers in the same
        residual block, if not a list, the same number of filters is used in all
        convolutions of all residual blocks.
        If set to None, defaults to [128, 64, 64].
    kernel_sizes : int or list of int, default = None
        The kernel size of all the convolution layers in one residual block, if not
        a list, the same kernel size is used in all convolution layers.
        If set to None, defaults to [8, 5, 3].
    strides : int or list of int, default = 1
        The strides of convolution kernels in each of the convolution layers in
        one residual block, if not a list, the same kernel size is used in all
        convolution layers.
    dilation_rate : int or list of int, default = 1
        The dilation rate of the convolution layers in one residual block, if not
        a list, the same kernel size is used in all convolution layers.
    padding : str or list of str, default = 'padding'
        The type of padding used in the convolution layers in one residual block, if
        not a list, the same kernel size is used in all convolution layers.
    activation : str or list of str, default = 'relu'
        keras activation used in the convolution layers in one residual block,
        if not a list, the same kernel size is used in all convolution layers.
    use_bias : bool or list of bool, default = True
        Condition on whether or not to use bias values in the convolution layers
        in one residual block, if not a list, the same kernel size is used in all
        convolution layers.
    n_epochs : int, default = 1500
        The number of epochs to train the model.
    batch_size : int, default = 64
        The number of samples per gradient update.
    use_mini_batch_size : bool, default = False
        Condition on using the mini batch size formula Wang et al.
    callbacks : callable or None, default ReduceOnPlateau and ModelCheckpoint
        List of tf.keras.callbacks.Callback objects.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
        Seeded random number generation can only be guaranteed on CPU processing,
        GPU processing will be non-deterministic.
    file_path : str, default = './'
        file_path when saving model_Checkpoint callback.
    save_best_model : bool, default = False
        Whether or not to save the best model, if the modelcheckpoint callback is
        used by default, this condition, if True, will prevent the automatic
        deletion of the best saved model from file and the user can choose the
        file name.
    save_last_model : bool, default = False
        Whether or not to save the last model, last epoch trained, using the base
        class method save_last_model_to_file.
    save_init_model : bool, default = False
        Whether to save the initialization of the  model.
    best_file_name : str, default = "best_model"
        The name of the file of the best model, if save_best_model is set to
        False, this parameter is discarded.
    last_file_name : str, default = "last_model"
        The name of the file of the last model, if save_last_model is set to
        False, this parameter is discarded.
    init_file_name : str, default = "init_model"
        The name of the file of the init model, if save_init_model is set to False,
        this parameter is discarded.
    verbose : boolean, default = False
        whether to output extra information
    loss : str, default = "categorical_crossentropy"
        The name of the keras training loss.
    optimizer : keras.optimizer, default = tf.keras.optimizers.Adam()
        The keras optimizer used for training.
    metrics : str or list[str], default="accuracy"
        The evaluation metrics to use during training. If
        a single string metric is provided, it will be
        used as the only metric. If a list of metrics are
        provided, all will be used for evaluation.

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
    >>> from aeon.classification.deep_learning import ResNetClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> clf = ResNetClassifier(n_epochs=20, batch_size=4) # doctest: +SKIP
    >>> clf.fit(X_train, Y_train) # doctest: +SKIP
    ResNetClassifier(...)
    """

    def __init__(
        self,
        n_residual_blocks: int = 3,
        n_conv_per_residual_block: int = 3,
        n_filters: int | list[int] = None,
        kernel_size: int | list[int] = None,
        strides: int | list[int] = 1,
        dilation_rate: int | list[int] = 1,
        padding: str | list[str] = "same",
        activation: str | list[str] = "relu",
        use_bias: bool | list[bool] = True,
        n_epochs: int = 1500,
        callbacks=None,
        verbose: bool = False,
        loss: str = "categorical_crossentropy",
        metrics: str | list[str] = "accuracy",
        batch_size: int = 64,
        use_mini_batch_size: bool = False,
        random_state=None,
        file_path: str = "./",
        save_best_model: bool = False,
        save_last_model: bool = False,
        save_init_model: bool = False,
        best_file_name: str = "best_model",
        last_file_name: str = "last_model",
        init_file_name: str = "init_model",
        optimizer=None,
    ):
        # for sklearn compatibility, set all parameters as attributes :
        self.n_residual_blocks = n_residual_blocks
        self.n_conv_per_residual_block = n_conv_per_residual_block
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias

        # create the network instance
        network = ResNetNetwork(
            n_residual_blocks=n_residual_blocks,
            n_conv_per_residual_block=n_conv_per_residual_block,
            n_filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=use_bias,
            activation=activation,
            dilation_rate=dilation_rate,
            padding=padding,
        )

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
    def _get_test_params(cls, parameter_set="default"):
        return [
            {
                "n_epochs": 10,
                "batch_size": 4,
                "n_residual_blocks": 1,
                "n_filters": 5,
                "n_conv_per_residual_block": 1,
                "kernel_size": 3,
            }
        ]
