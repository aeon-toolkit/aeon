"""Encoder Classifier."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["EncoderClassifier"]

from aeon.classification.deep_learning._deep_classifier import DeepClassifier
from aeon.networks import EncoderNetwork


class EncoderClassifier(DeepClassifier):
    """
    Establish the network structure for an Encoder.

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    kernel_size : array of int, default = None
        Specifying the length of the 1D convolution windows.
        If set to None, defaults to [5, 11, 21].
    n_filters : array of int, default = None
        Specifying the number of 1D convolution filters used for each layer,
        the shape of this array should be the same as kernel_size.
        If set to None, defaults to [128, 256, 512].
    max_pool_size : int, default = 2
        Size of the max pooling windows.
    activation : string, default = sigmoid
        Keras activation function.
    dropout_proba : float, default = 0.2
        Specifying the dropout layer probability.
    padding : string, default = same
        Specifying the type of padding used for the 1D convolution.
    strides : int, default = 1
        Specifying the sliding rate of the 1D convolution filter.
    fc_units : int, default = 256
        Specifying the number of units in the hidden fully
        connected layer used in the EncoderNetwork.
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
    callbacks : keras callback or list of callbacks,
        default = None
        The default list of callbacks are set to
        ModelCheckpoint.
    file_path : str, default = "./"
        File path when saving model_Checkpoint callback.
    n_epochs : int, default = 2000
        The number of epochs to train the model.
    batch_size : int, default = 16
        The number of samples per gradient update.
    use_mini_batch_size : bool, default = False
        Whether or not to use the mini batch size formula.
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
        The name of the file of the init model, if save_init_model is set to False,
        this parameter is discarded.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
        Seeded random number generation can only be guaranteed on CPU processing,
        GPU processing will be non-deterministic.

    Notes
    -----
    Adapted from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/encoder.py

    References
    ----------
    .. [1] Serrà et al. Towards a Universal Neural Network Encoder for Time Series
    In proceedings International Conference of the Catalan Association
    for Artificial Intelligence, 120--129 2018.


    """

    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 12,
        kernel_size: list[int] = None,
        n_filters: list[int] = None,
        dropout_proba: float = 0.2,
        activation: str = "sigmoid",
        max_pool_size: int = 2,
        padding: str = "same",
        strides: int = 1,
        fc_units: int = 256,
        callbacks=None,
        file_path: str = "./",
        save_best_model: bool = False,
        save_last_model: bool = False,
        save_init_model: bool = False,
        best_file_name: str = "best_model",
        last_file_name: str = "last_model",
        init_file_name: str = "init_model",
        verbose: bool = False,
        loss: str = "categorical_crossentropy",
        metrics: str | list[str] = "accuracy",
        random_state=None,
        use_bias: bool = True,
        optimizer=None,
    ):
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.max_pool_size = max_pool_size
        self.activation = activation
        self.dropout_proba = dropout_proba
        self.padding = padding
        self.strides = strides
        self.fc_units = fc_units
        self.use_bias = use_bias  # not used?

        network = EncoderNetwork(
            kernel_size=kernel_size,
            max_pool_size=max_pool_size,
            n_filters=n_filters,
            fc_units=fc_units,
            strides=strides,
            padding=padding,
            dropout_proba=dropout_proba,
            activation=activation,
        )

        _optimizer = optimizer
        if _optimizer is None:
            import tensorflow as tf

            # keep the old optimizer behavior
            _optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

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
                "n_epochs": 8,
                "batch_size": 4,
                "use_bias": False,
                "fc_units": 8,
                "strides": 2,
                "dropout_proba": 0,
            }
        ]
