"""Multi Layer Perceptron Network (MLP) classifier."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["MLPClassifier"]

from aeon.classification.deep_learning._deep_classifier import DeepClassifier
from aeon.networks import MLPNetwork


class MLPClassifier(DeepClassifier):
    """Multi Layer Perceptron Network (MLP).

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    n_layers : int, optional (default=3)
        The number of dense layers in the MLP.
    n_units : Union[int, List[int]], optional (default=500)
        Number of units in each dense layer.
    activation : Union[str, List[str]], optional (default='relu')
        Activation function(s) for each dense layer.
    dropout_rate : Union[float, List[Union[int, float]]], optional (default=None)
        Dropout rate(s) for each dense layer. If None, a default rate of 0.2 is used
        except the first element, being 0.1. Dropout rate(s) are typically a number
        in the interval [0, 1].
    dropout_last : float, default = None
        The dropout rate of the last layer. If set to None, defaults to 0.3.
    use_bias : bool, default = True
        Condition on whether or not to use bias values for dense layers.
    n_epochs : int, default = 2000
        the number of epochs to train the model
    batch_size : int, default = 16
        the number of samples per gradient update.
    use_mini_batch_size : boolean, default = False
        Condition on using the mini batch size formula
    callbacks : keras callback or list of callbacks,
        default = None
        The default list of callbacks are set to
        ModelCheckpoint and ReduceLROnPlateau.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
        Seeded random number generation can only be guaranteed on CPU processing,
        GPU processing will be non-deterministic.
    verbose : boolean, default = False
        whether to output extra information
    loss : str, default = "categorical_crossentropy"
        The name of the keras training loss.
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
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py

    References
    ----------
    .. [1] Wang et. al, Time series classification from
    scratch with deep neural networks: A strong baseline,
    International joint conference on neural networks (IJCNN), 2017.

    Examples
    --------
    >>> from aeon.classification.deep_learning import MLPClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> mlp = MLPClassifier(n_epochs=20, batch_size=4)  # doctest: +SKIP
    >>> mlp.fit(X_train, y_train)  # doctest: +SKIP
    MLPClassifier(...)
    """

    def __init__(
        self,
        n_layers: int = 3,
        n_units: int | list[int] = 500,
        activation: str | list[str] = "relu",
        dropout_rate: float | list[int | float] = None,
        dropout_last: float = None,
        use_bias: bool = True,
        n_epochs: int = 2000,
        batch_size: int = 16,
        use_mini_batch_size: bool = False,
        callbacks=None,
        verbose: bool = False,
        loss: str = "categorical_crossentropy",
        metrics: str | list[str] = "accuracy",
        file_path: str = "./",
        save_best_model: bool = False,
        save_last_model: bool = False,
        save_init_model: bool = False,
        best_file_name: str = "best_model",
        last_file_name: str = "last_model",
        init_file_name: str = "init_model",
        random_state=None,
        optimizer=None,
    ):
        # Store architecture parameters for sklearn compatibility.
        self.n_layers = n_layers
        self.n_units = n_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout_last = dropout_last
        self.use_bias = use_bias

        # Create the network instance.
        network = MLPNetwork(
            n_layers=n_layers,
            n_units=n_units,
            activation=activation,
            dropout_rate=dropout_rate,
            dropout_last=dropout_last,
            use_bias=use_bias,
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
                "use_bias": False,
            }
        ]
