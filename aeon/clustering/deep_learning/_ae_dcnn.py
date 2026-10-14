"""Deep Learning Auto-Encoder using DCNN Network."""

__all__ = ["AEDCNNClusterer"]


from aeon.clustering import DummyClusterer
from aeon.clustering.deep_learning.base import BaseDeepClusterer
from aeon.networks import AEDCNNNetwork
from aeon.typing import LATENT_SPACE


class AEDCNNClusterer(BaseDeepClusterer):
    """Auto-Encoder based Dilated Convolutional Networks (DCNN), as described in [1]_.

    Parameters
    ----------
    estimator : aeon clusterer, default=None
        An aeon estimator to be built using the transformed data.
        Defaults to aeon TimeSeriesKMeans() with euclidean distance
        and mean averaging method and n_clusters set to 2.
    latent_space_dim : int, default=128
        Dimension of the latent space of the auto-encoder.
    latent_space_type : LATENT_SPACE, default = LATENT_SPACE.FLAT
        Type of latent space to use. Options are:
        - LATENT_SPACE.FLAT: The latent space is a flattened vector.
        - LATENT_SPACE.TIME: The latent space is a time series.
        - LATENT_SPACE.REPEATED: The latent space is a repeated vector.
    n_layers : int, default = 3
        Number of convolution layers in the encoder.
    kernel_size : int or list of int, default = 3
        Size of convolution kernel in the encoder.
    activation : str or list of str, default = "relu"
        Activation used after the convolution in the encoder.
    n_filters : int or list of int, default = None
        Number of filters used in convolution layers in the encoder.
    dilation_rate : int or list of int, default = 1
        The dilation rate for convolution in the encoder.
        `dilation_rate` greater than `1` is not supported on
        `Conv1DTranspose` for some devices/OS.
    padding_encoder : str or list of str, default = "causal"
        Keras compatible Padding string for the encoder. Defaults to a list
        of "causal" paddings.
    padding_decoder : str or list of str, default = "same"
        Keras compatible Padding string for the decoder. Defaults to a list
        of "same" paddings.
    use_bias : bool or list of bool, default = True
        Whether or not to use bias in convolution.
    n_epochs : int, default = 2000
        The number of epochs to train the model.
    batch_size : int, default = 16
        The number of samples per gradient update.
    validation_split: float, default = 0
        Fraction of the training data to be used as validation data.
    use_mini_batch_size : bool, default = True,
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
    loss : string, default="mean_squared_error"
        Fit parameter for the keras model.
    metrics : List[str], default=["mean_squared_error"]
        Metrics to evaluate the performance of the deep learning network.
    optimizer : keras.optimizers object, default = Adam(lr=0.01)
        Specify the optimizer and the learning rate to be used.
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
    callbacks : keras.callbacks, default = None
        List of keras callbacks.

    Attributes
    ----------
    estimator_ : BaseClusterer
        The fitted clustering estimator used to assign cluster labels
        from the model's latent space representation.

    References
    ----------
    .. [1] Franceschi et. al, Unsupervised scalable representation
    learning for multivariate time series, Advances in neural
    information processing systems (NeurIPS), 2019.

    Examples
    --------
    >>> from aeon.clustering.deep_learning import AEDCNNClusterer
    >>> from aeon.datasets import load_unit_test
    >>> from aeon.clustering import DummyClusterer
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> _clst = DummyClusterer(n_clusters=2)
    >>> aedcnn=AEDCNNClusterer(estimator=_clst, n_epochs=20,
    ... batch_size=4)  # doctest: +SKIP
    >>> aedcnn.fit(X_train)  # doctest: +SKIP
    AEDCNNClusterer(...)
    """

    def __init__(
        self,
        estimator=None,
        latent_space_dim=128,
        latent_space_type=LATENT_SPACE.FLAT,
        n_layers=3,
        kernel_size=3,
        activation="relu",
        n_filters=None,
        dilation_rate=1,
        padding_encoder="same",
        padding_decoder="same",
        n_epochs=2000,
        batch_size=32,
        validation_split=0,
        use_mini_batch_size=False,
        random_state=None,
        verbose=False,
        loss="mse",
        metrics=None,
        optimizer="Adam",
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        save_init_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        init_file_name="init_model",
        callbacks=None,
    ):
        super().__init__(
            estimator=estimator,
            n_epochs=n_epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            use_mini_batch_size=use_mini_batch_size,
            random_state=random_state,
            verbose=verbose,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            file_path=file_path,
            save_best_model=save_best_model,
            save_last_model=save_last_model,
            save_init_model=save_init_model,
            best_file_name=best_file_name,
            last_file_name=last_file_name,
            init_file_name=init_file_name,
            callbacks=callbacks,
        )

        self.latent_space_dim = latent_space_dim
        self.latent_space_type = latent_space_type
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.activation = activation
        self.n_filters = n_filters
        self.dilation_rate = dilation_rate
        self.padding_encoder = padding_encoder
        self.padding_decoder = padding_decoder

        self._network = AEDCNNNetwork(
            latent_space_dim=self.latent_space_dim,
            latent_space_type=self.latent_space_type,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            activation=self.activation,
            n_filters=self.n_filters,
            dilation_rate=self.dilation_rate,
            padding_encoder=self.padding_encoder,
            padding_decoder=self.padding_decoder,
        )

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
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
        param = {
            "estimator": DummyClusterer(n_clusters=2),
            "n_epochs": 1,
            "batch_size": 4,
            "n_layers": 1,
            "n_filters": 1,
            "kernel_size": None,
        }

        return [param]
