"""Deep Learning Auto-Encoder using DRNN Network."""

__maintainer__ = []
__all__ = ["AEDRNNClusterer"]


from aeon.clustering import DummyClusterer
from aeon.clustering.deep_learning.base import BaseDeepClusterer
from aeon.networks import AEDRNNNetwork
from aeon.typing import LATENT_SPACE


class AEDRNNClusterer(BaseDeepClusterer):
    """
    Auto-Encoder based Dilated Recurrent Neural Network (DRNN).

    Adapted from [1]_.

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
    n_layers_encoder : int, default = 3
        Number of layers in the encoder.
    n_layers_decoder : int, default = 3
        Number of layers in the decoder.
    dilation_rate_encoder : int or list of int, default = 1
        The dilation rate for the encoder.
    dilation_rate_decoder : int or list of int, default = 1
        The dilation rate for the decoder.
    activation_encoder : str or list of str, default = "relu"
        Activation used after DRNN Layer in the encoder.
    activation_decoder : str or list of str, default = "relu"
        Activation used after DRNN Layer in the decoder.
    n_units_encoder : list or int, default = None
        Number of Units in each DRNN Layer of the encoder.
    n_units_decoder : list or int, default = None
        Number of Units in each DRNN Layer of the decoder.
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
    metrics : keras metrics, default = ["mean_squared_error"]
        will be set to mean_squared_error as default if None
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
    .. [1] Ma Q et. al, Learning representations for time series
    clustering, Advances in neural information processing systems
    (NeurIPS), 2019.


    Examples
    --------
    >>> from aeon.clustering.deep_learning import AEDRNNClusterer
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> from aeon.clustering import DummyClusterer
    >>> _clst = DummyClusterer(n_clusters=2)
    >>> aefcn = AEDRNNClusterer(estimator = _clst,
    ... n_epochs=20,batch_size=4)  # doctest: +SKIP
    >>> aefcn.fit(X_train)  # doctest: +SKIP
    AEDRNNClusterer(...)
    """

    def __init__(
        self,
        estimator=None,
        latent_space_dim=128,
        latent_space_type=LATENT_SPACE.REPEATED,
        n_layers_encoder=3,
        n_layers_decoder=3,
        dilation_rate_encoder=1,
        dilation_rate_decoder=1,
        n_units_encoder=None,
        n_units_decoder=None,
        activation_encoder="relu",
        activation_decoder="relu",
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
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.activation_encoder = activation_encoder
        self.activation_decoder = activation_decoder
        self.dilation_rate_encoder = dilation_rate_encoder
        self.dilation_rate_decoder = dilation_rate_decoder
        self.n_units_encoder = n_units_encoder
        self.n_units_decoder = n_units_decoder

        self._network = AEDRNNNetwork(
            latent_space_dim=self.latent_space_dim,
            latent_space_type=self.latent_space_type,
            n_layers_encoder=self.n_layers_encoder,
            n_layers_decoder=self.n_layers_decoder,
            dilation_rate_encoder=self.dilation_rate_encoder,
            dilation_rate_decoder=self.dilation_rate_decoder,
            activation_encoder=self.activation_encoder,
            activation_decoder=self.activation_decoder,
            n_units_encoder=self.n_units_encoder,
            n_units_decoder=self.n_units_decoder,
        )

        self.can_multi_rec = False

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
        param1 = {
            "estimator": DummyClusterer(n_clusters=2),
            "n_epochs": 1,
            "batch_size": 4,
            "n_layers_encoder": 1,
            "n_layers_decoder": 1,
        }

        return [param1]
