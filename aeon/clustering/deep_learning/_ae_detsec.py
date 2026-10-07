"""Deep Learning Auto-Encoder using Attention Bidirectional GRU Network."""

__maintainer__ = []
__all__ = ["AEDeTSECClusterer"]

import gc

from aeon.clustering import DummyClusterer
from aeon.clustering.deep_learning.base import BaseDeepClusterer
from aeon.networks import AEDeTSECNetwork


class AEDeTSECClusterer(BaseDeepClusterer):
    """
    Auto-Encoder based on DeTSEC network.

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    estimator : aeon clusterer, default=None
        An aeon estimator to be built using the transformed data.
        Defaults to aeon TimeSeriesKMeans() with euclidean distance
        and mean averaging method and n_clusters set to 2.
    n_filters_encoder : int, default=64
        Number of filters in the encoder.
    n_filters_decoder : int or list of int, default=64
        Number of filters in the decoder.
    activation_encoder : str, default='tanh'
        Activation function of the encoder.
    activation_decoder : str, default='tanh'
        Activation function of the decoder.
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
    loss : str, default="mean_squared_error"
        Fit parameter for the keras model.
    metrics : str, default=["mean_squared_error"]
        Metrics to evaluate model predictions.
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

    Examples
    --------
    >>> from aeon.clustering.deep_learning import AEAttentionBiGRUClusterer
    >>> from aeon.clustering import DummyClusterer
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> _clst = DummyClusterer(n_clusters=2)
    >>> abgruc=AEAttentionBiGRUClusterer(estimator=_clst, n_epochs=20,
    ... batch_size=4) # doctest: +SKIP
    >>> abgruc.fit(X_train)  # doctest: +SKIP
    AEAttentionBiGRUClusterer(...)

    Notes
    -----
    While originally based on [1]_, this implementation
    derives from the adaptation in [2]_, this
    implementation only uses the reconstruction training
    (pretext loss) phase and not the cluster optimization

    References
    ----------
    .. [1] Ienco & Interdonato, Deep Multivariate Time Series Embedding
    Clustering via Attentive-Gated Autoencoder, Advances in Knowledge
    Discovery and Data Mining, 2020.

    .. [2] Lafabregue et. al, End-to-end deep representation learning
    for time series clustering: a comparative study, Data Mining
    and Knowledge Discovery, 2022.

    """

    def __init__(
        self,
        estimator=None,
        n_filters_encoder=64,
        n_filters_decoder=64,
        activation_encoder="tanh",
        activation_decoder="tanh",
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

        self.n_filters_encoder = n_filters_encoder
        self.n_filters_decoder = n_filters_decoder
        self.activation_encoder = activation_encoder
        self.activation_decoder = activation_decoder

        self._network = AEDeTSECNetwork(
            n_filters_encoder=self.n_filters_encoder,
            n_filters_decoder=self.n_filters_decoder,
            activation_encoder=self.activation_encoder,
            activation_decoder=self.activation_decoder,
        )

        self.can_multi_rec = False

    def build_model(self, input_shape, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In aeon, time series are stored in numpy arrays of shape
        (n_channels,n_timepoints). Keras/tensorflow assume
        data is in shape (n_timepoints,n_channels). This method also assumes
        (n_timepoints,n_channels). Transpose should happen in fit.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be
            (n_timepoints,n_channels).

        Returns
        -------
        output : a compiled Keras Model.
        """
        import tensorflow as tf

        self._check_params()

        tf.keras.utils.set_random_seed(self._random_state)
        self._network.build_network(input_shape, **kwargs)
        model = self._network.get_autoencoder_model()

        self._metrics = self._metrics + self._metrics
        model.compile(optimizer=self._optimizer, loss=self.loss, metrics=self._metrics)

        return model

    def _fit(self, X):
        """Fit the Clusterer on the training set X.

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases (n), n_channels (d), n_timepoints (m))
            The training input samples.

        Returns
        -------
        self : object
        """
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        self.input_shape = X.shape[1:]
        self.training_model_ = self.build_model(self.input_shape)

        if self.save_init_model:
            self.training_model_.save(self.file_path + self.init_file_name + ".keras")

        if self.verbose:
            self.training_model_.summary()

        if self.use_mini_batch_size:
            self._batch_size = min(self.batch_size, X.shape[0] // 10)
        else:
            self._batch_size = self.batch_size

        self.history = self.training_model_.fit(
            X,
            (X, X),
            batch_size=self._batch_size,
            validation_split=self.validation_split,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self._callbacks,
        )

        self._load_best_model_and_save()

        self._fit_clustering(X=X)

        gc.collect()
        return self

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
            "n_filters_encoder": 2,
            "n_filters_decoder": 2,
        }

        return [param1]
