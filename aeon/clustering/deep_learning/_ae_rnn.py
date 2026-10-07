"""Deep Learning Auto-Encoder using Bidirectional GRU Network."""

__maintainer__ = []
__all__ = ["AERecurrentClusterer"]

from aeon.clustering import DummyClusterer
from aeon.clustering.deep_learning.base import BaseDeepClusterer
from aeon.networks import AERecurrentNetwork
from aeon.typing import LATENT_SPACE, RNN_TYPE


class AERecurrentClusterer(BaseDeepClusterer):
    """Auto-Encoder based on RNNs for clustering time series data.

    Parameters
    ----------
    estimator : aeon clusterer, default=None
        An aeon estimator to be built using the transformed data.
        Defaults to aeon TimeSeriesKMeans() with euclidean distance
        and mean averaging method and n_clusters set to 2.
    latent_space_dim : int, default=128
        Dimension of the latent space.
    latent_space_type : LATENT_SPACE, default = LATENT_SPACE.REPEATED
        Type of latent space to use. Options are:
        - LATENT_SPACE.FLAT: The latent space is a flattened vector.
        - LATENT_SPACE.TIME: The latent space is a time series.
        - LATENT_SPACE.REPEATED: The latent space is a repeated vector.
    rnn_type : RNN_TYPE, default=RNN_TYPE.LSTM
        Type of RNN cell to use. Options are:
        - RNN_TYPE.LSTM: Long Short-Term Memory cell.
        - RNN_TYPE.GRU: Gated Recurrent Unit cell.
        - RNN_TYPE.SIMPLE: Simple RNN cell.
    n_layers : int, default=1
        Number of recurrent layers.
    n_units : int or list of int, default=64
        Number of units in each recurrent layer. If an int, the same number
        of units is used in each layer. If a list, specifies the number of
        units for each layer and must match the number of layers.
    dropout : float or list of float, default=0.0
        Dropout rate applied to each recurrent layer.
    residual : float or list of float or 2D numpy array of float, default=0
        Residual connections strength for each layer.
        Zero means no residual connection. One means full residual connection.
        A float value between 0 and 1 means a weighted residual connection.
        If a residual connect two layers with different number of units,
        a conv1D layer with kernel_size=1 is added to match the shape.
        When a 2D numpy array is provided, it should have shape (n_layers, n_layers) and
        represent residual connections from input of layer i to output of layer j.
        No connections where 'j' is less than 'i' are allowed.
    bidirectional : bool, default=False
        Whether to use bidirectional recurrent layers.
    activation : str or list of str, default='tanh'
        Activation function(s) for the recurrent layers. If a string, the same
        activation is used for all layers. If a list, specifies activation for
        each layer and must match the number of layers.
    attention : bool or list of bool, default=False
        Whether to apply self-attention mechanism after each recurrent layer.
    use_bias : bool or list of bool, default=True
        Condition on whether or not to use bias values in the RNN layers.
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
    """

    def __init__(
        self,
        estimator=None,
        latent_space_dim=128,
        latent_space_type=LATENT_SPACE.REPEATED,
        rnn_type=RNN_TYPE.LSTM,
        n_layers=2,
        n_units=64,
        dropout=0.0,
        residual=0,
        bidirectional=False,
        activation="tanh",
        attention=False,
        use_bias=True,
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
            batch_size=batch_size,
            last_file_name=last_file_name,
        )

        self.latent_space_dim = latent_space_dim
        self.latent_space_type = latent_space_type
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.n_units = n_units
        self.dropout = dropout
        self.residual = residual
        self.bidirectional = bidirectional
        self.activation = activation
        self.attention = attention
        self.use_bias = use_bias
        self.n_epochs = n_epochs
        self.validation_split = validation_split
        self.use_mini_batch_size = use_mini_batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.file_path = file_path
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.init_file_name = init_file_name
        self.callbacks = callbacks

        self._network = AERecurrentNetwork(
            latent_space_dim=self.latent_space_dim,
            latent_space_type=self.latent_space_type,
            rnn_type=self.rnn_type,
            n_layers=self.n_layers,
            n_units=self.n_units,
            dropout=self.dropout,
            residual=self.residual,
            bidirectional=self.bidirectional,
            activation=self.activation,
            attention=self.attention,
            use_bias=self.use_bias,
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
        param = {
            "estimator": DummyClusterer(n_clusters=2),
            "n_epochs": 1,
            "batch_size": 4,
            "n_layers": 1,
            "n_units": 2,
            "latent_space_dim": 2,
        }

        return [param]
