"""Time Convolutional Neural Network (CNN) for classification."""

__maintainer__ = []
__all__ = [
    "TapNetRegressor",
]

import gc
from copy import deepcopy

from sklearn.utils import check_random_state

from aeon.networks import TapNetNetwork
from aeon.regression.deep_learning.base import BaseDeepRegressor


class TapNetRegressor(BaseDeepRegressor):
    """Time series attentional prototype network (TapNet).

    Adapted from the implementation used in [1]_.

    TapNet was initially proposed for multivariate time series classification. The is an
    adaptation for time series regression. TapNet comprises these components: random
    dimension permutation, multivariate time series encoding, and attentional
    prototype learning.

    Parameters
    ----------
    filter_sizes        : array of int, default = (256, 256, 128)
        sets the kernel size argument for each convolutional block.
        Controls number of convolutional filters
        and number of neurons in attention dense layers.
    kernel_size        : array of int, default = (8, 5, 3)
        controls the size of the convolutional kernels
    layers              : array of int, default = (500, 300)
        size of dense layers
    n_epochs            : int, default = 2000
        number of epochs to train the model
    batch_size          : int, default = 16
        number of samples per update
    callbacks           : list of str, default = None
        list of callbacks to apply during training
    dropout             : float, default = 0.5
        dropout rate, in the range [0, 1)
    dilation            : int, default = 1
        dilation value
    activation          : str, default = "sigmoid"
        activation function for the last output layer
    loss                : str, default = "mean_squared_error"
        loss function for the classifier
    metrics: str or list of str, default="mean_squared_error"
        The evaluation metrics to use during training. If
        a single string metric is provided, it will be
        used as the only metric. If a list of metrics are
        provided, all will be used for evaluation.
    optimizer           : str or None, default = "Adam(lr=0.01)"
        gradient updating function for the classifier
    padding             : str, default = "same"
        padding argument for the convolutional layers
    use_bias            : bool, default = True
        whether to use bias in the output dense layer
    use_rp              : bool, default = True
        whether to use random projections
    use_att             : bool, default = True
        whether to use self attention
    use_lstm        : bool, default = True
        whether to use an LSTM layer
    use_cnn         : bool, default = True
        whether to use a CNN layer
    verbose         : bool, default = False
        whether to output extra information
    rp_params       : tuple, default = (-1, 3)
        parameters for random projection
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
        Seeded random number generation can only be guaranteed on CPU processing,
        GPU processing will be non-deterministic.

    References
    ----------
    .. [1] Zhang et al. Tapnet: Multivariate time series classification with
    attentional prototypical network,
    Proceedings of the AAAI Conference on Artificial Intelligence
    34(4), 6845-6852, 2020

    Notes
    -----
    The Implementation of TapNet found at https://github.com/kdd2019-tapnet/tapnet
    Currently does not implement custom distance matrix loss function
    or class  based self attention.
    """

    _tags = {
        "python_dependencies": ["tensorflow", "keras_self_attention"],
    }

    def __init__(
        self,
        n_epochs=2000,
        batch_size=16,
        dropout=0.5,
        filter_sizes=(256, 256, 128),
        kernel_size=(8, 5, 3),
        dilation=1,
        layers=(500, 300),
        use_rp=True,
        activation=None,
        rp_params=(-1, 3),
        use_bias=True,
        use_att=True,
        use_lstm=True,
        use_cnn=True,
        random_state=None,
        padding="same",
        loss="mean_squared_error",
        optimizer=None,
        metrics="mean_squared_error",
        callbacks=None,
        verbose=False,
    ):
        self.random_state = random_state
        self.kernel_size = kernel_size
        self.layers = layers
        self.rp_params = rp_params
        self.filter_sizes = filter_sizes
        self.activation = activation
        self.use_att = use_att
        self.use_bias = use_bias

        self.dilation = dilation
        self.padding = padding
        self.n_epochs = n_epochs
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.callbacks = callbacks
        self.verbose = verbose

        self.dropout = dropout
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn

        # parameters for random projection
        self.use_rp = use_rp
        self.rp_params = rp_params

        super().__init__(
            batch_size=batch_size,
        )

        self._network = TapNetNetwork(
            dropout=self.dropout,
            filter_sizes=self.filter_sizes,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            layers=self.layers,
            use_rp=self.use_rp,
            rp_params=self.rp_params,
            use_att=self.use_att,
            use_lstm=self.use_lstm,
            use_cnn=self.use_cnn,
            padding=self.padding,
        )

    def build_model(self, input_shape, **kwargs):
        """Construct a complied, un-trained, keras model that is ready for training.

        In aeon, time series are stored in numpy arrays of shape (d,m), where d
        is the number of dimensions, m is the series length. Keras/tensorflow assume
        data is in shape (m,d). This method also assumes (m,d). Transpose should
        happen in fit.

        Parameters
        ----------
        input_shape     : tuple
            The shape of the data fed into the input layer, should be (m, d)

        Returns
        -------
        output: a compiled Keras model
        """
        import numpy as np
        import tensorflow as tf
        from tensorflow import keras

        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)
        tf.keras.utils.set_random_seed(self.random_state_)
        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(
            units=1, activation=self.activation, use_bias=self.use_bias
        )(output_layer)

        self.optimizer_ = (
            keras.optimizers.Adam(learning_rate=0.01)
            if self.optimizer is None
            else self.optimizer
        )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=self._metrics,
        )

        return model

    def _fit(self, X, y):
        """
        Fit the regressor on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray
            The training input samples of shape (n_cases, n_channels, n_timepoints).
        y : np.ndarray
            The training data target values of shape (n_cases,).

        Returns
        -------
        self: object
        """
        # Transpose to conform to expectation format from keras
        X = X.transpose(0, 2, 1)

        self.input_shape = X.shape[1:]

        if isinstance(self.metrics, str):
            self._metrics = [self.metrics]
        else:
            self._metrics = self.metrics
        self.model_ = self.build_model(self.input_shape)
        if self.verbose:
            self.model_.summary()
        self.history = self.model_.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=deepcopy(self.callbacks) if self.callbacks else [],
        )

        gc.collect()
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
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
            "n_epochs": 10,
            "batch_size": 4,
            "padding": "valid",
            "filter_sizes": (16, 16, 16),
            "kernel_size": (3, 3, 1),
            "layers": (25, 50),
        }
        return [param1]
