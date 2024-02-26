"""Residual Network (ResNet) for clustering."""

__maintainer__ = []
__all__ = ["AEResNetClusterer"]

import gc
import os
import time
from copy import deepcopy

from sklearn.utils import check_random_state

from aeon.clustering.deep_learning.base import BaseDeepClusterer
from aeon.networks import AEResNetNetwork


class AEResNetClusterer(BaseDeepClusterer):
    """
    Residual Neural Network (RNN).

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    n_clusters : int, default=None
    Number of clusters for the deep learning model.
    clustering_algorithm : str, default="kmeans"
        The clustering algorithm used in the latent space.
    clustering_params : dict, default=None
        Dictionary containing the parameters of the clustering algorithm chosen.
    latent_space_dim : int, default=128
        Dimension of the latent space of the auto-encoder.
    temporal_latent_space : bool, default = False
        Flag to choose whether the latent space is an MTS or Euclidean space.
    n_residual_blocks : int, default = 3
        The number of residual blocks of ResNet's model.
    n_conv_per_residual_block : int, default = 3
        The number of convolution blocks in each residual block.
    n_filters : int or list of int, default = [128, 64, 64]
        The number of convolution filters for all the convolution layers in the same
        residual block, if not a list, the same number of filters is used in all
        convolutions of all residual blocks.
    kernel_sizes : int or list of int, default = [8, 5, 3]
        The kernel size of all the convolution layers in one residual block, if not
        a list, the same kernel size is used in all convolution layers.
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
    use_bia : bool or list of bool, default = True
        Condition on whether or not to use bias values in the convolution layers
        in one residual block, if not a list, the same kernel size is used in all
        convolution layers.
    n_epochs : int, default = 1500
        The number of epochs to train the model.
    batch_size : int, default = 16
        The number of samples per gradient update.
    use_mini_batch_size : bool, default = False
        Condition on using the mini batch size formula Wang et al.
    callbacks : callable or None, default ReduceOnPlateau and ModelCheckpoint
        List of tf.keras.callbacks.Callback objects.
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
    best_file_name : str, default = "best_model"
        The name of the file of the best model, if save_best_model is set to
        False, this parameter is discarded.
    last_file_name : str, default = "last_model"
        The name of the file of the last model, if save_last_model is set to
        False, this parameter is discarded.
    verbose : boolean, default = False
        whether to output extra information
    loss : string, default = "mean_squared_error"
        fit parameter for the keras model.
    optimizer : keras.optimizer, default = keras.optimizers.Adam()
    metrics : list of strings, default = ["accuracy"]

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
    >>> from aeon.clustering.deep_learning import AEResNetClusterer
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> ae_resnet = AEResNetClusterer(n_clusters=2,n_epochs=20) # doctest: +SKIP
    >>> ae_resnet.fit(X_train, y_train) # doctest: +SKIP
    AEResNetClusterer(...)
    """

    def __init__(
        self,
        n_clusters,
        n_residual_blocks=3,
        clustering_algorithm="kmeans",
        clustering_params=None,
        n_conv_per_residual_block=3,
        n_filters=None,
        kernel_size=None,
        strides=1,
        dilation_rate=1,
        padding="same",
        activation="relu",
        use_bias=True,
        n_epochs=1500,
        callbacks=None,
        verbose=False,
        loss="mse",
        metrics=None,
        batch_size=32,
        use_mini_batch_size=False,
        random_state=None,
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        best_file_name="best_model",
        last_file_name="last_file",
        optimizer="Adam",
    ):
        self.n_residual_blocks = n_residual_blocks
        self.n_conv_per_residual_block = n_conv_per_residual_block
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.n_epochs = n_epochs
        self.callbacks = callbacks
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.use_mini_batch_size = use_mini_batch_size
        self.random_state = random_state
        self.activation = activation
        self.use_bias = use_bias
        self.file_path = file_path
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.best_file_name = best_file_name
        self.last_file_name = last_file_name
        self.optimizer = optimizer

        self.history = None

        super().__init__(
            n_clusters=n_clusters,
            clustering_algorithm=clustering_algorithm,
            clustering_params=clustering_params,
            batch_size=batch_size,
            last_file_name=last_file_name,
        )

        self._network = AEResNetNetwork(
            n_residual_blocks=self.n_residual_blocks,
            n_conv_per_residual_block=self.n_conv_per_residual_block,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            use_bias=self.use_bias,
            activation=self.activation,
            dilation_rate=self.dilation_rate,
            padding=self.padding,
        )

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

        tf.random.set_seed(self.random_state)

        self.optimizer_ = (
            tf.keras.optimizers.Adam(learning_rate=0.01)
            if self.optimizer is None
            else self.optimizer
        )

        encoder, decoder = self._network.build_network(input_shape, **kwargs)

        input_layer = tf.keras.layers.Input(input_shape, name="input layer")
        encoder_output = encoder(input_layer)
        decoder_output = decoder(encoder_output)
        output_layer = tf.keras.layers.Reshape(
            target_shape=input_shape, name="outputlayer"
        )(decoder_output)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
        )

        return model

    def _fit(self, X):
        """Fit the Clusterer on the training set X.

        Parameters
        ----------
        X : np.ndarray of shape = (n_instances (n), n_dimensions (d), series_length (m))
            The training input samples.

        Returns
        -------
        self : object
        """
        import tensorflow as tf

        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        check_random_state(self.random_state)

        self.input_shape = X.shape[1:]
        self.training_model_ = self.build_model(self.input_shape)

        if self.verbose:
            self.training_model_.summary()

        self.file_name_ = (
            self.best_file_name if self.save_best_model else str(time.time_ns())
        )

        self.callbacks_ = (
            [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=50, min_lr=0.0001
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.file_path + self.file_name_ + ".hdf5",
                    monitor="loss",
                    save_best_only=True,
                ),
            ]
            if self.callbacks is None
            else self.callbacks
        )

        if self.use_mini_batch_size:
            mini_batch_size = min(self.batch_size, X.shape[0] // 10)
        else:
            mini_batch_size = self.batch_size

        self.history = self.training_model_.fit(
            X,
            X,
            batch_size=mini_batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks_,
        )

        try:
            self.model_ = tf.keras.models.load_model(
                self.file_path + self.file_name_ + ".hdf5", compile=False
            )
            if not self.save_best_model:
                os.remove(self.file_path + self.file_name_ + ".hdf5")
        except FileNotFoundError:
            self.model_ = deepcopy(self.training_model_)

        if self.save_last_model:
            self.save_last_model_to_file(file_path=self.file_path)

        self._fit_clustering(X=X)

        gc.collect()
        return self

    def _score(self, X, y=None):
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)
        latent_space = self.model_.layers[1].predict(X)
        return self.clusterer.score(latent_space)

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
        param = {
            "n_clusters": 2,
            "n_epochs": 1,
            "batch_size": 4,
            "n_residual_blocks": 1,
            "n_conv_per_residual_block": 1,
            "clustering_params": {
                "distance": "euclidean",
                "averaging_method": "mean",
                "n_init": 1,
                "max_iter": 30,
            },
        }

        test_params = [param]

        return test_params
