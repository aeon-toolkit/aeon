"""Residual Network (ResNet) for clustering."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["AEResNetClusterer"]

from aeon.clustering.deep_learning.base import BaseDeepClusterer
from aeon.clustering.dummy import DummyClusterer
from aeon.networks import AEResNetNetwork
from aeon.typing import LATENT_SPACE


class AEResNetClusterer(BaseDeepClusterer):
    """
    Auto-Encoder with Residual Network backbone for clustering.

    Adapted from the implementation used in [1]_.

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
    n_residual_blocks : int, default = 3
        The number of residual blocks of ResNet's model.
    n_conv_per_residual_block : int, default = 3
        The number of convolution blocks in each residual block.
    n_filters : int or list of int, default = [128, 64, 64]
        The number of convolution filters for all the convolution layers in the same
        residual block, if not a list, the same number of filters is used in all
        convolutions of all residual blocks.
    kernel_size : int or list of int, default = [8, 5, 3]
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
    use_bias : bool or list of bool, default = True
        Condition on whether or not to use bias values in the convolution layers
        in one residual block, if not a list, the same kernel size is used in all
        convolution layers.
    n_epochs : int, default = 1500
        The number of epochs to train the model.
    batch_size : int, default = 16
        The number of samples per gradient update.
    validation_split: float, default = 0
        Fraction of the training data to be used as validation data.
    use_mini_batch_size : bool, default = False
        Condition on using the mini batch size formula Wang et al.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
        Seeded random number generation can only be guaranteed on CPU processing,
        GPU processing will be non-deterministic.
    verbose : boolean, default = False
        whether to output extra information
    loss : string, default = "mean_squared_error"
        fit parameter for the keras model. "multi_rec" for multiple mse loss.
        Multiple mse loss computes mean squared error between all embeddings
        of encoder layers with the corresponding reconstructions of the
        decoder layers.
    metrics : list of strings, default = ["mean_squared_error"]
        will be set to mean_squared_error as default if None
    optimizer : keras.optimizer, default = keras.optimizers.Adam()
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
        The name of the file of the init model, if
        save_init_model is set to False,
        this parameter is discarded.
    callbacks : callable or None, default ReduceOnPlateau and ModelCheckpoint
        List of tf.keras.callbacks.Callback objects.

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
    >>> ae_resnet = AEResNetClusterer(n_epochs=20) # doctest: +SKIP
    >>> ae_resnet.fit(X_train, y_train) # doctest: +SKIP
    AEResNetClusterer(...)
    """

    def __init__(
        self,
        estimator=None,
        latent_space_dim=128,
        latent_space_type=LATENT_SPACE.FLAT,
        n_residual_blocks=3,
        n_conv_per_residual_block=3,
        n_filters=None,
        kernel_size=None,
        strides=1,
        dilation_rate=1,
        padding="same",
        activation="relu",
        use_bias=True,
        n_epochs=1500,
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
        self.n_residual_blocks = n_residual_blocks
        self.n_conv_per_residual_block = n_conv_per_residual_block
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias

        self._network = AEResNetNetwork(
            latent_space_dim=latent_space_dim,
            latent_space_type=latent_space_type,
            n_residual_blocks=self.n_residual_blocks,
            n_conv_per_residual_block=self.n_conv_per_residual_block,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            padding=self.padding,
            activation=self.activation,
            use_bias=self.use_bias,
        )

        self.can_multi_rec = True

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
        """
        param = {
            "estimator": DummyClusterer(n_clusters=2),
            "n_epochs": 2,
            "batch_size": 4,
            "n_residual_blocks": 2,
            "n_conv_per_residual_block": 1,
            "n_filters": [2, 2],
            "kernel_size": 2,
            "use_bias": False,
        }

        test_params = [param]

        return test_params
