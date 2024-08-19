"""Dilated Convolutional Nerual Networks (DCNN) Model."""

__maintainer__ = []

from aeon.networks.base import BaseDeepLearningNetwork


class DCNNNetwork(BaseDeepLearningNetwork):
    """Establish the network structure for a DCNN-Model.

    Dilated Convolutional Neural Network based Model
    for low-rank embeddings.

    Parameters
    ----------
    latent_space_dim: int, default=128
        Dimension of the models's latent space.
    n_layers: int, default=4
        Number of convolution layers.
    kernel_size: Union[int, List[int]], default=3
        Size of the 1D Convolutional Kernel. Defaults
        to a list of three's for `n_layers` elements.
    activation: Union[str, List[str]], default="relu"
        The activation function used by convolution layers.
        Defaults to a list of "relu" for `n_layers` elements.
    n_filters: Union[int, List[int]], default=None
        Number of filters used in convolution layers. Defaults
        to a list of multiple's of 32 for `n_layers` elements.
    dilation_rate: Union[int, List[int]], default=None
        The dilation rate for convolution. Defaults to a list of
        powers of 2 for `n_layers` elements.
    padding: Union[str, List[str]], default="causal"
        Padding to be used in each DCNN Layer. Defaults to a list
        of causal paddings for `n_layers` elements.

    References
    ----------
    .. [1] Franceschi, J. Y., Dieuleveut, A., & Jaggi, M. (2019).
    Unsupervised scalable representation learning for multivariate
    time series. Advances in neural information processing systems, 32.
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.12",
        "structure": "encoder",
    }

    def __init__(
        self,
        latent_space_dim=128,
        n_layers=4,
        kernel_size=3,
        activation="relu",
        n_filters=None,
        dilation_rate=None,
        padding="causal",
    ):
        super().__init__()

        self.latent_space_dim = latent_space_dim
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.padding = padding

    def build_network(self, input_shape):
        """Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple of shape = (n_timepoints (m), n_channels (d))
            The shape of the data fed into the input layer.

        Returns
        -------
        model : a keras Model.
        """
        import tensorflow as tf

        if self.n_filters is None:
            self._n_filters = [32 * i for i in range(1, self.n_layers + 1)]
        elif isinstance(self.n_filters, int):
            self._n_filters = [self.n_filters for _ in range(self.n_layers)]
        elif isinstance(self.n_filters, list):
            self._n_filters = self.n_filters
            assert len(self.n_filters) == self.n_layers

        if self.dilation_rate is None:
            self._dilation_rate = [
                2**layer_num for layer_num in range(1, self.n_layers + 1)
            ]
        elif isinstance(self.dilation_rate, int):
            self._dilation_rate = [self.dilation_rate for _ in range(self.n_layers)]
        else:
            self._dilation_rate = self.dilation_rate
            assert isinstance(self.dilation_rate, list)
            assert len(self.dilation_rate) == self.n_layers

        if self.kernel_size is None:
            self._kernel_size = [3 for _ in range(self.n_layers)]
        elif isinstance(self.kernel_size, int):
            self._kernel_size = [self.kernel_size for _ in range(self.n_layers)]
        elif isinstance(self.kernel_size, list):
            self._kernel_size = self.kernel_size
            assert len(self.kernel_size) == self.n_layers

        if self.activation is None:
            self._activation = ["relu" for _ in range(self.n_layers)]
        elif isinstance(self.activation, str):
            self._activation = [self.activation for _ in range(self.n_layers)]
        elif isinstance(self.activation, list):
            self._activation = self.activation
            assert len(self._activation) == self.n_layers

        if self.padding is None:
            self._padding = ["causal" for _ in range(self.n_layers)]
        elif isinstance(self.padding, str):
            self._padding = [self.padding for _ in range(self.n_layers)]
        elif isinstance(self.padding, list):
            self._padding = self.padding
            assert len(self._padding) == self.n_layers

        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer
        for i in range(0, self.n_layers):
            x = self._dcnn_layer(
                x,
                self._n_filters[i],
                self._dilation_rate[i],
                _activation=self._activation[i],
                _kernel_size=self._kernel_size[i],
                _padding=self._padding[i],
            )

        x = tf.keras.layers.GlobalMaxPool1D()(x)
        output_layer = tf.keras.layers.Dense(self.latent_space_dim)(x)

        return input_layer, output_layer

    def _dcnn_layer(
        self, _inputs, _n_filters, _dilation_rate, _activation, _kernel_size, _padding
    ):
        import tensorflow as tf

        _add = tf.keras.layers.Conv1D(_n_filters, kernel_size=1)(_inputs)
        x = tf.keras.layers.Conv1D(
            _n_filters,
            kernel_size=_kernel_size,
            dilation_rate=_dilation_rate,
            padding=_padding,
            kernel_regularizer="l2",
        )(_inputs)
        x = tf.keras.layers.Conv1D(
            _n_filters,
            kernel_size=_kernel_size,
            dilation_rate=_dilation_rate,
            padding="causal",
            kernel_regularizer="l2",
            activation=_activation,
        )(x)
        output = tf.keras.layers.Add()([x, _add])
        output = tf.keras.layers.Activation(_activation)(output)
        return output
