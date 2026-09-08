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
        **BaseDeepLearningNetwork._config,
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
        self.latent_space_dim = latent_space_dim
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.activation = activation
        self.n_filters = n_filters
        self.dilation_rate = dilation_rate
        self.padding = padding

        super().__init__()

    def _check_params(self):
        default_n_filters = [32 * (i + 1) for i in range(self.n_layers)]
        default_dilation_rate = [2**i for i in range(self.n_layers)]
        self._kernel_size = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.kernel_size, "kernels", default=3
        )
        self._activation = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.activation, "activations", allow_none=True
        )
        self._n_filters = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.n_filters, "filters", default_n_filters
        )
        self._dilation_rate = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.dilation_rate, "dilation rates", default_dilation_rate
        )
        self._padding = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, self.padding, "paddings", default="causal"
        )

    def build_base_graph(self, x):

        self._check_params()

        for i in range(0, self.n_layers):
            x = self._dcnn_layer(
                x,
                self._n_filters[i],
                self._dilation_rate[i],
                _activation=self._activation[i],
                _kernel_size=self._kernel_size[i],
                _padding=self._padding[i],
            )
        return x

    def build_network(self, input_shape, **kwargs):
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

        input_layer = tf.keras.layers.Input(input_shape)
        x = self.build_base_graph(input_layer)
        x = tf.keras.layers.GlobalMaxPool1D()(x)
        output_layer = tf.keras.layers.Dense(self.latent_space_dim)(x)

        return input_layer, output_layer

    def _dcnn_layer(
        self, _inputs, _n_filters, _dilation_rate, _activation, _kernel_size, _padding
    ):
        import tensorflow as tf

        from aeon.utils.networks.weight_norm import _WeightNormalization

        _add = tf.keras.layers.Conv1D(_n_filters, kernel_size=1)(_inputs)
        x = _WeightNormalization(
            tf.keras.layers.Conv1D(
                _n_filters,
                kernel_size=_kernel_size,
                dilation_rate=_dilation_rate,
                padding=_padding,
            )
        )(_inputs)
        x = _WeightNormalization(
            tf.keras.layers.Conv1D(
                _n_filters,
                kernel_size=_kernel_size,
                dilation_rate=_dilation_rate,
                padding=_padding,
                activation=_activation,
            )
        )(x)
        output = tf.keras.layers.Add()([x, _add])
        output = tf.keras.layers.Activation(_activation)(output)
        return output
