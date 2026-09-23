"""Auto-Encoder based on Dilated Convolutional Nerual Networks (DCNN) Model."""

__maintainer__ = ["aadya940", "hadifawaz1999"]


from aeon.networks.base import BaseDeepAENetwork
from aeon.networks.encoder._dcnn import DCNNNetwork


class AEDCNNNetwork(BaseDeepAENetwork):
    """Establish the Auto-Encoder based structure for a DCN Network.

    Dilated Convolutional Neural (DCN) Network based Model
    for low-rank embeddings.

    Parameters
    ----------
    latent_space_dim: int, default=128
        Dimension of the models's latent space.
    temporal_latent_space : bool, default = False
        Flag to choose whether the latent space is an MTS or Euclidean space.
    n_layers: int, default=4
        Number of convolution layers in the autoencoder.
    kernel_size: Union[int, List[int]], default=3
        Size of the 1D Convolutional Kernel of the encoder. Defaults to a
        list of length `n_layers` with `kernel_size` value.
    activation: Union[str, List[str]], default="relu"
        The activation function used by convolution layers of the encoder.
        Defaults to a list of "relu" for `n_layers` elements.
    n_filters: Union[int, List[int]], default=None
        Number of filters used in convolution layers of the encoder. Defaults
        to a list of multiples of `32` for `n_layers` elements.
    dilation_rate: Union[int, List[int]], default=1
        The dilation rate for convolution of the encoder. Defaults to a list
        of powers of `2` for `n_layers` elements. `dilation_rate` greater than
        `1` is not supported on `Conv1DTranspose` for some devices/OS.
    padding_encoder: Union[str, List[str]], default="same"
        The padding string for the encoder layers. Defaults to a list of "same"
        for `n_layers` elements. Valid strings are "causal", "valid", "same" or
        any other Keras compatible string.
    padding_decoder: Union[str, List[str]], default="same"
        The padding string for the decoder layers. Defaults to a list of "same"
        for `n_layers` elements.

    References
    ----------
    .. [1] Franceschi, J. Y., Dieuleveut, A., & Jaggi, M. (2019). Unsupervised
    scalable representation learning for multivariate time series. Advances in
    neural information processing systems, 32.

    """

    _config = {
        **BaseDeepAENetwork._config,
        "structure": "auto-encoder",
    }

    def __init__(
        self,
        latent_space_dim=128,
        temporal_latent_space=False,
        repeated_latent_space=False,
        n_layers=4,
        kernel_size=3,
        activation="relu",
        n_filters=None,
        dilation_rate=1,
        padding_encoder="same",
        padding_decoder="same",
    ):
        super().__init__(latent_space_dim, temporal_latent_space, repeated_latent_space)
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.activation = activation
        self.n_filters = n_filters
        self.dilation_rate = dilation_rate
        self.padding_encoder = padding_encoder
        self.padding_decoder = padding_decoder

    def _check_params(self):
        default_n_filters = [32 * i for i in range(1, self.n_layers + 1)]
        default_dilation_rate = [2**l for l in range(1, self.n_layers + 1)]

        self._kernel_size_encoder = BaseDeepAENetwork._check_layer_param(
            self.n_layers, self.kernel_size, "kernel size", default=3
        )
        self._activation_encoder = BaseDeepAENetwork._check_layer_param(
            self.n_layers, self.activation, "activations", allow_none=True
        )
        self._n_filters_encoder = BaseDeepAENetwork._check_layer_param(
            self.n_layers, self.n_filters, "filters", default_n_filters
        )
        self._dilation_rate = BaseDeepAENetwork._check_layer_param(
            self.n_layers, self.dilation_rate, "dilation rates", default_dilation_rate
        )
        self._padding_encoder = BaseDeepAENetwork._check_layer_param(
            self.n_layers, self.padding_encoder, "padding for encoder", default="same"
        )
        self._padding_decoder = BaseDeepAENetwork._check_layer_param(
            self.n_layers, self.padding_decoder, "padding for decoder", default="same"
        )
        # add default values for strides, padding, and use_bias
        # for compatibility with _build_latent_space_graph
        self._strides = BaseDeepAENetwork._check_layer_param(
            self.n_layers, param_name="strides", default=1
        )
        self._padding = BaseDeepAENetwork._check_layer_param(
            self.n_layers, param_name="padding", default="same"
        )
        self._use_bias = BaseDeepAENetwork._check_layer_param(
            self.n_layers, param_name="use_bias", default=True
        )

    def _build_encoder_graph(self, x):
        self._input_shape = x.shape[1:]
        return DCNNNetwork(
            n_layers=self.n_layers,
            n_filters=self._n_filters_encoder,
            kernel_size=self._kernel_size_encoder,
            activation=self._activation_encoder,
            dilation_rate=self._dilation_rate,
            padding=self._padding_encoder,
        ).build_base_graph(x)

    def _build_decoder_graph(self, x):
        return DCNNNetwork(
            n_layers=self.n_layers,
            n_filters=self._n_filters_encoder[::-1],
            kernel_size=self._kernel_size_encoder[::-1],
            activation=self._activation_encoder[::-1],
            dilation_rate=self._dilation_rate[::-1],
            padding=self._padding_decoder,
            transpose=True,
        ).build_base_graph(x)
