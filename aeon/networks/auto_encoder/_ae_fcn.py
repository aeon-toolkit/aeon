"""Auto-Encoder using Fully Convolutional Network (FCN)."""

__maintainer__ = ["hadifawaz1999"]

from aeon.networks.base import BaseDeepAENetwork
from aeon.networks.encoder._fcn import FCNNetwork


class AEFCNNetwork(BaseDeepAENetwork):
    """Establish the network structure for a AE-FCN.

    Auto-Encoder based Fully Convolutional Netwwork (AE-FCN),
    adapted from the implementation used in [1]_.

    Parameters
    ----------
    latent_space_dim : int, default = 128
        Dimension of the auto-encoder's latent space.
    temporal_latent_space : bool, default = False
        Flag to choose whether the latent space is an MTS or Euclidean space.
    n_layers : int, default = 3
        Number of convolution layers.
    n_filters : int or list of int, default = [128,256,128]
        Number of filters used in convolution layers.
    kernel_size : int or list of int, default = [8,5,3]
        Size of convolution kernel.
    dilation_rate : int or list of int, default = 1
        The dilation rate for convolution.
    strides : int or list of int, default = 1
        The strides of the convolution filter.
    padding : str or list of str, default = "same"
        The type of padding used for convolution.
    activation : str or list of str, default = "relu"
        Activation used after the convolution.
    use_bias : bool or list of bool, default = True
        Whether or not to use bias in convolution.

    Notes
    -----
    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py

    References
    ----------
    .. [1] Network originally defined in:
    @inproceedings{wang2017time,
      title={Time series classification from scratch with deep neural networks:
       A strong baseline},
      author={Wang, Zhiguang and Yan, Weizhong and Oates, Tim},
      booktitle={2017 International joint conference on neural networks
      (IJCNN)},
      pages={1578--1585},
      year={2017},
      organization={IEEE}
    }
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
        n_layers=3,
        n_filters=None,
        kernel_size=None,
        dilation_rate=1,
        strides=1,
        padding="same",
        activation="relu",
        use_bias=True,
    ):
        super().__init__(latent_space_dim, temporal_latent_space, repeated_latent_space)
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias

    def _check_params(self):
        n = self.n_layers
        self._n_filters = BaseDeepAENetwork._check_layer_param(
            n, self.n_filters, "filters", default=[128, 256, 128]
        )
        self._kernel_size = BaseDeepAENetwork._check_layer_param(
            n, self.kernel_size, "kernels", default=[8, 5, 3]
        )
        self._dilation_rate = BaseDeepAENetwork._check_layer_param(
            n, self.dilation_rate, "dilation rates", default=1
        )
        self._strides = BaseDeepAENetwork._check_layer_param(
            n, self.strides, "strides", default=1
        )
        self._padding = BaseDeepAENetwork._check_layer_param(
            n, self.padding, "paddings", default="same"
        )
        self._activation = BaseDeepAENetwork._check_layer_param(
            n, self.activation, "activations", allow_none=True
        )
        self._use_bias = BaseDeepAENetwork._check_layer_param(
            n, self.use_bias, "use bias", default=True
        )

    def _build_encoder_graph(self, x):
        self._input_shape = x.shape[1:]
        encoder = FCNNetwork(
            n_layers=self.n_layers,
            n_filters=self._n_filters,
            kernel_size=self._kernel_size,
            strides=self._strides,
            dilation_rate=self._dilation_rate,
            padding=self._padding,
            activation=self._activation,
            use_bias=self._use_bias,
        )
        encoder._set_block_activation_names_for_ae(
            [f"__act_encoder_block_{i}" for i in range(self.n_layers)]
        )
        return encoder.build_base_graph(x)

    def _build_decoder_graph(self, x):
        decoder = FCNNetwork(
            n_layers=self.n_layers,
            n_filters=self._n_filters[::-1],
            kernel_size=self._kernel_size[::-1],
            strides=self._strides[::-1],
            dilation_rate=self._dilation_rate[::-1],
            padding=self._padding[::-1],
            activation=self._activation[::-1],
            use_bias=self._use_bias[::-1],
            transpose=True,
        )
        decoder._set_block_activation_names_for_ae(
            [f"__act_decoder_block_{i}" for i in range(self.n_layers)]
        )
        return decoder.build_base_graph(x)
