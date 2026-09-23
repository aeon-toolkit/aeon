"""Auto-Encoder using Residual Network (AEResNetNetwork)."""

__maintainer__ = ["hadifawaz1999"]


from aeon.networks.base import BaseDeepAENetwork
from aeon.networks.encoder._resnet import ResNetNetwork


class AEResNetNetwork(BaseDeepAENetwork):
    """
    Establish the network structure for a AE-ResNet.

    Adapted from the implementations used in [1]_.

    Parameters
    ----------
    latent_space_dim : int, default = 128
        Dimension of the auto-encoder's latent space.
    temporal_latent_space : bool, default = False
        Flag to choose whether the latent space is an MTS or Euclidean space.
    repeated_latent_space : bool, default = False
        Flag to choose whether the latent space is given as a repeated vector to the
        decoder.
    n_residual_blocks : int, default = 3
        The number of residual blocks of ResNet's model.
    n_conv_per_residual_block : int, default = 3
        The number of convolution blocks in each residual block.
    n_filters : int or list of int, default = [128, 64, 64]
        The number of convolution filters for all the convolution layers in the same
        residual block, if not a list, the same number of filters is used in all
        convolutions of all residual blocks.
    kernel_size : int or list of int, default = [8, 5, 3]
        The kernel size of all the convolution layers in one residual block, if not a
        list, the same kernel size is used in all convolution layers.
    strides : int or list of int, default = 1
        The strides of convolution kernels in each of the convolution layers in one
        residual block, if not a list, the same kernel size is used in all
        convolution layers.
    dilation_rate : int or list of int, default = 1
        The dilation rate of the convolution layers in one residual block, if not a
        list, the same kernel size is used in all convolution layers.
    padding : str or list of str, default = 'padding'
        The type of padding used in the convolution layers in one residual block, if not
        a list, the same kernel size is used in all convolution layers.
    activation : str or list of str, default = 'relu'
        Keras activation used in the convolution layers in one residual block, if not
        a list, the same kernel size is used in all convolution layers.
    use_bias : bool or list of bool, default = True
        Condition on whether or not to use bias values in the convolution layers in
        one residual block, if not a list, the same kernel size is used in all
        convolution layers.

    Notes
    -----
    Adapted from the implementation source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py

    References
    ----------
    .. [1] H. Fawaz, G. B. Lanckriet, F. Petitjean, and L. Idoumghar,

    Network originally defined in:

    @inproceedings{wang2017time, title={Time series classification from
    scratch with deep neural networks: A strong baseline}, author={Wang,
    Zhiguang and Yan, Weizhong and Oates, Tim}, booktitle={2017
    International joint conference on neural networks (IJCNN)}, pages={
    1578--1585}, year={2017}, organization={IEEE} }

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
        n_residual_blocks=3,
        n_conv_per_residual_block=3,
        n_filters=None,
        kernel_size=None,
        strides=1,
        dilation_rate=1,
        padding="same",
        activation="relu",
        use_bias=True,
    ):
        super().__init__(latent_space_dim, temporal_latent_space, repeated_latent_space)
        self.n_residual_blocks = n_residual_blocks
        self.n_conv_per_residual_block = n_conv_per_residual_block
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias

    def _check_params(self):
        n_conv = self.n_conv_per_residual_block
        n_res = self.n_residual_blocks
        res = "number of residual blocks"
        conv = "number of convolution layers per residual block"
        self._n_filters = BaseDeepAENetwork._check_layer_param(
            n_res, self.n_filters, "filters", default=[64, 128, 128], same_as=res
        )
        self._kernel_size = BaseDeepAENetwork._check_layer_param(
            n_conv, self.kernel_size, "kernels", default=[8, 5, 3], same_as=conv
        )
        self._strides = BaseDeepAENetwork._check_layer_param(
            n_conv, self.strides, "strides", default=1, same_as=conv
        )
        self._dilation_rate = BaseDeepAENetwork._check_layer_param(
            n_conv, self.dilation_rate, "dilations", default=1, same_as=conv
        )
        self._padding = BaseDeepAENetwork._check_layer_param(
            n_conv, self.padding, "paddings", default="same", same_as=conv
        )
        self._activation = BaseDeepAENetwork._check_layer_param(
            n_conv, self.activation, "activations", allow_none=True, same_as=conv
        )
        self._use_bias = BaseDeepAENetwork._check_layer_param(
            n_conv, self.use_bias, "biases", default=True, same_as=conv
        )

    def _build_encoder_graph(self, x):
        self._input_shape = x.shape[1:]
        encoder = ResNetNetwork(
            n_residual_blocks=self.n_residual_blocks,
            n_conv_per_residual_block=self.n_conv_per_residual_block,
            n_filters=self._n_filters,
            kernel_size=self._kernel_size,
            strides=self._strides,
            dilation_rate=self._dilation_rate,
            padding=self._padding,
            activation=self._activation,
            use_bias=self._use_bias,
        )
        encoder._set_block_activation_names_for_ae(
            [f"__act_encoder_block_{i}" for i in range(self.n_residual_blocks)]
        )
        return encoder.build_base_graph(x)

    def _build_decoder_graph(self, x):
        decoder = ResNetNetwork(
            n_residual_blocks=self.n_residual_blocks,
            n_conv_per_residual_block=self.n_conv_per_residual_block,
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
            [f"__act_decoder_block_{i}" for i in range(self.n_residual_blocks)]
        )
        return decoder.build_base_graph(x)
