"""Residual Network (ResNetNetwork)."""

__maintainer__ = ["hadifawaz1999"]


from aeon.networks.base import BaseDeepLearningNetwork


class ResNetNetwork(BaseDeepLearningNetwork):
    """
    Establish the network structure for a ResNet.

    Adapted from the implementations used in [1]_.

    Parameters
    ----------
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
        **BaseDeepLearningNetwork._config,
        "structure": "encoder",
    }

    def __init__(
        self,
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
        self.n_residual_blocks = n_residual_blocks
        self.n_conv_per_residual_block = n_conv_per_residual_block
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias

        super().__init__()

    def _check_params(self):
        self._n_filters = BaseDeepLearningNetwork._check_layer_param(
            self.n_residual_blocks,
            "filters",
            self.n_filters,
            default=[64, 128, 128],
            depth_label="number of residual blocks",
        )
        self._kernel_size = BaseDeepLearningNetwork._check_layer_param(
            self.n_conv_per_residual_block,
            "kernels",
            self.kernel_size,
            default=[8, 5, 3],
            depth_label="number of convolution layers per residual block",
        )
        self._strides = BaseDeepLearningNetwork._check_layer_param(
            self.n_conv_per_residual_block,
            "strides",
            self.strides,
            depth_label="number of convolution layers per residual block",
        )
        self._dilation_rate = BaseDeepLearningNetwork._check_layer_param(
            self.n_conv_per_residual_block,
            "dilations",
            self.dilation_rate,
            depth_label="number of convolution layers per residual block",
        )
        self._padding = BaseDeepLearningNetwork._check_layer_param(
            self.n_conv_per_residual_block,
            "paddings",
            self.padding,
            depth_label="number of convolution layers per residual block",
        )
        self._activation = BaseDeepLearningNetwork._check_layer_param(
            self.n_conv_per_residual_block,
            "activations",
            self.activation,
            depth_label="number of convolution layers per residual block",
        )
        self._use_bias = BaseDeepLearningNetwork._check_layer_param(
            self.n_conv_per_residual_block,
            "biases",
            self.use_bias,
            depth_label="number of convolution layers per residual block",
        )

    def build_base_graph(self, x):
        import tensorflow as tf

        self._check_params()

        for d in range(self.n_residual_blocks):
            input_block_tensor = x

            for c in range(self.n_conv_per_residual_block):
                conv = tf.keras.layers.Conv1D(
                    filters=self._n_filters[d],
                    kernel_size=self._kernel_size[c],
                    strides=self._strides[c],
                    padding=self._padding[c],
                    dilation_rate=self._dilation_rate[c],
                )(x)
                conv = tf.keras.layers.BatchNormalization()(conv)

                if c == self.n_conv_per_residual_block - 1:
                    conv = self._shortcut_layer(
                        input_tensor=input_block_tensor, output_tensor=conv
                    )

                conv = tf.keras.layers.Activation(activation=self._activation[c])(conv)

                x = conv
        return x

    def _shortcut_layer(
        self, input_tensor, output_tensor, padding="same", use_bias=True
    ):
        import tensorflow as tf

        n_out_filters = int(output_tensor.shape[-1])

        shortcut_layer = tf.keras.layers.Conv1D(
            filters=n_out_filters, kernel_size=1, padding=padding, use_bias=use_bias
        )(input_tensor)
        shortcut_layer = tf.keras.layers.BatchNormalization()(shortcut_layer)

        return tf.keras.layers.Add()([output_tensor, shortcut_layer])
