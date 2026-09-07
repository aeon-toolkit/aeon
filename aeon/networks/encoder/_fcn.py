"""Fully Convolutional Network (FCNNetwork)."""

__maintainer__ = ["hadifawaz1999"]


from aeon.networks.base import BaseDeepLearningNetwork


class FCNNetwork(BaseDeepLearningNetwork):
    """
    Establish the network structure for a FCN.

    Adapted from the implementation used in [1]_

    Parameters
    ----------
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
    .. [1] Wang et al. Time series classification from scratch with deep neural
    networks: a strong baseline, IJCNN, 2017
    """

    _config = {
        **BaseDeepLearningNetwork._config,
        "structure": "encoder",
    }

    def __init__(
        self,
        n_layers=3,
        n_filters=None,
        kernel_size=None,
        dilation_rate=1,
        strides=1,
        padding="same",
        activation="relu",
        use_bias=True,
    ):
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias

        super().__init__()

    def _check_params(self):
        self._n_filters = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, "filters", self.n_filters, default=[128, 256, 128]
        )
        self._kernel_size = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, "kernels", self.kernel_size, default=[8, 5, 3]
        )
        self._dilation_rate = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, "dilations", self.dilation_rate
        )
        self._strides = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, "strides", self.strides
        )
        self._padding = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, "paddings", self.padding
        )
        self._activation = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, "activations", self.activation, accept_none=True
        )
        self._use_bias = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, "biases", self.use_bias
        )

    def build_base_graph(self, x):
        import tensorflow as tf

        self._check_params()

        for i in range(self.n_layers):
            conv = tf.keras.layers.Conv1D(
                filters=self._n_filters[i],
                kernel_size=self._kernel_size[i],
                strides=self._strides[i],
                dilation_rate=self._dilation_rate[i],
                padding=self._padding[i],
                use_bias=self._use_bias[i],
            )(x)

            conv = tf.keras.layers.BatchNormalization()(conv)
            conv = tf.keras.layers.Activation(activation=self._activation[i])(conv)

            x = conv
        return x
