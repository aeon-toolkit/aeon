"""Time Convolutional Neural Network (TimeCNNNetwork)."""

__maintainer__ = ["hadifawaz1999"]

from aeon.networks.base import BaseDeepLearningNetwork


class TimeCNNNetwork(BaseDeepLearningNetwork):
    """Establish the network structure for a CNN.

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    n_layers : int, default = 2
        The number of convolution layers in the network.
    kernel_size : int or list of int, default = 7
        Kernel size of convolution layers, if not a list, the same kernel size is
        used for all layer, len(list) should be n_layers.
    n_filters : int or list of int, default = [6, 12]
        Number of filters for each convolution layer, if not a list, the same
        `n_filters` is used in all layers.
    avg_pool_size : int or list of int, default = 3
        The size of the average pooling layer, if not a list, the same max pooling
        size is used for all convolution layer.
    activation : str or list of str, default = "sigmoid"
        Keras activation function used in the model for each layer, if not a list,
        the same activation is used for all layers.
    padding : str or list of str, default = "valid"
        The method of padding in convolution layers, if not a list, the same padding
        used for all convolution layers.
    strides : int or list of int, default = 1
        The strides of kernels in the convolution and max pooling layers, if not a list,
        the same strides are used for all layers.
    strides_pooling : int or list of int, default = None
        Strides for the pooling layers. If None, defaults to pool_size.
        If not a list, the same strides are used for all pooling layers.
    dilation_rate : int or list of int, default = 1
        The dilation rate of the convolution layers, if not a list, the same dilation
        rate is used all over the network.
    use_bias : bool or list of bool, default = True
        Condition on whether or not to use bias values for convolution layers, if not
        a list, the same condition is used for all layers.

    Notes
    -----
    Adapted from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/cnn.py

    References
    ----------
    .. [1] Zhao et al. Convolutional neural networks for time series classification,
    Journal of Systems Engineering and Electronics 28(1), 162--169, 2017
    """

    _config = {
        **BaseDeepLearningNetwork._config,
        "structure": "encoder",
    }

    def __init__(
        self,
        n_layers=2,
        kernel_size=7,
        n_filters=None,
        avg_pool_size=3,
        activation="sigmoid",
        padding="valid",
        strides=1,
        strides_pooling=None,
        dilation_rate=1,
        use_bias=True,
    ):
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.avg_pool_size = avg_pool_size
        self.activation = activation
        self.padding = padding
        self.strides = strides
        self.strides_pooling = strides_pooling
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias

        super().__init__()

    def _check_params(self):
        self._kernel_size = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, "kernels", self.kernel_size
        )
        self._n_filters = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, "filters", self.n_filters, default=[6, 12]
        )
        self._avg_pool_size = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, "average pool sizes", self.avg_pool_size
        )
        self._activation = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, "activations", self.activation, accept_none=True
        )
        self._padding = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, "paddings", self.padding
        )
        self._strides = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, "strides", self.strides
        )
        self._strides_pooling = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers,
            "strides for pooling",
            self.strides_pooling,
            default=self.avg_pool_size,
        )
        self._dilation_rate = BaseDeepLearningNetwork._check_layer_param(
            self.n_layers, "dilations", self.dilation_rate
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
                padding=self._padding[i],
                dilation_rate=self._dilation_rate[i],
                activation=self._activation[i],
                use_bias=self._use_bias[i],
            )(x)

            conv = tf.keras.layers.AveragePooling1D(
                pool_size=self._avg_pool_size[i],
                strides=self._strides_pooling[i],
            )(conv)

            x = conv
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

        # TODO : explain why we need to force padding to "same" for short time series
        if input_shape[0] < 60:
            self._padding = ["same"] * self.n_layers

        input_layer = tf.keras.layers.Input(input_shape)
        x = self.build_base_graph(input_layer)
        flatten_layer = tf.keras.layers.Flatten()(x)

        return input_layer, flatten_layer
