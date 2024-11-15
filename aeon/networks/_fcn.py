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
        Whether or not ot use bias in convolution.

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
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
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
        self.activation = activation
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias

        super().__init__()

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple
          shape = (n_timepoints (m), n_channels (d)), the shape of the data fed
          into the input layer.

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        import tensorflow as tf

        self._n_filters_ = [128, 256, 128] if self.n_filters is None else self.n_filters
        self._kernel_size_ = [8, 5, 3] if self.kernel_size is None else self.kernel_size

        if isinstance(self._n_filters_, list):
            if len(self._n_filters_) != self.n_layers:
                raise ValueError(
                    f"Number of filters {len(self._n_filters_)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._n_filters = self._n_filters_
        else:
            self._n_filters = [self._n_filters_] * self.n_layers

        if isinstance(self._kernel_size_, list):
            if len(self._kernel_size_) != self.n_layers:
                raise ValueError(
                    f"Number of kernels {len(self._kernel_size_)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._kernel_size = self._kernel_size_
        else:
            self._kernel_size = [self._kernel_size_] * self.n_layers

        if isinstance(self.dilation_rate, list):
            if len(self.dilation_rate) != self.n_layers:
                raise ValueError(
                    f"Number of dilations {len(self.dilation_rate)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._dilation_rate = self.dilation_rate
        else:
            self._dilation_rate = [self.dilation_rate] * self.n_layers

        if isinstance(self.strides, list):
            if len(self.strides) != self.n_layers:
                raise ValueError(
                    f"Number of strides {len(self.strides)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._strides = self.strides
        else:
            self._strides = [self.strides] * self.n_layers

        if isinstance(self.padding, list):
            if len(self.padding) != self.n_layers:
                raise ValueError(
                    f"Number of paddings {len(self.padding)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._padding = self.padding
        else:
            self._padding = [self.padding] * self.n_layers

        if isinstance(self.activation, list):
            if len(self.activation) != self.n_layers:
                raise ValueError(
                    f"Number of activations {len(self.activation)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._activation = self.activation
        else:
            self._activation = [self.activation] * self.n_layers

        if isinstance(self.use_bias, list):
            if len(self.use_bias) != self.n_layers:
                raise ValueError(
                    f"Number of biases {len(self.use_bias)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._use_bias = self.use_bias
        else:
            self._use_bias = [self.use_bias] * self.n_layers

        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer

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

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(conv)

        return input_layer, gap_layer
