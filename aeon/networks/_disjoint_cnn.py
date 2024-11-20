"""Disjoint Convolutional Neural Network (DisjointCNNNetwork)."""

__maintainer__ = ["hadifawaz1999"]


from aeon.networks.base import BaseDeepLearningNetwork


class DisjointCNNNetwork(BaseDeepLearningNetwork):
    """Establish the network structure for a DisjointCNN Network.

    The model is proposed in [1]_ to apply convolutions
    specifically for multivariate series, temporal-spatial
    phases using 1+1D Convolution layers.

    Parameters
    ----------
    n_layers : int, default = 4
        Number of 1+1D Convolution layers.
    n_filters : int or list of int, default = 64
        Number of filters used in convolution layers. If
        input is set to a list, the lenght should be the same
        as `n_layers`, if input is int the a list of the same
        element is created of length `n_layers`.
    kernel_size : int or list of int, default = [8, 5, 5, 3]
        Size of convolution kernel. If
        input is set to a list, the lenght should be the same
        as `n_layers`, if input is int the a list of the same
        element is created of length `n_layers`.
    dilation_rate : int or list of int, default = 1
        The dilation rate for convolution. If
        input is set to a list, the lenght should be the same
        as `n_layers`, if input is int the a list of the same
        element is created of length `n_layers`.
    strides : int or list of int, default = 1
        The strides of the convolution filter. If
        input is set to a list, the lenght should be the same
        as `n_layers`, if input is int the a list of the same
        element is created of length `n_layers`.
    padding : str or list of str, default = "same"
        The type of padding used for convolution. If
        input is set to a list, the lenght should be the same
        as `n_layers`, if input is int the a list of the same
        element is created of length `n_layers`.
    activation : str or list of str, default = "elu"
        Activation used after the convolution. If
        input is set to a list, the lenght should be the same
        as `n_layers`, if input is int the a list of the same
        element is created of length `n_layers`.
    use_bias : bool or list of bool, default = True
        Whether or not ot use bias in convolution. If
        input is set to a list, the lenght should be the same
        as `n_layers`, if input is int the a list of the same
        element is created of length `n_layers`.
    kernel_initializer: str or list of str, default = "he_uniform"
        The initialization method of convolution layers. If
        input is set to a list, the lenght should be the same
        as `n_layers`, if input is int the a list of the same
        element is created of length `n_layers`.
    pool_size: int, default = 5
        The size of the one max pool layer at the end of
        the model, default = 5.
    pool_strides: int, default = None
        The strides used for the one max pool layer at
        the end of the model, default = None.
    pool_padding: str, default = "valid"
        The padding method for the one max pool layer at
        the end of the model, default = "valid".
    hidden_fc_units: int, default = 128
        The number of fully connected units.
    activation_fc: str, default = "relu"
        The activation of the fully connected layer.

    Notes
    -----
    The code is adapted from:
    https://github.com/Navidfoumani/Disjoint-CNN

    References
    ----------
    .. [1] Foumani, Seyed Navid Mohammadi, Chang Wei Tan, and Mahsa Salehi.
    "Disjoint-cnn for multivariate time series classification."
    2021 International Conference on Data Mining Workshops
    (ICDMW). IEEE, 2021.
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
        "structure": "encoder",
    }

    def __init__(
        self,
        n_layers=4,
        n_filters=64,
        kernel_size=None,
        dilation_rate=1,
        strides=1,
        padding="same",
        activation="elu",
        use_bias=True,
        kernel_initializer="he_uniform",
        pool_size=5,
        pool_strides=None,
        pool_padding="valid",
        hidden_fc_units=128,
        activation_fc="relu",
    ):
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.pool_padding = pool_padding
        self.hidden_fc_units = hidden_fc_units
        self.activation_fc = activation_fc

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

        self._kernel_size_ = (
            [8, 5, 5, 3] if self.kernel_size is None else self.kernel_size
        )

        if isinstance(self._kernel_size_, list):
            if len(self._kernel_size_) != self.n_layers:
                raise ValueError(
                    f"Kernel sizes {len(self._kernel_size_)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._kernel_size = self._kernel_size_
        else:
            self._kernel_size = [self._kernel_size_] * self.n_layers

        if isinstance(self.n_filters, list):
            if len(self.n_filters) != self.n_layers:
                raise ValueError(
                    f"Number of filters {len(self.n_filters)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._n_filters = self.n_filters
        else:
            self._n_filters = [self.n_filters] * self.n_layers

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

        if isinstance(self.kernel_initializer, list):
            if len(self.kernel_initializer) != self.n_layers:
                raise ValueError(
                    f"Number of Kernel initializers {len(self.kernel_initializer)}"
                    f" should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._kernel_initializer = self.kernel_initializer
        else:
            self._kernel_initializer = [self.kernel_initializer] * self.n_layers

        input_layer = tf.keras.layers.Input(input_shape)
        reshape_layer = tf.keras.layers.Reshape(
            target_shape=(input_shape[0], input_shape[1], 1)
        )(input_layer)

        x = reshape_layer

        for i in range(self.n_layers):
            x = self._one_plus_one_d_convolution_layer(
                input_tensor=x,
                n_filters=self._n_filters[i],
                kernel_size=self._kernel_size[i],
                dilation_rate=self._dilation_rate[i],
                strides=self._strides[i],
                padding=self._padding[i],
                use_bias=self._use_bias[i],
                activation=self._activation[i],
                kernel_initializer=self._kernel_initializer[i],
            )

        max_pool_layer = tf.keras.layers.MaxPooling2D(
            pool_size=(self.pool_size, 1),
            strides=self.pool_strides,
            padding=self.pool_padding,
        )(x)

        gap = tf.keras.layers.GlobalAveragePooling2D()(max_pool_layer)

        projection_head = tf.keras.layers.Dense(
            self.hidden_fc_units, activation=self.activation_fc
        )(gap)

        return input_layer, projection_head

    def _one_plus_one_d_convolution_layer(
        self,
        input_tensor,
        n_filters,
        kernel_size,
        dilation_rate,
        strides,
        padding,
        use_bias,
        activation,
        kernel_initializer,
    ):
        import tensorflow as tf

        temporal_conv = tf.keras.layers.Conv2D(
            n_filters,
            (kernel_size, 1),
            padding=padding,
            kernel_initializer=kernel_initializer,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            strides=strides,
        )(input_tensor)

        temporal_conv = tf.keras.layers.BatchNormalization()(temporal_conv)
        temporal_conv = tf.keras.layers.Activation(activation=activation)(temporal_conv)

        temporal_conv_output_channels = int(temporal_conv.shape[2])

        spatial_conv = tf.keras.layers.Conv2D(
            n_filters,
            (1, temporal_conv_output_channels),
            padding="valid",
            kernel_initializer=kernel_initializer,
        )(temporal_conv)

        spatial_conv = tf.keras.layers.BatchNormalization()(spatial_conv)
        spatial_conv = tf.keras.layers.Activation(activation=activation)(spatial_conv)

        spatial_conv = tf.keras.layers.Permute((1, 3, 2))(spatial_conv)

        return spatial_conv
