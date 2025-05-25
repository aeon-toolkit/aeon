"""LITE Network (LITENetwork)."""

__maintainer__ = ["hadifawaz1999"]


from aeon.networks.base import BaseDeepLearningNetwork


class LITENetwork(BaseDeepLearningNetwork):
    """LITE and LITE Multivariate (LITEMV) Networks.

    LITE deep neural network architecture from [1]_ and its
    multivariate adaptation LITEMV from [2]_. For using
    LITEMV, simply set the `use_litemv` bool parameter to
    True.

    Parameters
    ----------
    use_litemv : bool, default = False
        The boolean value to control which version of the
        network to use. If set to `False`, then LITE is used,
        if set to `True` then LITEMV is used. LITEMV is the
        same architecture as LITE but specifically designed
        to better handle multivariate time series.
    n_filters : int, default = 32
        The number of filters used in one lite layer.
    kernel_size : int , default = 40
        The head kernel size used for each lite layer,.
    strides : int or list of int, default = 1
        The strides of kernels in convolution layers for each lite layer,
        if not a list, the same is used in all lite layers.
    activation : str or list of str, default = 'relu'
        The activation function used in each lite layer, if not a list,
        the same is used in all lite layers.

    Notes
    -----
    Adapted from the implementation from Ismail-Fawaz et. al

    https://github.com/MSD-IRIMAS/LITE

    References
    ----------
    ..[1] Ismail-Fawaz et al. LITE: Light Inception with boosTing
    tEchniques for Time Series Classificaion, IEEE International
    Conference on Data Science and Advanced Analytics, 2023.

    ..[2] Ismail-Fawaz, Ali, et al. "Look Into the LITE
    in Deep Learning for Time Series Classification."
    arXiv preprint arXiv:2409.02869 (2024).
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
        "structure": "encoder",
    }

    def __init__(
        self,
        use_litemv=False,
        n_filters=32,
        kernel_size=40,
        strides=1,
        activation="relu",
    ):
        self.use_litemv = use_litemv
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        super().__init__()

    def hybrid_layer(self, input_tensor, input_channels, kernel_sizes=None):
        """Construct the hybrid layer to compute features of custom filters.

        Parameters
        ----------
        input_tensor : tensorflow tensor, usually the input layer of the model.
        input_channels : int, the number of input channels in case of multivariate.
        kernel_sizes : list of int, default = [2,4,8,16,32,64],
        the size of the hand-crafted filters.

        Returns
        -------
        hybrid_layer : tensorflow tensor containing the concatenation
        of the output features extracted form hand-crafted convolution filters.

        """
        import numpy as np
        import tensorflow as tf

        kernel_sizes = [2, 4, 8, 16, 32, 64] if kernel_sizes is None else kernel_sizes

        self.keep_track = 0

        """
        Function to create the hybrid layer consisting of non
        trainable Conv1D layers with custom filters.

        Args:

            input_tensor: input tensor
            input_channels : number of input channels, 1 in case of UCR Archive
        """

        conv_list = []

        # for increasing detection filters

        for kernel_size in kernel_sizes:
            filter_ = np.ones(
                shape=(kernel_size, input_channels, 1)
            )  # define the filter weights with the shape corresponding
            # the Conv1D layer in keras (kernel_size, input_channels, output_channels)
            indices_ = np.arange(kernel_size)

            filter_[indices_ % 2 == 0] *= -1  # formula of increasing detection filter

            if not self.use_litemv:
                # Create a Conv1D layer with non trainable option and no
                # biases and set the filter weights that were calculated in the
                # line above as the initialization

                conv = tf.keras.layers.Conv1D(
                    filters=1,
                    kernel_size=kernel_size,
                    padding="same",
                    use_bias=False,
                    kernel_initializer=tf.keras.initializers.Constant(filter_.tolist()),
                    trainable=False,
                    name="hybrid-increasse-"
                    + str(self.keep_track)
                    + "-"
                    + str(kernel_size),
                )(input_tensor)
            else:
                # Create a DepthwiseConv1D layer with non trainable option and no
                # biases and set the filter weights that were calculated in the
                # line above as the initialization

                conv = tf.keras.layers.DepthwiseConv1D(
                    kernel_size=kernel_size,
                    padding="same",
                    use_bias=False,
                    depthwise_initializer=tf.keras.initializers.Constant(
                        filter_.tolist()
                    ),
                    trainable=False,
                    name="hybrid-increasse-"
                    + str(self.keep_track)
                    + "-"
                    + str(kernel_size),
                )(input_tensor)

            conv_list.append(conv)  # add the conv layer to the list

            self.keep_track += 1

        # for decreasing detection filters

        for kernel_size in kernel_sizes:
            filter_ = np.ones(
                shape=(kernel_size, input_channels, 1)
            )  # define the filter weights with the shape
            # corresponding the Conv1D layer in keras
            # (kernel_size, input_channels, output_channels)
            indices_ = np.arange(kernel_size)

            filter_[indices_ % 2 > 0] *= -1  # formula of decreasing detection filter

            if not self.use_litemv:
                # Create a Conv1D layer with non trainable option
                # and no biases and set the filter weights that were
                # calculated in the line above as the initialization

                conv = tf.keras.layers.Conv1D(
                    filters=1,
                    kernel_size=kernel_size,
                    padding="same",
                    use_bias=False,
                    kernel_initializer=tf.keras.initializers.Constant(filter_.tolist()),
                    trainable=False,
                    name="hybrid-decrease-"
                    + str(self.keep_track)
                    + "-"
                    + str(kernel_size),
                )(input_tensor)
            else:
                # Create a DepthwiseConv1D layer with non trainable option
                # and no biases and set the filter weights that were
                # calculated in the line above as the initialization

                conv = tf.keras.layers.DepthwiseConv1D(
                    kernel_size=kernel_size,
                    padding="same",
                    use_bias=False,
                    depthwise_initializer=tf.keras.initializers.Constant(
                        filter_.tolist()
                    ),
                    trainable=False,
                    name="hybrid-decrease-"
                    + str(self.keep_track)
                    + "-"
                    + str(kernel_size),
                )(input_tensor)

            conv_list.append(conv)  # add the conv layer to the list

            self.keep_track += 1

        # for peak detection filters

        for kernel_size in kernel_sizes[1:]:
            filter_ = np.zeros(
                shape=(kernel_size + kernel_size // 2, input_channels, 1)
            )

            xmesh = np.linspace(start=0, stop=1, num=kernel_size // 4 + 1)[1:].reshape(
                (-1, 1, 1)
            )

            # see utils.custom_filters.py to understand the formulas below

            filter_left = xmesh**2
            filter_right = filter_left[::-1]

            filter_[0 : kernel_size // 4] = -filter_left
            filter_[kernel_size // 4 : kernel_size // 2] = -filter_right
            filter_[kernel_size // 2 : 3 * kernel_size // 4] = 2 * filter_left
            filter_[3 * kernel_size // 4 : kernel_size] = 2 * filter_right
            filter_[kernel_size : 5 * kernel_size // 4] = -filter_left
            filter_[5 * kernel_size // 4 :] = -filter_right

            if not self.use_litemv:
                # Create a Conv1D layer with non trainable option and
                # no biases and set the filter weights that were
                # calculated in the line above as the initialization

                conv = tf.keras.layers.Conv1D(
                    filters=1,
                    kernel_size=kernel_size + kernel_size // 2,
                    padding="same",
                    use_bias=False,
                    kernel_initializer=tf.keras.initializers.Constant(filter_.tolist()),
                    trainable=False,
                    name="hybrid-peeks-"
                    + str(self.keep_track)
                    + "-"
                    + str(kernel_size),
                )(input_tensor)
            else:
                # Create a DepthwiseConv1D layer with non trainable option and
                # no biases and set the filter weights that were
                # calculated in the line above as the initialization

                conv = tf.keras.layers.DepthwiseConv1D(
                    kernel_size=kernel_size + kernel_size // 2,
                    padding="same",
                    use_bias=False,
                    depthwise_initializer=tf.keras.initializers.Constant(
                        filter_.tolist()
                    ),
                    trainable=False,
                    name="hybrid-peeks-"
                    + str(self.keep_track)
                    + "-"
                    + str(kernel_size),
                )(input_tensor)

            conv_list.append(conv)  # add the conv layer to the list

            self.keep_track += 1

        hybrid_layer = tf.keras.layers.Concatenate(axis=2)(
            conv_list
        )  # concantenate all convolution layers
        hybrid_layer = tf.keras.layers.Activation(activation="relu")(
            hybrid_layer
        )  # apply activation ReLU

        return hybrid_layer

    def _inception_module(
        self,
        input_tensor,
        dilation_rate,
        stride=1,
        activation="linear",
        use_custom_filters=True,
        use_multiplexing=True,
    ):
        import tensorflow as tf

        input_inception = input_tensor

        if not use_multiplexing:
            n_convs = 1
            n_filters = self.n_filters * 3

        else:
            n_convs = 3
            n_filters = self.n_filters

        kernel_size_s = [self.kernel_size // (2**i) for i in range(n_convs)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            if not self.use_litemv:
                conv_list.append(
                    tf.keras.layers.Conv1D(
                        filters=n_filters,
                        kernel_size=kernel_size_s[i],
                        strides=stride,
                        padding="same",
                        dilation_rate=dilation_rate,
                        activation=activation,
                        use_bias=False,
                    )(input_inception)
                )
            else:
                conv_list.append(
                    tf.keras.layers.SeparableConv1D(
                        filters=n_filters,
                        kernel_size=kernel_size_s[i],
                        strides=stride,
                        padding="same",
                        dilation_rate=dilation_rate,
                        activation=activation,
                        use_bias=False,
                    )(input_inception)
                )

        if use_custom_filters:
            hybrid_layer = self.hybrid_layer(
                input_tensor=input_tensor, input_channels=input_tensor.shape[-1]
            )
            conv_list.append(hybrid_layer)

        if len(conv_list) > 1:
            x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        else:
            x = conv_list[0]

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation="relu")(x)

        return x

    def _fcn_module(
        self,
        input_tensor,
        kernel_size=20,
        dilation_rate=2,
        n_filters=32,
        stride=1,
        activation="relu",
    ):
        import tensorflow as tf

        x = tf.keras.layers.SeparableConv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding="same",
            strides=stride,
            dilation_rate=dilation_rate,
            use_bias=False,
        )(input_tensor)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation=activation)(x)

        return x

    def build_network(self, input_shape, **kwargs):
        """
        Construct a network and return its input and output layers.

        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        import tensorflow as tf

        input_layer = tf.keras.layers.Input(input_shape)

        inception = self._inception_module(
            input_tensor=input_layer,
            dilation_rate=1,
            use_custom_filters=True,
            use_multiplexing=True,
        )

        _kernel_size = self.kernel_size // 2

        input_tensor = inception

        dilation_rate = 1

        for i in range(2):
            dilation_rate = 2 ** (i + 1)

            x = self._fcn_module(
                input_tensor=input_tensor,
                kernel_size=_kernel_size // (2**i),
                n_filters=self.n_filters,
                dilation_rate=dilation_rate,
            )

            input_tensor = x

        gap = tf.keras.layers.GlobalAveragePooling1D()(x)

        return input_layer, gap
