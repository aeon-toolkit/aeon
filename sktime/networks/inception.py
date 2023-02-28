# -*- coding: utf-8 -*-
"""Inception Time Classifier."""
__author__ = ["James-Large, Withington, TonyBagnall"]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class InceptionNetwork(BaseDeepNetwork):
    """InceptionTime Network.

        :param nb_filters: int,
        :param use_residual: boolean,
        :param use_bottleneck: boolean,
        :param depth: int
        :param kernel_size: int, specifying the length of the 1D convolution
         window
        :param bottleneck_size: int,
        :param random_state: int, seed to any needed random actions
    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/
    inception.py

    Network originally defined in:

    @article{IsmailFawaz2019inceptionTime, Title                    = {
    InceptionTime: Finding AlexNet for Time Series Classification}, Author
                    = {Ismail Fawaz, Hassan and Lucas, Benjamin and
                    Forestier, Germain and Pelletier, Charlotte and Schmidt,
                    Daniel F. and Weber, Jonathan and Webb, Geoffrey I. and
                    Idoumghar, Lhassane and Muller, Pierre-Alain and
                    Petitjean, François}, journal                  = {
                    ArXiv}, Year                     = {2019} }
    """

    def __init__(
        self,
        nb_filters=32,
        nb_conv_per_layer=3,
        kernel_size=40,
        use_max_pooling=True,
        max_pool_size=3,
        strides=1,
        dilation_rate=1,
        padding='same',
        activation='relu',
        use_bias=False,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")

        if isinstance(nb_filters, list):
            self.nb_filters = nb_filters
        else:
            self.nb_filters = [nb_filters] * depth
        
        if isinstance(kernel_size, list):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = [kernel_size] * depth
        
        if isinstance(nb_conv_per_layer, list):
            self.nb_conv_per_layer = nb_conv_per_layer
        else:
            self.nb_conv_per_layer = [nb_conv_per_layer] * depth
        
        if isinstance(strides, list):
            self.strides = strides
        else:
            self.strides = [strides] * depth
        
        if isinstance(dilation_rate, list):
            self.dilation_rate = dilation_rate
        else:
            self.dilation_rate = [dilation_rate] * depth
        
        if isinstance(padding, list):
            self.padding = padding
        else:
            self.padding = [padding] * depth
        
        if isinstance(activation, list):
            self.activation = activation
        else:
            self.activation = [activation] * depth
        
        if isinstance(use_max_pooling, list):
            self.use_max_pooling = use_max_pooling
        else:
            self.use_max_pooling = [use_max_pooling] * depth
        
        if isinstance(max_pool_size, list):
            self.max_pool_size = max_pool_size
        else:
            self.max_pool_size = [max_pool_size] * depth
        
        if isinstance(use_bias, list):
            self.use_bias = use_bias
        else:
            self.use_bias = [use_bias] * depth

        assert(len(self.nb_filters) == depth)
        assert(len(self.kernel_size) == depth)
        assert(len(self.nb_conv_per_layer) == depth)
        assert(len(self.strides) == depth)
        assert(len(self.dilation_rate) == depth)
        assert(len(self.padding) == depth)
        assert(len(self.activation) == depth)
        assert(len(self.use_max_pooling) == depth)
        assert(len(self.max_pool_size) == depth)
        assert(len(self.use_bias) == depth)

        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.bottleneck_size = bottleneck_size

        self.random_state = random_state

        super(InceptionNetwork, self).__init__()

    def _inception_module(self, input_tensor, nb_filters=32, dilation_rate=1, padding='same',
                                strides=1, activation="relu", use_bias=False, kernel_size=40,
                                nb_conv_per_layer=3, use_max_pooling=True,
                                max_pool_size=3):

        import tensorflow as tf

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = tf.keras.layers.Conv1D(
                filters=self.bottleneck_size,
                kernel_size=1,
                padding=padding,
                activation='linear',
                use_bias=use_bias,
            )(input_tensor)
        else:
            input_inception = input_tensor

        kernel_size_s = [kernel_size // (2**i) for i in range(nb_conv_per_layer)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(
                tf.keras.layers.Conv1D(
                    filters=nb_filters,
                    kernel_size=kernel_size_s[i],
                    strides=strides,
                    dilation_rate=dilation_rate,
                    padding=padding,
                    activation='linear',
                    use_bias=use_bias,
                )(input_inception)
            )
        
        if use_max_pooling:

            max_pool_1 = tf.keras.layers.MaxPool1D(
                pool_size=max_pool_size, strides=strides, padding=padding
            )(input_tensor)

            conv_max_pool = tf.keras.layers.Conv1D(
                filters=nb_filters,
                kernel_size=1,
                padding=padding,
                activation='linear',
                use_bias=use_bias,
            )(max_pool_1)

            conv_list.append(conv_max_pool)

        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation=activation)(x)

        return x

    def _shortcut_layer(self, input_tensor, out_tensor, padding='same', use_bias=False):
        
        import tensorflow as tf

        shortcut_y = tf.keras.layers.Conv1D(
            filters=int(out_tensor.shape[-1]),
            kernel_size=1,
            padding=padding,
            use_bias=use_bias,
        )(input_tensor)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

        x = tf.keras.layers.Add()([shortcut_y, out_tensor])
        x = tf.keras.layers.Activation("relu")(x)
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
        # not sure of the whole padding thing

        import tensorflow as tf
        
        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x,
                                nb_filters=self.nb_filters[d],
                                dilation_rate=self.dilation_rate[d],
                                kernel_size=self.kernel_size[d],
                                padding=self.padding[d],
                                strides=self.strides[d],
                                activation=self.activation[d],
                                use_bias=self.use_bias[d],
                                use_max_pooling=self.use_max_pooling[d],
                                max_pool_size=self.max_pool_size[d],
                                nb_conv_per_layer=self.nb_conv_per_layer[d])

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x, padding=self.padding[d])
                input_res = x

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

        return input_layer, gap_layer
