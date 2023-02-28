# -*- coding: utf-8 -*-
"""Inception Time Classifier."""
__author__ = ["James-Large, Withington, TonyBagnall"]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class InceptionNetwork(BaseDeepNetwork):
    """InceptionTime Network.

    
        depth               : int, default = 6,
            the number of inception modules used
        nb_filters          : int or list of int32, default = 32,
            the number of filters used in one inception module, if not a list,
            the same number of filters is used in all inception modules
        nb_conv_per_layer   : int or list of int, default = 3,
            the number of convolution layers in each inception module, if not a list,
            the same number of convolution layers is used in all inception modules
        kernel_size         : int or list of int, default = 40,
            the head kernel size used for each inception module, if not a list,
            the same is used in all inception modules
        use_max_pooling     : bool or list of bool, default = True,
            conditioning wether or not to use max pooling layer in inception modules,if not a list,
            the same is used in all inception modules
        max_pool_size       : int or list of int, default = 3,
            the size of the max pooling layer, if not a list,
            the same is used in all inception modules
        strides             : int or list of int, default = 1,
            the strides of kernels in convolution layers for each inception module, if not a list,
            the same is used in all inception modules
        dilation_rate       : int or list of int, default = 1,
            the dilation rate of convolutions in each inception module, if not a list,
            the same is used in all inception modules
        padding             : str or list of str, default = 'same',
            the type of padding used for convoltuon for each inception module, if not a list,
            the same is used in all inception modules
        activation          : str or list of str, default = 'relu',
            the activation function used in each inception module, if not a list,
            the same is used in all inception modules
        use_bias            : bool or list of bool, default = False,
            conditioning wether or not convolutions should use bias values in each inception
            module, if not a list,
            the same is used in all inception modules
        use_residual        : bool, default = True,
            condition wether or not to use residual connections all over Inception
        use_bottleneck      : bool, default = True,
            confition wether or not to use bottlesnecks all over Inception
        bottleneck_size     : int, default = 32,
            the bottleneck size in case use_bottleneck = True
        use_custom_filters  : bool, default = True,
            condition on wether or not to use custom filters in the first inception module
        random_state        : int, default = 0,

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/
    inception.py

    and

    https://github.com/MSD-IRIMAS/CF-4-TSC/blob/main/classifiers/H_Inception.py
    for the custom filters

    Network originally defined in:

    @article{IsmailFawaz2019inceptionTime, Title                    = {
    InceptionTime: Finding AlexNet for Time Series Classification}, Author
                    = {Ismail Fawaz, Hassan and Lucas, Benjamin and
                    Forestier, Germain and Pelletier, Charlotte and Schmidt,
                    Daniel F. and Weber, Jonathan and Webb, Geoffrey I. and
                    Idoumghar, Lhassane and Muller, Pierre-Alain and
                    Petitjean, François}, journal                  = {
                    ArXiv}, Year                     = {2019} }
    
    Custom filters defined in:

    @inproceedings{ismail-fawaz2022hccf,
    author = {Ismail-Fawaz, Ali and Devanne, Maxime and Weber, Jonathan and Forestier, Germain},
    title = {Deep Learning For Time Series Classification Using New Hand-Crafted Convolution Filters},
    booktitle = {2022 IEEE International Conference on Big Data (IEEE BigData 2022)},
    city = {Osaka},
    country = {Japan},
    pages = {972-981},
    url = {doi.org/10.1109/BigData55660.2022.10020496},
    year = {2022},
    organization = {IEEE}
    }
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
        use_custom_filters=True,
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

        self.use_custom_filters = use_custom_filters

        self.random_state = random_state

        super(InceptionNetwork, self).__init__()
    
    def hybrid_layer(self,input_tensor, input_channels, kernel_sizes=[2,4,8,16,32,64]):

        import numpy as np
        import tensorflow as tf

        self.keep_track = 0
    

        '''
        Function to create the hybrid layer consisting of non trainable Conv1D layers with custom filters.

        Args:

            input_tensor: input tensor
            input_channels : number of input channels, 1 in case of UCR Archive
        '''

        conv_list = []

        # for increasing detection filters

        for kernel_size in kernel_sizes:

            filter_ = np.ones(shape=(kernel_size,input_channels,1)) # define the filter weights with the shape corresponding the Conv1D layer in keras (kernel_size, input_channels, output_channels)
            indices_ = np.arange(kernel_size)

            filter_[indices_ % 2 == 0] *= -1 # formula of increasing detection filter

            # Create a Conv1D layer with non trainable option and no biases and set the filter weights that were calculated in the line above as the initialization

            conv = tf.keras.layers.Conv1D(filters=1,kernel_size=kernel_size,padding='same',
                                          use_bias=False,kernel_initializer=tf.keras.initializers.Constant(filter_),
                                          trainable=False,name='hybrid-increasse-'+str(self.keep_track)+'-'+str(kernel_size))(input_tensor)

            conv_list.append(conv) # add the conv layer to the list

            self.keep_track += 1

        # for decreasing detection filters
        
        for kernel_size in kernel_sizes:

            filter_ = np.ones(shape=(kernel_size,input_channels,1)) # define the filter weights with the shape corresponding the Conv1D layer in keras (kernel_size, input_channels, output_channels)
            indices_ = np.arange(kernel_size)

            filter_[indices_ % 2 > 0] *= -1 # formula of decreasing detection filter

            # Create a Conv1D layer with non trainable option and no biases and set the filter weights that were calculated in the line above as the initialization

            conv = tf.keras.layers.Conv1D(filters=1,kernel_size=kernel_size,padding='same',
                                          use_bias=False,kernel_initializer=tf.keras.initializers.Constant(filter_),
                                          trainable=False,name='hybrid-decrease-'+str(self.keep_track)+'-'+str(kernel_size))(input_tensor)
            
            conv_list.append(conv) # add the conv layer to the list

            self.keep_track += 1

        # for peak detection filters
        
        for kernel_size in kernel_sizes[1:]:

            filter_ = np.zeros(shape=(kernel_size + kernel_size // 2,input_channels,1))

            xmesh = np.linspace(start=0,stop=1,num=kernel_size//4+1)[1:].reshape((-1,1,1))

            # see utils.custom_filters.py to understand the formulas below

            filter_left = xmesh**2
            filter_right = filter_left[::-1]

            filter_[0:kernel_size // 4] = -filter_left
            filter_[kernel_size // 4:kernel_size // 2] = -filter_right
            filter_[kernel_size // 2:3 * kernel_size // 4] = 2 * filter_left
            filter_[3 * kernel_size // 4:kernel_size] = 2 * filter_right
            filter_[kernel_size:5 * kernel_size // 4] = -filter_left
            filter_[5 * kernel_size // 4:] = -filter_right
            
            # Create a Conv1D layer with non trainable option and no biases and set the filter weights that were calculated in the line above as the initialization

            conv = tf.keras.layers.Conv1D(filters=1,kernel_size=kernel_size+kernel_size//2,padding='same',
                                          use_bias=False,kernel_initializer=tf.keras.initializers.Constant(filter_),
                                          trainable=False,name='hybrid-peeks-'+str(self.keep_track)+'-'+str(kernel_size))(input_tensor)

            conv_list.append(conv) # add the conv layer to the list

            self.keep_track += 1

        
        hybrid_layer = tf.keras.layers.Concatenate(axis=2)(conv_list) # concantenate all convolution layers
        hybrid_layer = tf.keras.layers.Activation(activation='relu')(hybrid_layer) # apply activation ReLU

        return hybrid_layer


    def _inception_module(self, input_tensor, nb_filters=32, dilation_rate=1, padding='same',
                                strides=1, activation="relu", use_bias=False, kernel_size=40,
                                nb_conv_per_layer=3, use_max_pooling=True,
                                max_pool_size=3, use_custom_filters=False):

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
        
        if use_custom_filters:

            hybrid_layer = self.hybrid_layer(input_tensor=input_tensor,
                                             input_channels=int(input_tensor.shape[-1]))
            conv_list.append(hybrid_layer)

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

        use_custom_filters = self.use_custom_filters

        for d in range(self.depth):

            if d > 0:
                use_custom_filters = False

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
                                nb_conv_per_layer=self.nb_conv_per_layer[d],
                                use_custom_filters=use_custom_filters)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x, padding=self.padding[d])
                input_res = x

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

        return input_layer, gap_layer
